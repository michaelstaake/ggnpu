#!/usr/bin/env python3
"""
Triton-XDNA kernel compiler for ggnpu.

Compiles @triton.jit kernels to .xclbin + sequence binaries using the
Triton-XDNA compile-only pipeline (no NPU hardware required when
AMD_TRITON_NPU_TARGET is set).

Usage:
    python3 compile_kernels.py --op matmul --profile npu6 --M 64 --N 64 --K 64
    python3 compile_kernels.py --op silu --profile npu6 --N 8192
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
TRANSFORMS_DIR = SCRIPT_DIR / "transforms"

KERNELS = {
    "matmul": {
        "description": "INT8 matrix multiplication",
        "params": ["M", "N", "K"],
        "defaults": {"M": 64, "N": 64, "K": 64},
        "transform": "matmul_aie2p.mlir",
    },
    "rmsnorm": {
        "description": "RMS normalization",
        "params": ["N"],
        "defaults": {"N": 2048},
        "transform": "rmsnorm_aie2p.mlir",
    },
    "softmax": {
        "description": "Softmax activation",
        "params": ["rows", "cols"],
        "defaults": {"rows": 1, "cols": 1024},
        "transform": "softmax_aie2p.mlir",
    },
    "silu": {
        "description": "SiLU/Swish activation",
        "params": ["N"],
        "defaults": {"N": 8192},
        "transform": "silu_aie2p.mlir",
    },
    "rope": {
        "description": "Rotary positional embeddings",
        "params": ["N", "dims"],
        "defaults": {"N": 2048, "dims": 64},
        "transform": "rope_aie2p.mlir",
    },
    "flash_attn": {
        "description": "FlashAttention v1 (decomposed matmul path)",
        "params": ["n_head", "head_dim", "ctx_len"],
        "defaults": {"n_head": 8, "head_dim": 128, "ctx_len": 2048},
        "transform": "flash_attn_aie2p.mlir",
    },
}

# npu4/5/6 (Strix/Krackan) -> Triton target npu2 (AIE2P)
PROFILE_TO_TARGET = {
    "npu4": "npu2",
    "npu5": "npu2",
    "npu6": "npu2",
    "npu1": "npu1",
}


def _valid_xrt_root(path: Path) -> bool:
    return (path / "include/xrt").is_dir() and (path / "lib").is_dir()


def _linker_lib_names() -> tuple[str, ...]:
    return ("libxrt_coreutil.so", "libuuid.so")


def _can_link_lib(lib_dir: Path, base: str) -> bool:
    if not lib_dir.is_dir():
        return False
    unversioned = lib_dir / f"{base}.so"
    if unversioned.exists():
        return True
    return any(lib_dir.glob(f"{base}.so.*"))


def _ensure_dev_symlinks(lib_dir: Path, dest_lib_dir: Path) -> None:
    """Symlink runtime .so.N files and add unversioned .so names for -l flags."""
    dest_lib_dir.mkdir(parents=True, exist_ok=True)
    if not lib_dir.is_dir():
        return

    for lib in sorted(lib_dir.glob("libxrt*.so*")) + sorted(lib_dir.glob("libuuid.so*")):
        link = dest_lib_dir / lib.name
        if not link.exists():
            link.symlink_to(lib.resolve())

    for lib in sorted(dest_lib_dir.glob("*.so.*")):
        base = lib.name.split(".so.", 1)[0] + ".so"
        unversioned = dest_lib_dir / base
        if not unversioned.exists():
            unversioned.symlink_to(lib.name)


def find_xilinx_xrt(repo_root: Path) -> Path | None:
    """Locate an XRT SDK tree (include/xrt + lib) for Triton-XDNA compile-only builds."""
    explicit = os.environ.get("XILINX_XRT", "")
    if explicit:
        root = Path(explicit)
        if _valid_xrt_root(root):
            return root.resolve()

    for candidate in (
        Path("/opt/xilinx/xrt"),
        repo_root / "third_party/xrt-dev/usr",
    ):
        if _valid_xrt_root(candidate):
            return candidate.resolve()

    # Ubuntu libxrt-dev splits headers (/usr/include/xrt) and libs (/usr/lib/...).
    # Triton expects a unified tree — build a shim under build/xrt-sdk-shim.
    usr_inc = Path("/usr/include/xrt")
    if usr_inc.is_dir():
        shim = repo_root / "build" / "xrt-sdk-shim"
        shim_inc = shim / "include" / "xrt"
        shim_lib = shim / "lib"
        shim.mkdir(parents=True, exist_ok=True)
        (shim / "include").mkdir(parents=True, exist_ok=True)
        shim_lib.mkdir(parents=True, exist_ok=True)

        if not shim_inc.exists():
            shim_inc.symlink_to(usr_inc.resolve(), target_is_directory=True)

        lib_dirs = [
            Path("/usr/lib/x86_64-linux-gnu"),
            Path("/usr/lib"),
            Path("/opt/xilinx/xrt/lib"),
        ]
        for lib_dir in lib_dirs:
            _ensure_dev_symlinks(lib_dir, shim_lib)

        if _valid_xrt_root(shim):
            return shim.resolve()

    return None


def linker_lib_search_dirs(xrt_root: Path, repo_root: Path) -> list[Path]:
    dirs: list[Path] = [xrt_root / "lib"]
    for sub in ("lib/x86_64-linux-gnu", "lib"):
        p = xrt_root / sub
        if p.is_dir():
            dirs.append(p)
    uuid_lib = repo_root / "third_party/uuid-dev/usr/lib/x86_64-linux-gnu"
    if uuid_lib.is_dir():
        dirs.append(uuid_lib)
    system_lib = Path("/usr/lib/x86_64-linux-gnu")
    if system_lib.is_dir():
        dirs.append(system_lib)
    seen: set[Path] = set()
    unique: list[Path] = []
    for d in dirs:
        resolved = d.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(d)
    return unique


def check_linker_prereqs(xrt_root: Path, repo_root: Path) -> str | None:
    """Return an error message when g++ cannot link the Triton compile-only launcher."""
    lib_dirs = linker_lib_search_dirs(xrt_root, repo_root)
    required = ("libxrt_coreutil", "libuuid")
    missing = [base for base in required if not any(_can_link_lib(d, base) for d in lib_dirs)]
    if not missing:
        return None
    return (
        "Missing linker libraries for Triton compile-only launcher: "
        + ", ".join(f"{name}.so" for name in missing)
        + ".\n"
        "  Install development packages (headers + linker names):\n"
        "    sudo apt install libxrt-dev uuid-dev\n"
        "  Or extract debs without sudo:\n"
        "    bash scripts/fetch-xrt-dev.sh\n"
        "    export XILINX_XRT=$PWD/third_party/xrt-dev/usr\n"
        "  You also need libxrt2 runtime: sudo apt install libxrt2"
    )


def setup_compile_env(repo_root: Path) -> dict[str, str]:
    """Return environment variables needed for Triton-XDNA compile-only builds."""
    env = os.environ.copy()

    xrt_root = find_xilinx_xrt(repo_root)
    if xrt_root is None:
        raise RuntimeError(
            "XRT development files not found. Triton-XDNA needs include/xrt and libxrt.\n"
            "  Install: sudo apt install libxrt-dev uuid-dev\n"
            "  Or set:  export XILINX_XRT=/opt/xilinx/xrt\n"
            "  Or extract headers to: third_party/xrt-dev/usr/"
        )
    env["XILINX_XRT"] = str(xrt_root)

    linker_err = check_linker_prereqs(xrt_root, repo_root)
    if linker_err:
        raise RuntimeError(linker_err)

    include_paths = [str(xrt_root / "include")]
    uuid_inc = repo_root / "third_party/uuid-dev/usr/include"
    if uuid_inc.is_dir():
        include_paths.append(str(uuid_inc))
    elif Path("/usr/include/uuid").is_dir():
        include_paths.append("/usr/include")
    env["CPLUS_INCLUDE_PATH"] = ":".join(include_paths + ([env["CPLUS_INCLUDE_PATH"]] if env.get("CPLUS_INCLUDE_PATH") else []))

    lib_paths = [str(d) for d in linker_lib_search_dirs(xrt_root, repo_root)]
    if lib_paths:
        env["LIBRARY_PATH"] = ":".join(lib_paths + ([env["LIBRARY_PATH"]] if env.get("LIBRARY_PATH") else []))

    venv = os.environ.get("VIRTUAL_ENV") or os.environ.get("GGNPU_TRITON_VENV")
    if not venv:
        for candidate in (repo_root / ".venv-triton", Path.home() / "triton-env"):
            if (candidate / "bin/python").exists():
                venv = str(candidate)
                break

    if venv:
        peano = Path(venv) / "lib"
        for site in peano.glob("python*/site-packages/llvm-aie"):
            env["PEANO_INSTALL_DIR"] = str(site)
            env["PATH"] = f"{site}/bin:{env.get('PATH', '')}"
            mlir_aie_bin = site.parent / "mlir_aie/bin"
            if mlir_aie_bin.is_dir():
                env["PATH"] = f"{mlir_aie_bin}:{env['PATH']}"
            break

    return env


def profile_to_target(profile: str) -> str:
    return PROFILE_TO_TARGET.get(profile, "npu2")


def transform_path(op: str) -> Path:
    script = KERNELS[op]["transform"]
    path = TRANSFORMS_DIR / script
    if not path.is_file():
        raise FileNotFoundError(f"Missing transform script for {op}: {path}")
    return path.resolve()


def build_kernel_script(op: str, params: dict) -> str:
    """Generate a Python source file that compiles one kernel (Triton requires a file)."""
    p = params

    if op == "matmul":
        launch = textwrap.dedent(f"""
            M, N, K = {p.get("M", 64)}, {p.get("N", 64)}, {p.get("K", 64)}
            a = torch.randint(-8, 8, (M, K), dtype=torch.int8)
            b = torch.randint(-8, 8, (K, N), dtype=torch.int8)
            c = torch.zeros((M, N), dtype=torch.int32)
            grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]), triton.cdiv(N, META["BLOCK_SIZE_N"]))
            compiled = matmul_kernel[grid](
                a, b, c, M, N, K,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
            )
        """)
        kernel_src = textwrap.dedent("""
            @triton.jit
            def matmul_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                stride_am: tl.constexpr, stride_ak: tl.constexpr,
                stride_bk: tl.constexpr, stride_bn: tl.constexpr,
                stride_cm: tl.constexpr, stride_cn: tl.constexpr,
                BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
                pid_m = tl.program_id(0)
                pid_n = tl.program_id(1)
                offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                offs_k = tl.arange(0, BLOCK_SIZE_K)
                a_block = tl.load(A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
                b_block = tl.load(B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
                c_block = tl.dot(a_block, b_block)
                tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c_block)
        """)

    elif op == "rmsnorm":
        launch = textwrap.dedent(f"""
            N = {p.get("N", 2048)}
            x = torch.randn(N, dtype=torch.float32)
            y = torch.empty(N, dtype=torch.float32)
            grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
            compiled = rmsnorm_kernel[grid](x, y, N, eps=1e-5, BLOCK_SIZE=1024)
        """)
        kernel_src = textwrap.dedent("""
            @triton.jit
            def rmsnorm_kernel(input_ptr, output_ptr, N: tl.constexpr, eps: tl.constexpr, BLOCK_SIZE: tl.constexpr):
                pid = tl.program_id(0)
                offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask = offset < N
                input_block = tl.load(input_ptr + offset, mask=mask)
                mean_square = tl.sum(input_block * input_block, axis=0) / N
                rms = tl.sqrt(mean_square + eps)
                output_block = input_block / rms
                tl.store(output_ptr + offset, output_block, mask=mask)
        """)

    elif op == "softmax":
        launch = textwrap.dedent(f"""
            rows, cols = {p.get("rows", 1)}, {p.get("cols", 1024)}
            x = torch.randn(rows, cols, dtype=torch.float32)
            y = torch.empty(rows, cols, dtype=torch.float32)
            grid = lambda META: (rows,)
            compiled = softmax_kernel[grid](x, y, rows, cols, cols, cols, BLOCK_SIZE=1024)
        """)
        kernel_src = textwrap.dedent("""
            @triton.jit
            def softmax_kernel(input_ptr, output_ptr, rows: tl.constexpr, cols: tl.constexpr,
                stride_row_in: tl.constexpr, stride_row_out: tl.constexpr, BLOCK_SIZE: tl.constexpr):
                row = tl.program_id(0)
                offset = tl.arange(0, BLOCK_SIZE)
                mask = offset < cols
                input_block = tl.load(input_ptr + row * stride_row_in + offset, mask=mask)
                max_val = tl.max(input_block, axis=0)
                exp_vals = tl.exp(input_block - max_val)
                sum_vals = tl.sum(exp_vals, axis=0)
                output_block = exp_vals / sum_vals
                tl.store(output_ptr + row * stride_row_out + offset, output_block, mask=mask)
        """)

    elif op == "silu":
        launch = textwrap.dedent(f"""
            N = {p.get("N", 8192)}
            x = torch.randn(N, dtype=torch.bfloat16)
            y = torch.empty(N, dtype=torch.bfloat16)
            grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
            compiled = silu_kernel[grid](x, y, N, BLOCK_SIZE=1024)
        """)
        kernel_src = textwrap.dedent("""
            @triton.jit
            def silu_kernel(X, Y, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
                pid = tl.program_id(0)
                offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                x = tl.load(X + offsets)
                x_f32 = x.to(tl.float32)
                sig = tl.sigmoid(x_f32)
                y = (x_f32 * sig).to(x.dtype)
                tl.store(Y + offsets, y)
        """)

    elif op == "rope":
        launch = textwrap.dedent(f"""
            N, dims = {p.get("N", 2048)}, {p.get("dims", 64)}
            x = torch.randn(N, dtype=torch.float32)
            y = torch.empty(N, dtype=torch.float32)
            grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
            compiled = rope_kernel[grid](x, y, N, dims, 0, 10000.0, 1.0, BLOCK_SIZE=1024)
        """)
        kernel_src = textwrap.dedent("""
            @triton.jit
            def rope_kernel(input_ptr, output_ptr, N: tl.constexpr, dims: tl.constexpr,
                offset: tl.constexpr, freq_base: tl.constexpr, freq_scale: tl.constexpr,
                BLOCK_SIZE: tl.constexpr):
                pid = tl.program_id(0)
                idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask = idx < N
                x = tl.load(input_ptr + idx, mask=mask, other=0.0)
                pair_idx = idx // 2
                is_odd = (idx % 2) != 0
                ratio = 1.0 / tl.exp2((2.0 * pair_idx.to(tl.float32)) * (3.32192809489 / dims))
                angle = offset * ratio * freq_scale
                cos_val = tl.cos(angle)
                sin_val = tl.sin(angle)
                x_pair = tl.load(input_ptr + (pair_idx * 2 + (1 - is_odd.to(tl.int32))), mask=mask, other=0.0)
                out = tl.where(is_odd, x * cos_val + x_pair * sin_val, x * cos_val - x_pair * sin_val)
                tl.store(output_ptr + idx, out, mask=mask)
        """)

    elif op == "flash_attn":
        n_head = p.get("n_head", 8)
        head_dim = p.get("head_dim", 128)
        ctx_len = p.get("ctx_len", 2048)
        launch = textwrap.dedent(f"""
            n_head, head_dim, ctx_len = {n_head}, {head_dim}, {ctx_len}
            Q = torch.randn(n_head, head_dim, dtype=torch.float32)
            K = torch.randn(ctx_len, head_dim, dtype=torch.float32)
            V = torch.randn(ctx_len, head_dim, dtype=torch.float32)
            O = torch.empty(n_head, head_dim, dtype=torch.float32)
            grid = lambda META: (n_head,)
            compiled = flash_attn_kernel[grid](
                Q, K, V, O, head_dim, ctx_len,
                head_dim, ctx_len * head_dim, ctx_len * head_dim, head_dim,
                BLOCK_K=64,
            )
        """)
        kernel_src = textwrap.dedent("""
            @triton.jit
            def flash_attn_kernel(Q, K, V, output, head_dim: tl.constexpr, ctx_len: tl.constexpr,
                stride_qh: tl.constexpr, stride_kh: tl.constexpr, stride_vh: tl.constexpr, stride_oh: tl.constexpr,
                BLOCK_K: tl.constexpr):
                pid = tl.program_id(0)
                q_offs = tl.arange(0, head_dim)
                q = tl.load(Q + pid * stride_qh + q_offs)
                m = tl.full([], -1e9, tl.float32)
                l = tl.full([], 0.0, tl.float32)
                acc = tl.zeros([head_dim], dtype=tl.float32)
                for start in range(0, ctx_len, BLOCK_K):
                    k_idx = start + tl.arange(0, BLOCK_K)
                    mask = k_idx < ctx_len
                    k_row = tl.load(K + k_idx[:, None] * head_dim + q_offs[None, :], mask=mask[:, None], other=0.0)
                    scores = tl.sum(q[None, :] * k_row, axis=1)
                    scores = tl.where(mask, scores, -1e9)
                    block_max = tl.max(scores, axis=0)
                    new_m = tl.maximum(m, block_max)
                    alpha = tl.exp(m - new_m)
                    beta = tl.exp(scores - new_m)
                    v_row = tl.load(V + k_idx[:, None] * head_dim + q_offs[None, :], mask=mask[:, None], other=0.0)
                    acc = acc * alpha + tl.sum(beta[:, None] * v_row, axis=0)
                    l = l * alpha + tl.sum(beta, axis=0)
                    m = new_m
                out = acc / l
                tl.store(output + pid * stride_oh + q_offs, out)
        """)
    else:
        raise ValueError(f"Unknown op: {op}")

    launch_body = textwrap.indent(launch.strip(), "    ")
    return (
        "import os\n"
        "import torch\n"
        "import triton\n"
        "import triton.language as tl\n"
        "from triton.backends.amd_triton_npu.driver import NPUDriver, get_npu_cache_dir\n"
        "from triton.backends.amd_triton_npu.config import npu_config\n"
        "\n"
        "triton.runtime.driver.set_active(NPUDriver())\n"
        "npu_config.compile_only = True\n"
        "npu_config.target = os.environ.get('AMD_TRITON_NPU_TARGET', 'npu2')\n"
        "npu_config.output_format = os.environ.get('AMD_TRITON_NPU_OUTPUT_FORMAT', 'xclbin')\n"
        "npu_config.air_project_path = os.environ.get('AMD_TRITON_NPU_AIR_PROJECT_PATH', './air_project')\n"
        "\n"
        f"{kernel_src.strip()}\n"
        "\n"
        "def main():\n"
        f"{launch_body}\n"
        "    cache = get_npu_cache_dir(compiled)\n"
        "    print('NPU_CACHE_DIR=' + (cache or ''))\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )


def run_aircc_fallback(air_project: Path, env: dict[str, str]) -> None:
    """Run aircc directly when Triton exits before writing aie.xclbin."""
    mlir = air_project / "asm_air_output.mlir"
    if not mlir.is_file():
        nested = air_project / "air_project" / "asm_air_output.mlir"
        if nested.is_file():
            mlir = nested
        else:
            return

    aircc = None
    for site in Path(sys.executable).parent.parent.glob("lib/python*/site-packages/mlir_air/bin/aircc"):
        if site.is_file():
            aircc = site
            break
    if not aircc:
        return

    insts = air_project / "insts.bin"
    xclbin = air_project / "aie.xclbin"
    cmd = [
        str(aircc),
        "--device", env.get("AMD_TRITON_NPU_TARGET", "npu2"),
        "--no-xchesscc", "--no-xbridge",
        "--output-format", "xclbin",
        "-i", str(insts),
        "-o", str(xclbin),
        "--air-runtime-loop-tiling-sizes=4",
        "--air-runtime-loop-tiling-sizes=4",
        "--stack-size", "2048",
        str(mlir),
    ]
    subprocess.run(cmd, env=env, cwd=str(mlir.parent), capture_output=True, timeout=600)


def find_artifacts(search_roots: list[Path]) -> tuple[Path | None, Path | None]:
    """Locate aie.xclbin and insts.bin under Triton/air project directories."""
    xclbin = None
    insts = None
    for root in search_roots:
        if not root or not root.exists():
            continue
        for hit in root.rglob("aie.xclbin"):
            if hit.is_file() and hit.stat().st_size > 0:
                xclbin = hit
                break
        for hit in root.rglob("insts.bin"):
            if hit.is_file() and hit.stat().st_size > 0:
                insts = hit
        if xclbin and insts:
            break
    return xclbin, insts


def compile_kernel(op: str, profile: str, params: dict, output_dir: Path, repo_root: Path) -> bool:
    if op not in KERNELS:
        print(f"Error: Unknown kernel '{op}'")
        return False

    target = profile_to_target(profile)
    transform = transform_path(op)
    build_dir = repo_root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    air_project = Path(tempfile.mkdtemp(prefix=f"ggnpu_{op}_{profile}_", dir=str(build_dir)))

    try:
        env = setup_compile_env(repo_root)
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        return False
    env["AMD_TRITON_NPU_TARGET"] = target
    env["AMD_TRITON_NPU_OUTPUT_FORMAT"] = "xclbin"
    env["AMD_TRITON_NPU_COMPILE_ONLY"] = "1"
    env["AIR_TRANSFORM_TILING_SCRIPT"] = str(transform)
    env["AMD_TRITON_NPU_AIR_PROJECT_PATH"] = str(air_project)

    print(f"Compiling {op}: {KERNELS[op]['description']}")
    print(f"  profile={profile} target={target}")
    print(f"  transform={transform.name}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=f"_{op}_compile.py", delete=False) as f:
        f.write(build_kernel_script(op, params))
        compile_script = f.name

    npu_cache_dir = None
    try:
        print("  Running Triton-XDNA compiler...")
        result = subprocess.run(
            [sys.executable, compile_script],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
            cwd=str(repo_root),
        )
        if result.stdout:
            for line in result.stdout.splitlines():
                if line.startswith("NPU_CACHE_DIR="):
                    npu_cache_dir = line.split("=", 1)[1].strip() or None
                elif line.strip():
                    print(f"  {line}")
        if result.returncode != 0:
            print("  Triton compile returned non-zero; trying aircc fallback...")
            if result.stderr:
                print(result.stderr[-2000:])
            run_aircc_fallback(air_project, env)
    except subprocess.TimeoutExpired:
        print("  ERROR: Compilation timed out")
        return False
    finally:
        os.unlink(compile_script)

    search_roots = [Path(npu_cache_dir) if npu_cache_dir else None, air_project, air_project / "air_project"]
    xclbin_src, insts_src = find_artifacts([r for r in search_roots if r])

    if not xclbin_src:
        print("  ERROR: aie.xclbin not found after compile")
        print("  Ensure XRT headers (libxrt-dev) and PEANO (llvm-aie wheel) are available.")
        print("  See docs/host-setup-guide.md Step 8.")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    xclbin_dst = output_dir / f"{op}_{profile}.xclbin"
    shutil.copy2(xclbin_src, xclbin_dst)

    if insts_src:
        insts_dst = output_dir / f"{op}_{profile}_sequence.bin"
        shutil.copy2(insts_src, insts_dst)

    size = xclbin_dst.stat().st_size
    print(f"  Wrote {xclbin_dst} ({size} bytes)")
    if insts_src:
        print(f"  Wrote {output_dir / f'{op}_{profile}_sequence.bin'}")

    try:
        shutil.rmtree(air_project, ignore_errors=True)
    except OSError:
        pass
    return True


def main():
    parser = argparse.ArgumentParser(description="Compile ggnpu Triton kernels to xclbin")
    parser.add_argument("--op", required=True, choices=list(KERNELS.keys()))
    parser.add_argument("--profile", default="npu6", choices=["npu4", "npu5", "npu6", "npu1"])
    parser.add_argument("--output-dir", default="~/.cache/ggnpu/xclbin")
    parser.add_argument("--M", type=int)
    parser.add_argument("--N", type=int)
    parser.add_argument("--K", type=int)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--cols", type=int)
    parser.add_argument("--dims", type=int)
    parser.add_argument("--n_head", type=int)
    parser.add_argument("--head_dim", type=int)
    parser.add_argument("--ctx_len", type=int)
    args = parser.parse_args()

    output_dir = Path(os.path.expanduser(args.output_dir))
    defaults = KERNELS[args.op]["defaults"].copy()
    for key in KERNELS[args.op]["params"]:
        val = getattr(args, key, None)
        if val is not None:
            defaults[key] = val

    ok = compile_kernel(args.op, args.profile, defaults, output_dir, REPO_ROOT)
    if ok:
        print(f"\nKernel compiled: {output_dir / f'{args.op}_{args.profile}.xclbin'}")
    else:
        print("\nKernel compilation FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
