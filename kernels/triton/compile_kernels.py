#!/usr/bin/env python3
"""
Triton-XDNA kernel compiler for ggnpu.

Compiles @triton.jit kernels to .xclbin + sequence binaries using the
Triton-XDNA compile-only pipeline (no NPU hardware required when
AMD_TRITON_NPU_TARGET is set).

Usage:
    python3 compile_kernels.py --op matmul --profile npu6 --M 256 --N 256 --K 256
    python3 compile_kernels.py --op silu --profile npu6 --N 8192

Kernel sizes must match the fixed tiling in the transform scripts under
transforms/ (e.g. matmul tiles 256-blocks to 64x64x64 L1 tiles).
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
        # Transform script tiles 256x256x256 blocks down to 64x64x64 L1 tiles;
        # launching at 64x64x64 leaves no L3->L2 copy loops to tile and the
        # script fails with empty transform handles.
        "description": "INT8 matrix multiplication",
        "params": ["M", "N", "K"],
        "defaults": {"M": 256, "N": 256, "K": 256},
        "transform": "matmul_aie2p.mlir",
    },
    "matmul_small_m": {
        # Decode-optimized INT8 matmul: small M tile (16) instead of 256, so a
        # single-token (M=1) decode matmul does not pay for 256 wasted output
        # rows. Same INT8 datapath/transform as `matmul`; only BLOCK_SIZE_M
        # shrinks. N/K stay 256 so the L3->L2 copy loops still tile (the M dim
        # is not what PHASE 1 tiles). Probe toward usable decode throughput.
        "description": "INT8 matmul, small M tile (decode GEMV)",
        "params": ["M", "N", "K"],
        "defaults": {"M": 16, "N": 256, "K": 256},
        "transform": "matmul_small_m_aie2p.mlir",
    },
    "matmul_bf16": {
        # bf16 A@B -> f32 C, reusing the int8 matmul transform with the contract
        # casts changed to bf16/f32 (16-bit packing is shared). Feasibility probe
        # for decomposed NPU attention (QK/AV as bf16 GEMMs).
        "description": "BF16 matrix multiply (tl.dot) [WIP]",
        "params": ["M", "N", "K"],
        "defaults": {"M": 256, "N": 256, "K": 256},
        "transform": "matmul_bf16_aie2p.mlir",
        "experimental": (
            "WIP: bf16 datapath probe for decomposed NPU attention. Reuses the "
            "int8 matmul transform with contract casts retargeted to bf16/f32."
        ),
    },
    "rmsnorm": {
        # 2D BLOCK_M x N bf16 kernel (upstream rms_norm example); the 1D
        # scalar-sum form is not tileable by the transform script.
        "description": "RMS normalization",
        "params": ["M", "N"],
        "defaults": {"M": 32, "N": 256},
        "transform": "rmsnorm_aie2p.mlir",
    },
    "softmax": {
        # 4 rows per program, bf16 (upstream test_softmax example).
        "description": "Softmax activation",
        "params": ["rows", "cols"],
        "defaults": {"rows": 256, "cols": 256},
        "transform": "softmax_aie2p.mlir",
    },
    "silu": {
        "description": "SiLU/Swish activation",
        "params": ["N"],
        "defaults": {"N": 8192},
        "transform": "silu_aie2p.mlir",
    },
    "rope": {
        "description": "Rotary positional embeddings (pre-deinterleaved even/odd + precomputed cos/sin)",
        "params": ["n_pairs"],
        "defaults": {"n_pairs": 32},
        "transform": "rope_aie2p.mlir",
    },
    "attn_qk": {
        # QK^T scores: one reduction over head_dim (rmsnorm-shaped). First piece of
        # the decomposed NPU attention (QK matvec -> softmax -> AV matvec) that
        # replaces the un-herd-able fused flash_attn. WIP: with the rmsnorm
        # transform this reaches AIE herd placement (the fused kernel's wall) but
        # fails later in air-dma-to-channel on the two-input (Q broadcast + K) DMA
        # pattern. Needs a dedicated transform handling both reduction operands.
        "description": "Attention QK^T scores (single-reduction matvec) [WIP]",
        "params": ["ctx_len", "head_dim"],
        "defaults": {"ctx_len": 256, "head_dim": 64},
        "transform": "attn_qk_aie2p.mlir",
        "experimental": (
            "WIP toward decomposed NPU attention (QK matvec -> softmax -> AV). "
            "Dedicated transform promotes both reduction inputs (Q broadcast + K) "
            "and reaches AIE herd placement, but QK's 1-D matvec output does not "
            "fit the rmsnorm-derived 2-D pipeline. Needs a GEMV transform or a "
            "bf16 tl.dot matmul. See §7.1 of IMPLEMENTATION.md."
        ),
    },
    "flash_attn": {
        "description": "FlashAttention v1 (online softmax; elementwise decomposition)",
        "params": ["n_head", "head_dim", "ctx_len"],
        "defaults": {"n_head": 8, "head_dim": 128, "ctx_len": 2048},
        "transform": "flash_attn_aie2p.mlir",
        "experimental": (
            "Triton-XDNA default pipeline fails on kernels with multiple tl.sum reductions. "
            "Kernel IR produces linalg.reduce ops that the transform dialect can't handle. "
            "Workaround: use host f32 decomposed path (GGNPU_FLASH_ATTN_FUSED not set). "
            "To fix: need Triton-XDNA support for multi-reduction kernel lowering."
        ),
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


def _is_drvfs(path: Path) -> bool:
    """True when path lives on a Windows mount (WSL drvfs). Symlinks there are unreliable."""
    try:
        return path.resolve().as_posix().startswith("/mnt/")
    except OSError:
        return False


def _is_wsl() -> bool:
    return Path("/proc/version").is_file() and "microsoft" in Path("/proc/version").read_text().lower()


def _shim_root(repo_root: Path) -> Path:
    # WSL users often clone under /mnt/c — keep the shim on the Linux filesystem.
    if _is_drvfs(repo_root) or _is_wsl():
        return Path.home() / ".cache/ggnpu" / "xrt-sdk-shim"
    return repo_root / "build" / "xrt-sdk-shim"


def _linker_lib_names() -> tuple[str, ...]:
    return ("libxrt_coreutil.so", "libuuid.so")


def _can_link_lib(lib_dir: Path, base: str) -> bool:
    if not lib_dir.is_dir():
        return False
    unversioned = lib_dir / f"{base}.so"
    if unversioned.exists():
        return True
    return any(lib_dir.glob(f"{base}.so.*"))


def _materialize_lib(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    shutil.copy2(src.resolve(), dest, follow_symlinks=True)


def _ensure_dev_symlinks(lib_dir: Path, dest_lib_dir: Path, *, materialize: bool) -> None:
    """Populate dest with XRT/uuid libs; copy on WSL drvfs instead of symlinking."""
    dest_lib_dir.mkdir(parents=True, exist_ok=True)
    if not lib_dir.is_dir():
        return

    for lib in sorted(lib_dir.glob("libxrt*.so*")) + sorted(lib_dir.glob("libuuid.so*")):
        dest = dest_lib_dir / lib.name
        if dest.exists():
            continue
        if materialize:
            _materialize_lib(lib, dest)
        else:
            dest.symlink_to(lib.resolve())

    for lib in sorted(dest_lib_dir.glob("*.so.*")):
        base = lib.name.split(".so.", 1)[0] + ".so"
        unversioned = dest_lib_dir / base
        if unversioned.exists():
            continue
        if materialize:
            _materialize_lib(lib, unversioned)
        else:
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
        shim = _shim_root(repo_root)
        shim_inc = shim / "include" / "xrt"
        shim_lib = shim / "lib"
        materialize = _is_drvfs(shim) or _is_drvfs(repo_root) or _is_wsl()
        shim.mkdir(parents=True, exist_ok=True)
        (shim / "include").mkdir(parents=True, exist_ok=True)
        shim_lib.mkdir(parents=True, exist_ok=True)

        if not shim_inc.exists():
            if materialize:
                shutil.copytree(usr_inc, shim_inc, dirs_exist_ok=True, symlinks=True)
            else:
                shim_inc.symlink_to(usr_inc.resolve(), target_is_directory=True)

        lib_dirs = [
            Path("/usr/lib/x86_64-linux-gnu"),
            Path("/usr/lib"),
            Path("/opt/xilinx/xrt/lib"),
        ]
        for lib_dir in lib_dirs:
            _ensure_dev_symlinks(lib_dir, shim_lib, materialize=materialize)

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


def _python_include_dir() -> Path:
    return Path(f"/usr/include/python{sys.version_info.major}.{sys.version_info.minor}")


def _triton_npu_include_dir() -> Path | None:
    for site in Path(sys.executable).parent.parent.glob(
        "lib/python*/site-packages/triton/backends/amd_triton_npu/include"
    ):
        if site.is_dir():
            return site
    return None


def check_python_dev_headers() -> str | None:
    inc = _python_include_dir()
    if inc.is_dir() and (inc / "Python.h").is_file():
        return None
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    return (
        f"Python development headers missing for {sys.executable} ({ver}).\n"
        f"  Expected: {inc}/Python.h\n"
        f"  Install:  sudo apt install python{ver}-dev"
    )


def _include_flags_for_env(env: dict[str, str], xrt_root: Path) -> list[str]:
    flags = [f"-I{xrt_root / 'include'}"]
    extra = env.get("CPLUS_INCLUDE_PATH", "")
    for entry in extra.split(":"):
        # Skip /usr/include: passing it as -I breaks libstdc++ include_next.
        if entry and entry not in (str(xrt_root / "include"), "/usr/include"):
            flags.append(f"-I{entry}")
    return flags


def run_launcher_link_smoke_test(env: dict[str, str], repo_root: Path) -> str | None:
    """Run the same style of g++ link Triton uses for xrt_dispatch.exe."""
    py_err = check_python_dev_headers()
    if py_err:
        return py_err

    xrt_root = Path(env["XILINX_XRT"])
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    triton_inc = _triton_npu_include_dir()
    lib_dirs = linker_lib_search_dirs(xrt_root, repo_root)
    uuid_hdr = repo_root / "third_party/uuid-dev/usr/include/uuid/uuid.h"
    if not uuid_hdr.is_file() and not Path("/usr/include/uuid/uuid.h").is_file():
        return (
            "uuid/uuid.h not found.\n"
            "  Install: sudo apt install uuid-dev\n"
            "  Or run:  bash scripts/fetch-xrt-dev.sh"
        )

    with tempfile.TemporaryDirectory(prefix="ggnpu_xrt_link_") as tmp:
        main_cxx = Path(tmp) / "main.cxx"
        main_cxx.write_text(
            '#include <xrt/xrt_bo.h>\n#include <Python.h>\nint main() { return 0; }\n',
            encoding="utf-8",
        )
        out = Path(tmp) / "xrt_dispatch.exe"
        cmd = [
            "g++",
            "-std=c++23",
            str(main_cxx),
            f"-I{_python_include_dir()}",
        ]
        if triton_inc:
            cmd.append(f"-I{triton_inc}")
        cmd.extend(_include_flags_for_env(env, xrt_root))
        cmd.extend(
            [
                "-shared",
                f"-lpython{py_ver}",
                "-fPIC",
                "-Wall",
                "-luuid",
                "-lxrt_coreutil",
                "-lrt",
                "-lstdc++",
                "-o",
                str(out),
            ]
        )
        for lib_dir in lib_dirs:
            cmd.extend(["-L", str(lib_dir)])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return None
        detail = "\n".join(line for line in (result.stdout + result.stderr).splitlines() if line.strip())
        return detail or "g++ link smoke test failed (no compiler output)"


def extract_compiler_error(stderr: str) -> str:
    """Pull the most useful g++/ld lines out of a long Triton traceback."""
    if not stderr:
        return ""
    lines = stderr.splitlines()
    interesting = [
        line
        for line in lines
        if any(
            token in line
            for token in (
                "error:",
                "fatal error:",
                "cannot find",
                "No such file",
                "collect2:",
                "ld returned",
                "Command '[",
            )
        )
    ]
    if interesting:
        return "\n".join(interesting[-20:])
    return stderr[-4000:]


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

    py_err = check_python_dev_headers()
    if py_err:
        raise RuntimeError(py_err)

    include_paths = [str(xrt_root / "include")]
    uuid_inc = repo_root / "third_party/uuid-dev/usr/include"
    if (uuid_inc / "uuid/uuid.h").is_file():
        include_paths.append(str(uuid_inc))
    elif Path("/usr/include/uuid/uuid.h").is_file():
        # /usr/include is already a default system include dir; adding it to
        # CPLUS_INCLUDE_PATH demotes it to a user dir and breaks libstdc++'s
        # `#include_next <stdlib.h>` (fatal error: stdlib.h: No such file).
        pass
    else:
        raise RuntimeError(
            "uuid/uuid.h not found. XRT headers require uuid-dev.\n"
            "  Install: sudo apt install uuid-dev\n"
            "  Or run:  bash scripts/fetch-xrt-dev.sh"
        )
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

    # xclbinutil is needed by aiecc to package aie.xclbin (XRT tools).
    if not shutil.which("xclbinutil", path=env.get("PATH")):
        for candidate in (
            Path("/opt/xilinx/xrt/bin"),
            repo_root / "third_party/xrt-tools/usr/bin",
            repo_root / "third_party/xrt-dev/usr/bin",
        ):
            if (candidate / "xclbinutil").is_file():
                env["PATH"] = f"{candidate}:{env.get('PATH', '')}"
                break
        else:
            print(
                "  WARNING: xclbinutil not found — aie.xclbin packaging will be skipped.\n"
                "  Install: sudo apt install libxrt-utils libxrt-utils-npu (or xrt-tools)\n"
                "  Or extract without sudo: bash scripts/fetch-xrt-dev.sh --tools"
            )

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
            # Block sizes must be >= 256 so the transform script has L3->L2
            # copy loops to tile (it tiles 256-blocks down to 64x64x64 L1 tiles).
            M, N, K = {p.get("M", 256)}, {p.get("N", 256)}, {p.get("K", 256)}
            a = torch.randint(-8, 8, (M, K), dtype=torch.int8)
            b = torch.randint(-8, 8, (K, N), dtype=torch.int8)
            c = torch.zeros((M, N), dtype=torch.int32)
            grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]), triton.cdiv(N, META["BLOCK_SIZE_N"]))
            compiled = matmul_kernel[grid](
                a, b, c, M, N, K,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=K,
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

    elif op == "matmul_small_m":
        # Same kernel body as matmul, but launched with a small BLOCK_SIZE_M so
        # the compute herd does less wasted M-row work for decode (M=1 padded
        # to BLOCK_SIZE_M). N/K block sizes stay 256 so the transform's L3->L2
        # copy loops still have something to tile.
        block_m = p.get("M", 16)
        launch = textwrap.dedent(f"""
            M, N, K = {block_m}, {p.get("N", 256)}, {p.get("K", 256)}
            a = torch.randint(-8, 8, (M, K), dtype=torch.int8)
            b = torch.randint(-8, 8, (K, N), dtype=torch.int8)
            c = torch.zeros((M, N), dtype=torch.int32)
            grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]), triton.cdiv(N, META["BLOCK_SIZE_N"]))
            compiled = matmul_kernel[grid](
                a, b, c, M, N, K,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                BLOCK_SIZE_M={block_m}, BLOCK_SIZE_N=256, BLOCK_SIZE_K=K,
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

    elif op == "matmul_bf16":
        launch = textwrap.dedent(f"""
            M, N, K = {p.get("M", 256)}, {p.get("N", 256)}, {p.get("K", 256)}
            a = torch.randn(M, K, dtype=torch.bfloat16)
            b = torch.randn(K, N, dtype=torch.bfloat16)
            c = torch.zeros((M, N), dtype=torch.float32)
            grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]), triton.cdiv(N, META["BLOCK_SIZE_N"]))
            compiled = matmul_bf16_kernel[grid](
                a, b, c, M, N, K,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=K,
            )
        """)
        kernel_src = textwrap.dedent("""
            @triton.jit
            def matmul_bf16_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
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
        # 2D BLOCK_M-rows form (upstream rms_norm example): tl.sum over axis=1
        # keeps a row dimension the transform script can tile at [1]. The 1D
        # form reduces to a scalar chain that is not tileable. M=1 does not
        # compile; host duplicates row 0 for M=2,N=2048 launches.
        m_rows = p.get("M", 32)
        n_cols = p.get("N", 256)
        block_m = 1 if m_rows < 2 else 2
        launch = textwrap.dedent(f"""
            M, N = {m_rows}, {n_cols}
            BLOCK_M = {block_m}
            x = torch.randn(M, N, dtype=torch.bfloat16)
            y = torch.empty(M, N, dtype=torch.bfloat16)
            grid = (max(1, M // BLOCK_M),)
            compiled = rmsnorm_kernel[grid](x, y, N, 1e-5, BLOCK_M=BLOCK_M, BLOCK_N=N)
        """)
        kernel_src = textwrap.dedent("""
            @triton.jit
            def rmsnorm_kernel(X, Y, N: tl.constexpr, eps: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
                pid = tl.program_id(0)
                rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
                cols = tl.arange(0, BLOCK_N)
                offsets = rows[:, None] * N + cols[None, :]
                x = tl.load(X + offsets)
                x_f32 = x.to(tl.float32)
                sum_sq = tl.sum(x_f32 * x_f32, axis=1)
                mean_sq = sum_sq / N
                rstd = tl.math.rsqrt(mean_sq + eps)
                y = (x_f32 * rstd[:, None]).to(x.dtype)
                tl.store(Y + offsets, y)
        """)

    elif op == "softmax":
        # 4-rows-per-program bf16 form (upstream test_softmax example); the
        # row-per-program f32 form leaves ops the transform script cannot map.
        launch = textwrap.dedent(f"""
            rows, cols = {p.get("rows", 256)}, {p.get("cols", 256)}
            x = torch.randn(rows, cols, dtype=torch.bfloat16)
            y = torch.empty(rows, cols, dtype=torch.bfloat16)
            grid = (rows // 4, 1)
            compiled = softmax_kernel[grid](x, y, cols, 1, cols, 1, cols, BLOCK_SIZE=4)
        """)
        kernel_src = textwrap.dedent("""
            @triton.jit
            def softmax_kernel(input_ptr, output_ptr,
                input_stride_row: tl.constexpr, input_stride_col: tl.constexpr,
                output_stride_row: tl.constexpr, output_stride_col: tl.constexpr,
                n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
                pid_row = tl.program_id(0)
                offs_row = pid_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                offs_col = tl.arange(0, n_cols)
                a_block = tl.load(input_ptr + offs_row[:, None] * input_stride_row + offs_col[None, :] * input_stride_col)
                row_minus_max = a_block - tl.max(a_block, axis=1, keep_dims=True)
                numerator = tl.exp(row_minus_max)
                denominator = tl.sum(numerator, axis=1, keep_dims=True)
                softmax_output = numerator / denominator
                tl.store(output_ptr + offs_row[:, None] * output_stride_row + offs_col[None, :] * output_stride_col, softmax_output)
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
        n_pairs = p.get("n_pairs", 32)
        # Pad the real n_pairs up to ROPE_PAD so flatten_tile_forall (tile 256)
        # yields >=2 trips -> a multi-core herd of standard 256-element tiles,
        # matching the proven silu structure. Tiny (16-elem) tiles were rejected
        # by the npu6 firmware at execution. Host zero-pads inputs to this size.
        rope_pad = max(512, ((n_pairs + 255) // 256) * 256 * 2)
        n_pairs = rope_pad
        block_size = rope_pad
        launch = textwrap.dedent(f"""
            n_pairs = {n_pairs}
            # in1_ptr: [t1[0..n_pairs-1]]   (n_pairs BF16)  — own buffer object
            # in2_ptr: [t2[0..n_pairs-1]]   (n_pairs BF16)  — own buffer object
            # out_ptr: [out[0..n_pairs-1]]  (n_pairs BF16)
            # out[i] = t1[i] + t2[i]
            #
            # HOST precomputes the products; NPU does the final addition:
            #   even call: t1 = x_e*cos,  t2 = -x_o*sin  → out_e = t1+t2 = x_e*cos - x_o*sin
            #   odd  call: t1 = x_e*sin,  t2 =  x_o*cos  → out_o = t1+t2 = x_e*sin + x_o*cos
            #
            # Three DISTINCT pointer args → three distinct buffer objects → three shim
            # DMA flows, exactly like the proven matmul (A, B, C) pattern. The earlier
            # single-buffer-with-offset form bound ONE bo to TWO shim input channels,
            # which the npu6 firmware rejected ("unexpected command state").
            #
            # Pure binary vector-add: ONE linalg.generic (2 inputs + 1 output = 3 DPS
            # operands) → @pad_and_promote_binary_bf16 matches exactly → single AIR herd.
            in1_ptr = torch.randn(n_pairs, dtype=torch.bfloat16)
            in2_ptr = torch.randn(n_pairs, dtype=torch.bfloat16)
            out_ptr = torch.empty(n_pairs, dtype=torch.bfloat16)
            grid = lambda META: (triton.cdiv(n_pairs, META["BLOCK_SIZE"]),)
            compiled = rope_kernel[grid](in1_ptr, in2_ptr, out_ptr, n_pairs, BLOCK_SIZE={block_size})
        """)
        kernel_src = textwrap.dedent("""
            @triton.jit
            def rope_kernel(in1_ptr, in2_ptr, out_ptr, n_pairs: tl.constexpr, BLOCK_SIZE: tl.constexpr):
                # in1_ptr: [t1[0..n_pairs-1]]   (own buffer)
                # in2_ptr: [t2[0..n_pairs-1]]   (own buffer)
                # out_ptr: [out[0..n_pairs-1]]
                # out[i] = t1[i] + t2[i]  (host precomputed t1 = x1*c1, t2 = x2*c2)
                #
                # No mask: BLOCK_SIZE == n_pairs (both constexpr), so offs < n_pairs
                # is always-true. Using a mask generates dynamic DMA sizes which
                # cause air-shrink-memref-sizes-by-access to crash.
                pid = tl.program_id(0)
                offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                t1 = tl.load(in1_ptr + offs)
                t2 = tl.load(in2_ptr + offs)
                out = t1 + t2
                tl.store(out_ptr + offs, out)
        """)

    elif op == "attn_qk":
        # QK^T scores for one head: scores[j] = scale * sum_d Q[d]*K[j,d].
        # Mirrors the rmsnorm 2D-rows form (reduce over the inner dim) so the
        # rmsnorm transform maps it to a herd: per-row reduce over head_dim, then
        # an output elementwise (scale) like rmsnorm's y = x*rstd.
        ctx_len = p.get("ctx_len", 256)
        head_dim = p.get("head_dim", 64)
        # Scale (1/sqrt(d)) is applied in-kernel as the post-reduce elementwise so
        # the transform has an output generic to anchor on (mirrors rmsnorm's
        # y = x*rstd). NOTE (WIP): this reaches AIE herd placement but the
        # rmsnorm-derived transform can't finish — QK's output is 1-D (a matvec),
        # so the 2-D-output assumptions (output-generic navigation, [0,16]
        # vectorization tiling) don't hold. A dedicated GEMV transform (or a
        # bf16 tl.dot matmul reusing matmul_aie2p.mlir) is the real path.
        launch = textwrap.dedent(f"""
            CTX, D = {ctx_len}, {head_dim}
            scale = 1.0 / (D ** 0.5)
            Q = torch.randn(D, dtype=torch.bfloat16)
            K = torch.randn(CTX, D, dtype=torch.bfloat16)
            S = torch.empty(CTX, dtype=torch.bfloat16)
            grid = (1,)
            compiled = attn_qk_kernel[grid](Q, K, S, scale, D, BLOCK_M=CTX, BLOCK_D=D)
        """)
        kernel_src = textwrap.dedent("""
            @triton.jit
            def attn_qk_kernel(Q, K, S, scale, head_dim: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
                rows = tl.arange(0, BLOCK_M)            # K rows (ctx positions)
                cols = tl.arange(0, BLOCK_D)            # head_dim
                k = tl.load(K + rows[:, None] * head_dim + cols[None, :]).to(tl.float32)
                q = tl.load(Q + cols).to(tl.float32)
                prod = k * q[None, :]                   # [BLOCK_M, head_dim]
                s = tl.sum(prod, axis=1)                # reduce over head_dim
                s = s * scale                           # output elementwise (attention scale)
                tl.store(S + rows, s.to(S.dtype.element_ty))
        """)

    elif op == "flash_attn":
        n_head = p.get("n_head", 8)
        head_dim = p.get("head_dim", 128)
        ctx_len = p.get("ctx_len", 2048)
        launch = textwrap.dedent(f"""
            n_head, head_dim, ctx_len = {n_head}, {head_dim}, {ctx_len}
            Q = torch.randn(n_head, head_dim, dtype=torch.float32)
            K = torch.randn(n_head, ctx_len, head_dim, dtype=torch.float32)
            V = torch.randn(n_head, ctx_len, head_dim, dtype=torch.float32)
            O = torch.empty(n_head, head_dim, dtype=torch.float32)
            grid = lambda META: (n_head,)
            compiled = flash_attn_kernel[grid](
                Q, K, V, O, head_dim, ctx_len,
                head_dim, ctx_len * head_dim, ctx_len * head_dim, head_dim,
            )
        """)
        kernel_src = textwrap.dedent("""
            @triton.jit
            def flash_attn_kernel(Q, K, V, output, head_dim: tl.constexpr, ctx_len: tl.constexpr,
                stride_qh: tl.constexpr, stride_kh: tl.constexpr, stride_vh: tl.constexpr, stride_oh: tl.constexpr):
                pid = tl.program_id(0)
                q_offs = tl.arange(0, head_dim)
                # Load Q for this head: shape [head_dim]
                q = tl.load(Q + pid * stride_qh + q_offs)
                k_base = pid * stride_kh
                v_base = pid * stride_vh
                # Index tensors for context and dimension dimensions
                k_idx = tl.arange(0, ctx_len)
                d_offs = tl.arange(0, head_dim)
                # QK^T: compute [ctx_len] attention scores via broadcasted dot products
                q_bcast = q[None, :]              # shape [1, head_dim]
                k_row = tl.load(K + k_base + k_idx[:, None] * head_dim + d_offs[None, :])  # shape [ctx_len, head_dim]
                scores = tl.sum(q_bcast * k_row, axis=1)                      # shape [ctx_len]
                # Softmax decomposition: max - sub - exp - sum - div
                scores_minus_max = scores - tl.max(scores, axis=0)            # shape [ctx_len]
                exp_scores = tl.exp(scores_minus_max)                         # shape [ctx_len]
                sum_exp = tl.sum(exp_scores, axis=0)                          # scalar
                weights = exp_scores / sum_exp                                # shape [ctx_len]
                # AV: weighted sum of V rows -> [head_dim] output
                v_row = tl.load(V + v_base + k_idx[:, None] * head_dim + d_offs[None, :])  # shape [ctx_len, head_dim]
                out = tl.sum(weights[:, None] * v_row, axis=0)                # shape [head_dim]
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
    if os.environ.get("GGNPU_DEBUG"):
        cmd.insert(1, "-v")
    result = subprocess.run(cmd, env=env, cwd=str(mlir.parent), capture_output=True, text=True, timeout=600)
    if os.environ.get("GGNPU_DEBUG"):
        print(f"  aircc fallback exit={result.returncode}")
        out = (result.stdout or "") + (result.stderr or "")
        for line in out.splitlines():
            print(f"  | {line}")


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

    exp_reason = KERNELS[op].get("experimental")
    if exp_reason and not os.environ.get("GGNPU_EXPERIMENTAL"):
        print(f"  SKIPPED: {op} is experimental — {exp_reason}.")
        print("  Set GGNPU_EXPERIMENTAL=1 to try building it anyway.")
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
    if _is_wsl():
        print("  note: WSL detected — using native Linux XRT shim under ~/.cache/ggnpu/")
    print(f"  XILINX_XRT={env['XILINX_XRT']}")

    link_err = run_launcher_link_smoke_test(env, repo_root)
    if link_err:
        print("  ERROR: Triton launcher link preflight failed (g++):")
        for line in link_err.splitlines():
            print(f"    {line}")
        if _is_wsl() and _is_drvfs(repo_root):
            print("  Hint: clone ggnpu under the Linux home dir (~/ggnpu), not /mnt/c/...")
        print("  Run: bash scripts/verify-kernel-build.sh")
        return False

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
                if os.environ.get("GGNPU_DEBUG"):
                    print(result.stderr)
                else:
                    print(extract_compiler_error(result.stderr))
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
        print("  The Triton/MLIR-AIR transform pipeline failed (see errors above).")
        print("  This usually means the kernel IR does not match the transform script")
        print(f"  ({transform.name}); kernel sizes must match the script's tiling.")
        print("  Re-run with GGNPU_DEBUG=1 for the full compiler output.")
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
    parser.add_argument("--n_pairs", type=int)
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
