#!/usr/bin/env python3
"""Run the bf16 matmul kernel on the NPU and compare to a torch reference.

Validates the matmul_bf16 datapath (foundation for decomposed NPU attention).
Unlike compile_kernels.py (compile-only), this executes the kernel on hardware,
so it needs the NPU present plus three env fixes the build path doesn't set:
  * LD_PRELOAD=libuuid.so.1     — Triton-XDNA launcher needs uuid_unparse_lower
  * bundled boost on LD_LIBRARY_PATH — xclbinutil (libboost_filesystem)
  * system XRT runtime under $XILINX_XRT/lib — libxrt_core.so.2 etc.
This script sets all three up itself, then runs.

Usage:
    source .venv-triton/bin/activate
    python3 scripts/validate-bf16-matmul.py [M N K]   # default 256 256 256
"""
import ctypes
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _setup_runtime_env() -> None:
    # boost (xclbinutil) + xrt-tools on PATH/LD_LIBRARY_PATH
    boost = REPO / "third_party/boost-lib/usr/lib/x86_64-linux-gnu"
    tools = REPO / "third_party/xrt-tools/usr/bin"
    if boost.is_dir():
        os.environ["LD_LIBRARY_PATH"] = f"{boost}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    if tools.is_dir():
        os.environ["PATH"] = f"{tools}:{os.environ.get('PATH', '')}"

    # Make the system XRT runtime reachable under $XILINX_XRT/lib (the dev tree
    # ships only headers + unversioned .so). Symlink the versioned runtime libs.
    sys.path.insert(0, str(REPO / "kernels" / "triton"))
    import compile_kernels as ck  # noqa: E402

    xrt_root = ck.find_xilinx_xrt(REPO)
    if xrt_root:
        os.environ["XILINX_XRT"] = str(xrt_root)
        xrt_lib = Path(xrt_root) / "lib" / "x86_64-linux-gnu"
        xrt_lib.mkdir(parents=True, exist_ok=True)
        syslib = Path("/usr/lib/x86_64-linux-gnu")
        for name in ("libxrt_core.so.2", "libxrt_coreutil.so.2", "libxrt_driver_xdna.so.2"):
            dst, src = xrt_lib / name, syslib / name
            if src.exists() and not dst.exists():
                dst.symlink_to(src)

    # libuuid for the JIT launcher .so (undefined symbol: uuid_unparse_lower)
    for cand in ("/usr/lib/x86_64-linux-gnu/libuuid.so.1", "libuuid.so.1"):
        try:
            ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL)
            break
        except OSError:
            continue
    return ck


def main() -> int:
    ck = _setup_runtime_env()
    env = ck.setup_compile_env(REPO)
    os.environ.update(env)
    os.environ["AMD_TRITON_NPU_TARGET"] = "npu2"
    os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = str(ck.transform_path("matmul_bf16"))

    import torch
    import triton
    import triton.language as tl
    from triton.backends.amd_triton_npu.driver import NPUDriver
    from triton.backends.amd_triton_npu.config import npu_config

    triton.runtime.driver.set_active(NPUDriver())
    npu_config.compile_only = False
    npu_config.target = "npu2"
    npu_config.output_format = "xclbin"
    npu_config.air_project_path = "/tmp/val_bf16_air"

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

    M, N, K = (int(x) for x in (sys.argv[1:4] if len(sys.argv) >= 4 else (256, 256, 256)))
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.bfloat16) * 0.5
    b = torch.randn(K, N, dtype=torch.bfloat16) * 0.5
    c = torch.zeros((M, N), dtype=torch.float32)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]), triton.cdiv(N, META["BLOCK_SIZE_N"]))
    matmul_bf16_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=K,
    )

    ref = a.float() @ b.float()
    diff = (c - ref).abs()
    max_abs, max_ref = diff.max().item(), ref.abs().max().item()
    rel = max_abs / max_ref if max_ref > 0 else max_abs
    mism = int((diff > 0.05 * max_ref).sum().item())
    print(f"NPU bf16 matmul {M}x{N}x{K}:")
    print(f"  max abs diff : {max_abs:.5f}")
    print(f"  max ref value: {max_ref:.5f}")
    print(f"  relative err : {rel:.5f}")
    print(f"  mismatches(>5% of max): {mism} / {M * N}")
    ok = rel < 0.03
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
