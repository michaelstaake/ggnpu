#!/usr/bin/env python3
"""Run an INT8 decode matmul on the NPU with a chosen transform script, check
numerics against a torch reference, and time per-launch latency.

Purpose: microbenchmark the split-K decode matmul transform in isolation
(before any host wiring), and diff it against the current wide kernel.

The transform is selected by AIR_TRANSFORM_TILING_SCRIPT so we can point at an
experimental split-K .mlir without touching compile_kernels.py.

Usage:
    source .venv-triton/bin/activate
    # baseline (current wide kernel transform):
    python3 scripts/validate-splitk-matmul.py --transform matmul_small_m_aie2p.mlir \
        --M 8 --N 512 --K 2048 --iters 200
    # split-K:
    python3 scripts/validate-splitk-matmul.py --transform matmul_small_m_splitk_aie2p.mlir \
        --M 8 --N 512 --K 2048 --iters 200
"""
import argparse
import ctypes
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _setup_runtime_env():
    boost = REPO / "third_party/boost-lib/usr/lib/x86_64-linux-gnu"
    tools = REPO / "third_party/xrt-tools/usr/bin"
    if boost.is_dir():
        os.environ["LD_LIBRARY_PATH"] = f"{boost}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    if tools.is_dir():
        os.environ["PATH"] = f"{tools}:{os.environ.get('PATH', '')}"

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

    for cand in ("/usr/lib/x86_64-linux-gnu/libuuid.so.1", "libuuid.so.1"):
        try:
            ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL)
            break
        except OSError:
            continue
    return ck


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transform", default="matmul_small_m_aie2p.mlir",
                    help="transform .mlir basename under kernels/triton/transforms/ (or abs path)")
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--K", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    ck = _setup_runtime_env()
    env = ck.setup_compile_env(REPO)
    os.environ.update(env)
    os.environ["AMD_TRITON_NPU_TARGET"] = "npu2"

    tpath = Path(args.transform)
    if not tpath.is_absolute():
        tpath = REPO / "kernels" / "triton" / "transforms" / args.transform
    if not tpath.exists():
        print(f"ERROR: transform not found: {tpath}")
        return 2
    os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = str(tpath)

    import torch
    import triton
    import triton.language as tl
    from triton.backends.amd_triton_npu.driver import NPUDriver
    from triton.backends.amd_triton_npu.config import npu_config

    triton.runtime.driver.set_active(NPUDriver())
    npu_config.compile_only = False
    npu_config.target = "npu2"
    npu_config.output_format = "xclbin"
    npu_config.air_project_path = "/tmp/val_splitk_air"

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

    M, N, K = args.M, args.N, args.K
    torch.manual_seed(0)
    a = torch.randint(-8, 8, (M, K), dtype=torch.int8)
    b = torch.randint(-8, 8, (K, N), dtype=torch.int8)
    c = torch.zeros((M, N), dtype=torch.int32)

    # Match compile_kernels.py convention: BLOCK_SIZE_N is pinned to 256 (the herd
    # maps N_groups=256/8/... to the 4 device rows; N>256 becomes extra Triton
    # programs, NOT a wider herd). BLOCK_SIZE_M = M, BLOCK_SIZE_K = K (deep-K).
    block_n = min(N, 256)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]), triton.cdiv(N, META["BLOCK_SIZE_N"]))

    def launch():
        matmul_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
            BLOCK_SIZE_M=M, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=K,
        )

    # First launch triggers compile.
    t_compile = time.time()
    launch()
    compile_s = time.time() - t_compile

    ref = a.to(torch.int32) @ b.to(torch.int32)
    diff = (c - ref).abs()
    max_abs = int(diff.max().item())
    mism = int((diff != 0).sum().item())
    ok = (max_abs == 0)
    print(f"transform: {tpath.name}")
    print(f"INT8 matmul {M}x{N}x{K}  (compile+first run {compile_s:.1f}s)")
    print(f"  max abs diff : {max_abs}")
    print(f"  mismatches   : {mism} / {M * N}")
    print(f"  numerics     : {'PASS' if ok else 'FAIL'}")

    # Timing: many launches, report per-launch ms. (Kernel is recompiled-cached
    # after the first call, so these are pure execute+DMA round-trips.)
    iters = args.iters
    # warmup
    for _ in range(5):
        launch()
    t0 = time.time()
    for _ in range(iters):
        launch()
    dt = time.time() - t0
    print(f"  per-launch   : {dt / iters * 1e3:.4f} ms  ({iters} iters, {dt:.2f}s total)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
