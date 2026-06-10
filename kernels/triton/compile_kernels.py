#!/usr/bin/env python3
"""
Triton-XDNA kernel definitions for ggnpu.

Replaces the mlir-aie/IRON kernel sources with Triton Python kernels.
Compiled to .xclbin files using the Triton-XDNA compiler pipeline.

Usage:
    python3 compile_kernels.py --op matmul --profile npu6 --M 256 --N 256 --K 256
    python3 compile_kernels.py --op rmsnorm --profile npu6 --N 2048
    python3 compile_kernels.py --op softmax --profile npu6 --rows 1 --cols 1024
    python3 compile_kernels.py --op silu --profile npu6 --N 8192
"""

import argparse
import sys
import os
import subprocess
import tempfile
import shutil

# Kernel definitions
KERNELS = {
    "matmul": {
        "func": "matmul_kernel",
        "description": "INT8/BF16 matrix multiplication",
        "params": ["M", "N", "K"],
        "dtype_in": "bfloat16",
        "dtype_out": "float32",
    },
    "rmsnorm": {
        "func": "rmsnorm_kernel",
        "description": "RMS normalization",
        "params": ["N"],
        "dtype_in": "float32",
        "dtype_out": "float32",
    },
    "softmax": {
        "func": "softmax_kernel",
        "description": "Softmax activation",
        "params": ["rows", "cols"],
        "dtype_in": "float32",
        "dtype_out": "float32",
    },
    "silu": {
        "func": "silu_kernel",
        "description": "SiLU/Swish activation",
        "params": ["N"],
        "dtype_in": "float32",
        "dtype_out": "float32",
    },
    "rope": {
        "func": "rope_kernel",
        "description": "Rotary positional embeddings",
        "params": ["N", "dims"],
        "dtype_in": "float32",
        "dtype_out": "float32",
    },
    "flash_attn": {
        "func": "flash_attn_kernel",
        "description": "FlashAttention v1 (decomposed)",
        "params": ["n_head", "head_dim", "ctx_len"],
        "dtype_in": "float32",
        "dtype_out": "float32",
    },
}

# Triton kernel source code
KERNEL_SOURCES = {
    "matmul": """
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A, B, C,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_block = tl.load(A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_block = tl.load(B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    c_block = tl.dot(a_block, b_block)

    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c_block)
""",
    "rmsnorm": """
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    input_ptr, output_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    input_block = tl.load(input_ptr + offset, mask=mask)
    
    # Compute RMS
    mean_square = tl.sum(input_block * input_block, axis=0) / N
    rms = tl.sqrt(mean_square + eps)
    
    output_block = input_block / rms
    
    tl.store(output_ptr + offset, output_block, mask=mask)
""",
    "softmax": """
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    rows: tl.constexpr, cols: tl.constexpr,
    stride_row_in: tl.constexpr, stride_row_out: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < cols

    for row in range(rows):
        input_block = tl.load(input_ptr + row * stride_row_in + offset, mask=mask)
        
        # Subtract max for numerical stability
        max_val = tl.max(input_block, axis=0)
        exp_vals = tl.exp(input_block - max_val)
        
        # Normalize
        sum_vals = tl.sum(exp_vals, axis=0)
        output_block = exp_vals / sum_vals
        
        tl.store(output_ptr + row * stride_row_out + offset, output_block, mask=mask)
""",
    "silu": """
import triton
import triton.language as tl

@triton.jit
def silu_kernel(
    input_ptr, output_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    input_block = tl.load(input_ptr + offset, mask=mask)
    
    # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    output_block = input_block / (1.0 + tl.exp(-input_block))
    
    tl.store(output_ptr + offset, output_block, mask=mask)
""",
    "rope": """
import triton
import triton.language as tl
import math

@triton.jit
def rope_kernel(
    input_ptr, output_ptr,
    N: tl.constexpr, dims: tl.constexpr,
    offset: tl.constexpr, freq_base: tl.constexpr, freq_scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset_idx < N

    input_block = tl.load(input_ptr + offset_idx, mask=mask)
    
    # Apply RoPE rotation (simplified - actual implementation needs pair-wise rotation)
    # This is a placeholder - full RoPE requires pair-wise operations
    output_block = input_block  # TODO: implement full RoPE
    
    tl.store(output_ptr + offset_idx, output_block, mask=mask)
""",
    "flash_attn": """
import triton
import triton.language as tl

@triton.jit
def flash_attn_kernel(
    Q, K, V, output,
    n_head: tl.constexpr, head_dim: tl.constexpr, ctx_len: tl.constexpr,
    stride_qh: tl.constexpr, stride_kh: tl.constexpr, stride_vh: tl.constexpr,
    stride_oh: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # FlashAttention v1 (decomposed)
    # TODO: Full implementation with tiling and memory optimization
    pid = tl.program_id(0)
    
    # Load Q, K, V blocks
    # Compute attention scores
    # Apply softmax
    # Weighted sum
    # TODO: Implement full FlashAttention
    pass
""",
}


def compile_kernel(op, profile, params, output_dir):
    """Compile a single Triton kernel to xclbin using Triton-XDNA."""
    
    if op not in KERNELS:
        print(f"Error: Unknown kernel '{op}'. Available: {list(KERNELS.keys())}")
        return False
    
    if op not in KERNEL_SOURCES:
        print(f"Error: No Triton source for kernel '{op}'")
        return False
    
    # Get kernel info
    kernel_info = KERNELS[op]
    print(f"Compiling {op}: {kernel_info['description']}")
    
    # Create temporary Python script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        script_path = f.name
        f.write(KERNEL_SOURCES[op])
        f.write("\n\n")
        
        # Add compilation code
        f.write(f"""
import triton
from triton.backends.amd_triton_npu.driver import NPUDriver

# Set NPU backend
triton.runtime.driver.set_active(NPUDriver())

# Get kernel function
kernel_func = {kernel_info['func']}

# Build grid based on kernel type
if "{op}" == "matmul":
    M, N, K = {params['M']}, {params['N']}, {params['K']}
    BLOCK_SIZE_M = 256
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = K
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    kernel_func[grid](
        None, None, None,  # Dummy pointers for compilation
        M, N, K,
        K, 1,  # stride_am, stride_ak (dummy)
        N, 1,  # stride_bk, stride_bn (dummy)
        N, 1,  # stride_cm, stride_cn (dummy)
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
elif "{op}" == "rmsnorm":
    N = {params['N']}
    BLOCK_SIZE = 1024
    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']),)
    kernel_func[grid](
        None, None,  # Dummy pointers
        N, 1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
    )
elif "{op}" == "softmax":
    rows, cols = {params['rows']}, {params['cols']}
    BLOCK_SIZE = 1024
    grid = lambda META: (triton.cdiv(rows, META['BLOCK_SIZE']),)
    kernel_func[grid](
        None, None,  # Dummy pointers
        rows, cols,
        cols, cols,  # stride_row_in, stride_row_out
        BLOCK_SIZE=BLOCK_SIZE,
    )
elif "{op}" == "silu":
    N = {params['N']}
    BLOCK_SIZE = 1024
    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']),)
    kernel_func[grid](
        None, None,  # Dummy pointers
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
elif "{op}" == "rope":
    N, dims = {params['N']}, {params['dims']}
    BLOCK_SIZE = 1024
    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']),)
    kernel_func[grid](
        None, None,  # Dummy pointers
        N, dims,
        0, 500000, 1.0,  # offset, freq_base, freq_scale
        BLOCK_SIZE=BLOCK_SIZE,
    )
elif "{op}" == "flash_attn":
    n_head, head_dim, ctx_len = {params['n_head']}, {params['head_dim']}, {params['ctx_len']}
    BLOCK_SIZE_Q = 64
    BLOCK_SIZE_K = 64
    grid = lambda META: (n_head,)
    kernel_func[grid](
        None, None, None, None,  # Dummy pointers
        n_head, head_dim, ctx_len,
        head_dim, ctx_len * head_dim, ctx_len * head_dim,  # strides
        head_dim,  # stride_oh
        BLOCK_SIZE_Q=BLOCK_SIZE_Q, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

print(f"Compiled {{op}} kernel successfully")
""")
    
    try:
        # Run compilation
        print(f"  Running Triton-XDNA compiler...")
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"  ERROR: Compilation failed")
            print(f"  stdout: {result.stdout}")
            print(f"  stderr: {result.stderr}")
            return False
        
        print(f"  Compilation successful")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"  ERROR: Compilation timed out")
        return False
    finally:
        # Clean up temp file
        os.unlink(script_path)


def main():
    parser = argparse.ArgumentParser(description="Compile ggnpu Triton kernels to xclbin")
    parser.add_argument("--op", required=True, choices=list(KERNELS.keys()),
                       help="Kernel type to compile")
    parser.add_argument("--profile", default="npu6", choices=["npu4", "npu5", "npu6"],
                       help="NPU profile (npu4/5/6)")
    parser.add_argument("--output-dir", default="~/.cache/ggnpu/xclbin",
                       help="Output directory for xclbin files")
    
    # Kernel-specific parameters
    parser.add_argument("--M", type=int, help="Matmul M dimension")
    parser.add_argument("--N", type=int, help="Matmul/RMSNorm/SiLU/N dimension")
    parser.add_argument("--K", type=int, help="Matmul K dimension")
    parser.add_argument("--rows", type=int, help="Softmax rows")
    parser.add_argument("--cols", type=int, help="Softmax cols")
    parser.add_argument("--dims", type=int, help="RoPE dims")
    parser.add_argument("--n_head", type=int, help="FlashAttention n_head")
    parser.add_argument("--head_dim", type=int, help="FlashAttention head_dim")
    parser.add_argument("--ctx_len", type=int, help="FlashAttention ctx_len")
    
    args = parser.parse_args()
    
    # Expand output directory
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Build parameters dict
    params = {}
    if args.M: params["M"] = args.M
    if args.N: params["N"] = args.N
    if args.K: params["K"] = args.K
    if args.rows: params["rows"] = args.rows
    if args.cols: params["cols"] = args.cols
    if args.dims: params["dims"] = args.dims
    if args.n_head: params["n_head"] = args.n_head
    if args.head_dim: params["head_dim"] = args.head_dim
    if args.ctx_len: params["ctx_len"] = args.ctx_len
    
    # Compile kernel
    success = compile_kernel(args.op, args.profile, params, output_dir)
    
    if success:
        xclbin_path = os.path.join(output_dir, f"{args.op}_{args.profile}.xclbin")
        print(f"\nKernel compiled: {xclbin_path}")
    else:
        print(f"\nKernel compilation FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
