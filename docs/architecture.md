# GGNPU Architecture

## Overview

GGNPU is a custom C++20 inference runtime that runs standard GGUF models on AMD NPUs.

## Layers

### Layer 1: GGUF I/O
- Parses GGUF v3 format
- Memory-maps tensor data
- Extracts metadata (architecture, hyperparameters)

### Layer 2: Architecture Plugin
- Maps tensor names to compute roles
- Handles GQA (Grouped Query Attention)
- Supports llama, qwen2, mistral, gemma2

### Layer 3: Quantization
- Decodes GGUF quant types to INT8 NPU buffers
- Supports F32, F16, Q4_0, Q8_0, Q4_K, Q6_K
- Cached weight conversion

### Layer 4: Compute Graph
- DAG of operations: matmul, norm, rope, softmax, silu, attention
- Prefill and decode execution modes
- KV cache management

### Layer 5: Backend Interface
- Abstract backend: mul_mat_q, rms_norm, rope, softmax, silu, flash_attn
- Production: AmdXdnaBackend (XRT)
- Tests: CpuRefBackend

### Layer 6: AMD XDNA Backend
- XRT device management
- Buffer allocation (pinned memory)
- Kernel compilation (mlir-aie, Peano)
- Work submission and sync

## Data Flow

```
GGUF file → Parser → TensorViews → Quant Decoder → NPU Buffers
                                                        ↓
Prompt → Tokenizer → Tokens → Graph Execute → Sampler → Text
```
