# GGNPU Usage

## Quick Start

```bash
# Basic inference
ggnpu -m models/llama-3.2-1b-Q4_K_M.gguf -p "The capital of France is" -n 64

# With sampling
ggnpu -m model.gguf -p "Once upon a time" -c 4096 -n 256 --temp 0.7 --seed 42

# Inspect model
ggnpu -m model.gguf --dump-tensors

# Benchmark
ggnpu bench-matmul

# Version info
ggnpu --version
```

## CLI Reference

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--model` | `-m` | required | Path to `.gguf` file |
| `--prompt` | `-p` | `""` | Input prompt text |
| `--max-tokens` | `-n` | `128` | Max new tokens to generate |
| `--ctx-size` | `-c` | model default | Context window size |
| `--threads` | `-t` | `4` | CPU threads for I/O |
| `--temp` | | `0.0` | Temperature (0 = greedy) |
| `--seed` | | `0` | RNG seed (0 = random) |
| `--npu-device` | | `0` | NPU device index |
| `--no-cache` | | off | Disable compile/cache |
| `--cache-dir` | | `~/.cache/ggnpu` | Cache directory path |
| `--quiet` | | off | Status to stderr; generated text only on stdout |
| `--verbose` | `-v` | off | Per-operation timings |
| `--help` | `-h` | | Print help |
| `--version` | | | Print version |

## Commands

- **default**: Text generation from prompt
- `--dump-tensors`: List model tensors and metadata (no NPU)
- `bench-matmul`: NPU matmul benchmark
- `--version`: Show version and backend info

## Examples

```bash
# Llama 3.2 1B
ggnpu -m models/llama-3.2-1b-Q4_K_M.gguf -p "Hello, world!" -n 32

# Qwen2.5 Coder
ggnpu -m models/qwen2.5-coder-1.5b-instruct-Q4_K_M.gguf -p "def hello():" -n 64

# Gemma
ggnpu -m models/gemma-4-E2B-it-qat-Q4_K_XL.gguf -p "Tell me a joke" -n 32 --temp 0.8
```
