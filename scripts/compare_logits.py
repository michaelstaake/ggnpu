#!/usr/bin/env python3
"""Compare ggnpu bench-logits vs llama-cpp-python reference."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "llama-3.2-1b-q4_k_m.gguf"
PROMPT = "The capital of France is"
PARIS_IDS = [12366, 60704, 264, 279, 374]


def llama_cpp_logits():
    from llama_cpp import Llama

    llm = Llama(model_path=str(MODEL), n_ctx=2048, n_gpu_layers=0, verbose=False, logits_all=True)
    llm.reset()
    ids = llm.tokenize(PROMPT.encode(), add_bos=True)
    llm.eval(ids)
    logits = llm._scores[-1]
    return logits, ids


def ggnpu_bench():
    exe = ROOT / "build" / "ggnpu"
    if not exe.exists():
        exe = ROOT / "build-npu" / "ggnpu"
    env = {"GGNPU_MAX_LAYERS": "16"}
    out = subprocess.check_output(
        [
            str(exe),
            "bench-logits",
            "-m",
            str(MODEL),
            "-p",
            PROMPT,
            "-c",
            "2048",
        ],
        env={**dict(__import__("os").environ), **env},
        text=True,
        cwd=ROOT,
    )
    return out


def main():
    print("=== llama-cpp-python ===")
    logits, ids = llama_cpp_logits()
    print("input ids:", ids)
    top = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:10]
    for i in top:
        tok = __import__("llama_cpp").Llama(model_path=str(MODEL), n_gpu_layers=0, verbose=False)
        break
    llm = __import__("llama_cpp").Llama(model_path=str(MODEL), n_gpu_layers=0, verbose=False)
    for i in top:
        print(f"  top: id={i} logit={logits[i]:.4f} text={llm.detokenize([i]).decode('utf-8', errors='replace')!r}")
    for pid in PARIS_IDS:
        print(f"  paris cand id={pid} logit={logits[pid]:.4f} text={llm.detokenize([pid]).decode('utf-8', errors='replace')!r}")

    print("\n=== ggnpu bench-logits ===")
    print(ggnpu_bench())


if __name__ == "__main__":
    main()
