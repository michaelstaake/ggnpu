#!/usr/bin/env python3
"""Compare ggnpu bench-logits vs llama-cpp-python reference.

Usage:
  python3 scripts/compare_logits.py          # print side-by-side (needs llama-cpp-python)
  python3 scripts/compare_logits.py --check  # exit 1 if ggnpu top-1 != Paris (12366)
  python3 scripts/compare_logits.py --check --ref  # also compare Paris logit vs llama.cpp
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "llama-3.2-1b-q4_k_m.gguf"
PROMPT = "The capital of France is"
PARIS_ID = 12366
LOGIT_TOL = 0.15


def find_ggnpu() -> Path:
    for candidate in (ROOT / "build" / "ggnpu", ROOT / "build-npu" / "ggnpu"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("ggnpu binary not found under build/ or build-npu/")


def ggnpu_bench(exe: Path) -> str:
    env = {**os.environ, "GGNPU_MAX_LAYERS": "16"}
    return subprocess.check_output(
        [str(exe), "bench-logits", "-m", str(MODEL), "-p", PROMPT, "-c", "2048"],
        env=env,
        text=True,
        cwd=ROOT,
    )


def parse_top0_id(output: str) -> int | None:
    m = re.search(r"top0: id=(\d+)", output)
    return int(m.group(1)) if m else None


def parse_token_logit(output: str, token_id: int) -> float | None:
    m = re.search(rf"top\d+: id={token_id} logit=([-\d.eE+]+)", output)
    return float(m.group(1)) if m else None


def llama_cpp_logits():
    from llama_cpp import Llama

    llm = Llama(
        model_path=str(MODEL),
        n_ctx=2048,
        n_gpu_layers=0,
        verbose=False,
        logits_all=True,
    )
    llm.reset()
    ids = llm.tokenize(PROMPT.encode(), add_bos=True)
    llm.eval(ids)
    return llm._scores[-1], ids, llm


def print_reference(logits, ids, llm) -> None:
    print("=== llama-cpp-python ===")
    print("input ids:", ids)
    top = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:10]
    for i in top:
        text = llm.detokenize([i]).decode("utf-8", errors="replace")
        print(f"  top: id={i} logit={logits[i]:.4f} text={text!r}")
    for pid in (PARIS_ID, 60704, 264, 279, 374):
        text = llm.detokenize([pid]).decode("utf-8", errors="replace")
        print(f"  cand id={pid} logit={logits[pid]:.4f} text={text!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="assert ggnpu top-1 is Paris; exit 1 on failure",
    )
    parser.add_argument(
        "--ref",
        action="store_true",
        help="with --check, also compare Paris logit to llama-cpp-python",
    )
    args = parser.parse_args()

    if not MODEL.is_file():
        print(f"SKIP: model not found: {MODEL}", file=sys.stderr)
        return 77 if args.check else 0

    exe = find_ggnpu()
    out = ggnpu_bench(exe)
    top0 = parse_top0_id(out)

    if not args.check:
        try:
            logits, ids, llm = llama_cpp_logits()
            print_reference(logits, ids, llm)
        except ImportError:
            print("(llama-cpp-python not installed; showing ggnpu only)", file=sys.stderr)
        print("\n=== ggnpu bench-logits ===")
        print(out)
        return 0

    if top0 != PARIS_ID:
        print(out)
        print(f"FAIL: ggnpu top0 id={top0}, expected {PARIS_ID} (Paris)", file=sys.stderr)
        return 1

    if args.ref:
        try:
            logits, _, _ = llama_cpp_logits()
            ref = logits[PARIS_ID]
            got = parse_token_logit(out, PARIS_ID)
            if got is None:
                print(out)
                print(f"FAIL: Paris logit not found in ggnpu output", file=sys.stderr)
                return 1
            if abs(got - ref) > LOGIT_TOL:
                print(out)
                print(
                    f"FAIL: Paris logit ggnpu={got:.4f} llama.cpp={ref:.4f} "
                    f"(tol={LOGIT_TOL})",
                    file=sys.stderr,
                )
                return 1
            print(f"PASS: top0 Paris; logit ggnpu={got:.4f} ref={ref:.4f}")
        except ImportError:
            print("PASS: top0 Paris (llama-cpp-python not installed; skipped logit compare)")
    else:
        print(f"PASS: top0 Paris (id={PARIS_ID})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
