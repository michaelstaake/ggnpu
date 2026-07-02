#!/usr/bin/env bash
# E2E regression for lfm2 (LiquidAI LFM2.5): France prompt → greedy top-1 is Paris.
# Exercises the full lfm2 hybrid forward on the NPU — interleaved gated ShortConv
# blocks and GQA attention (per-head QK-norm, NeoX RoPE), SwiGLU FFN, tied output
# with the token_embd_norm final norm, and the Llama3-family BPE pre-tokenizer.
set -euo pipefail

GGNPU="${1:?usage: test-lfm2-e2e.sh <ggnpu_binary> <model.gguf>}"
MODEL="${2:?usage: test-lfm2-e2e.sh <ggnpu_binary> <model.gguf>}"
PROMPT="The capital of France is"
PARIS_ID=5242   # " Paris" in the lfm2 (gpt2/Llama3-family BPE) vocab

if [[ ! -x "$GGNPU" ]]; then
    echo "FAIL: ggnpu binary not executable: $GGNPU"
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "SKIP: model not found: $MODEL (place an LFM2.5 Q6_K/Q8_0 GGUF in models/)"
    exit 77
fi

out=$("$GGNPU" bench-logits -m "$MODEL" -p "$PROMPT" -c 512 2>&1) || {
    echo "$out"
    echo "FAIL: bench-logits exited non-zero"
    exit 1
}
echo "$out"

if echo "$out" | grep -qE "top0: id=${PARIS_ID} "; then
    echo "PASS: lfm2 top-1 token is Paris (id=${PARIS_ID})"
    exit 0
fi

echo "FAIL: expected top0 id=${PARIS_ID} (Paris)"
exit 1
