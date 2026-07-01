#!/usr/bin/env bash
# E2E regression for gemma4 (Gemma 4 E2B): France prompt → greedy top-1 is Paris.
# Exercises the full gemma4 forward (unigram tokenizer, PLE, QK-norm, sandwich
# norms, shared-KV, sliding-window attention, soft-cap) on the NPU.
set -euo pipefail

GGNPU="${1:?usage: test-gemma-e2e.sh <ggnpu_binary> <model.gguf>}"
MODEL="${2:?usage: test-gemma-e2e.sh <ggnpu_binary> <model.gguf>}"
PROMPT="The capital of France is"
PARIS_ID=9079   # " Paris" in the gemma4 (SentencePiece-unigram) vocab

if [[ ! -x "$GGNPU" ]]; then
    echo "FAIL: ggnpu binary not executable: $GGNPU"
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "SKIP: model not found: $MODEL (place the gemma4 GGUF in models/)"
    exit 77
fi

out=$("$GGNPU" bench-logits -m "$MODEL" -p "$PROMPT" -c 2048 2>&1) || {
    echo "$out"
    echo "FAIL: bench-logits exited non-zero"
    exit 1
}
echo "$out"

if echo "$out" | grep -qE "top0: id=${PARIS_ID} "; then
    echo "PASS: gemma4 top-1 token is Paris (id=${PARIS_ID})"
    exit 0
fi

echo "FAIL: expected top0 id=${PARIS_ID} (Paris)"
exit 1
