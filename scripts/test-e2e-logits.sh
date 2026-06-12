#!/usr/bin/env bash
# E2E regression: France prompt → greedy top-1 token is Paris (id 12366).
set -euo pipefail

GGNPU="${1:?usage: test-e2e-logits.sh <ggnpu_binary> <model.gguf>}"
MODEL="${2:?usage: test-e2e-logits.sh <ggnpu_binary> <model.gguf>}"
PROMPT="The capital of France is"
PARIS_ID=12366

if [[ ! -x "$GGNPU" ]]; then
    echo "FAIL: ggnpu binary not executable: $GGNPU"
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "SKIP: model not found: $MODEL (place GGUF in models/ for E2E tests)"
    exit 77
fi

out=$(GGNPU_MAX_LAYERS=16 "$GGNPU" bench-logits -m "$MODEL" -p "$PROMPT" -c 2048 2>&1) || {
    echo "$out"
    echo "FAIL: bench-logits exited non-zero"
    exit 1
}
echo "$out"

if echo "$out" | grep -qE "top0: id=${PARIS_ID} "; then
    echo "PASS: top-1 token is Paris (id=${PARIS_ID})"
    exit 0
fi

echo "FAIL: expected top0 id=${PARIS_ID} (Paris)"
exit 1
