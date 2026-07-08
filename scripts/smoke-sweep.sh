#!/usr/bin/env bash
# Smoke-sweep every GGUF in models/ on the NPU.
#
# For each model it runs a short greedy generation on the prompt
# "The capital of France is" and classifies the result:
#
#   PASS      exit 0, produced text, and the decoded output mentions "Paris"
#             (vocab-agnostic coherence check — no per-tokenizer token id needed)
#   NO-PARIS  ran and produced text but never said Paris -> degraded / garbage,
#             eyeball it (expected for sub-2-bit i-quants)
#   EMPTY     exit 0 but produced no text
#   CRASH     non-zero exit
#   TIMEOUT   exceeded per-model timeout
#   SKIP      file larger than the size cap (see SWEEP_MAX_GB / FULL)
#
# Usage:
#   scripts/smoke-sweep.sh [ggnpu_binary] [models_dir]
#
# Env knobs:
#   FULL=1            include every file regardless of size
#   SWEEP_MAX_GB=5    skip files larger than this many GB (default 5)
#   SWEEP_TIMEOUT=600 per-model timeout in seconds (default 600)
#   SWEEP_NTOK=32     tokens to generate (default 32)
#   SWEEP_CTX=2048    context size (default 2048)
#   SWEEP_ONLY=glob   only run files whose name matches this shell glob
set -uo pipefail

GGNPU="${1:-build-npu/ggnpu}"
MODELS_DIR="${2:-models}"
PROMPT="The capital of France is"
MAX_GB="${SWEEP_MAX_GB:-5}"
TIMEOUT="${SWEEP_TIMEOUT:-600}"
NTOK="${SWEEP_NTOK:-32}"
CTX="${SWEEP_CTX:-2048}"
ONLY="${SWEEP_ONLY:-*}"

if [[ ! -x "$GGNPU" ]]; then
    echo "FATAL: ggnpu binary not executable: $GGNPU" >&2
    exit 1
fi
if [[ ! -d "$MODELS_DIR" ]]; then
    echo "FATAL: models dir not found: $MODELS_DIR" >&2
    exit 1
fi

# max file size in bytes (0 = no cap)
if [[ "${FULL:-0}" == "1" ]]; then MAX_BYTES=0; else
    MAX_BYTES=$(awk "BEGIN{printf \"%d\", $MAX_GB*1024*1024*1024}")
fi

STAMP=$(date +%Y%m%d-%H%M%S)
OUTDIR="sweep-results/$STAMP"
mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary.tsv"
printf "status\tsize_gb\tseconds\tmodel\tfirst_line\n" > "$SUMMARY"

echo "ggnpu:   $GGNPU"
echo "models:  $MODELS_DIR   (cap: $([[ $MAX_BYTES == 0 ]] && echo none || echo ${MAX_GB}GB), timeout ${TIMEOUT}s, -n $NTOK, -c $CTX)"
echo "output:  $OUTDIR/"
echo "prompt:  \"$PROMPT\""
echo

declare -A COUNT
pad() { printf '%-9s' "$1"; }

shopt -s nullglob
mapfile -t FILES < <(ls -1 "$MODELS_DIR"/$ONLY.gguf 2>/dev/null | sort)

for MODEL in "${FILES[@]}"; do
    name=$(basename "$MODEL")
    bytes=$(stat -c%s "$MODEL")
    gb=$(awk "BEGIN{printf \"%.1f\", $bytes/1073741824}")

    if [[ $MAX_BYTES != 0 && $bytes -gt $MAX_BYTES ]]; then
        COUNT[SKIP]=$(( ${COUNT[SKIP]:-0} + 1 ))
        echo "$(pad SKIP) ${gb}G  $name  (over ${MAX_GB}GB cap; FULL=1 to include)"
        printf "SKIP\t%s\t0\t%s\t\n" "$gb" "$name" >> "$SUMMARY"
        continue
    fi

    log="$OUTDIR/${name%.gguf}.log"
    start=$(date +%s)
    out=$(timeout "${TIMEOUT}s" "$GGNPU" -m "$MODEL" -p "$PROMPT" -n "$NTOK" -c "$CTX" --temp 0 --quiet 2>"$log")
    rc=$?
    secs=$(( $(date +%s) - start ))
    first=$(printf '%s' "$out" | tr '\n' ' ' | cut -c1-80)

    if [[ $rc == 124 ]]; then
        status=TIMEOUT
    elif [[ $rc != 0 ]]; then
        status=CRASH
    elif [[ -z "${out// }" ]]; then
        status=EMPTY
    elif printf '%s' "$out" | grep -qi 'paris'; then
        status=PASS
    else
        status="NO-PARIS"
    fi

    COUNT[$status]=$(( ${COUNT[$status]:-0} + 1 ))
    echo "$(pad "$status") ${gb}G  ${secs}s  $name  | ${first}"
    printf "%s\t%s\t%s\t%s\t%s\n" "$status" "$gb" "$secs" "$name" "$first" >> "$SUMMARY"
done

echo
echo "==================== summary ===================="
for s in PASS NO-PARIS EMPTY CRASH TIMEOUT SKIP; do
    printf "  %-9s %d\n" "$s" "${COUNT[$s]:-0}"
done
echo "full table: $SUMMARY"
echo "per-model stderr logs: $OUTDIR/*.log"

# non-zero exit if anything genuinely failed (crash/timeout/empty)
fail=$(( ${COUNT[CRASH]:-0} + ${COUNT[TIMEOUT]:-0} + ${COUNT[EMPTY]:-0} ))
[[ $fail == 0 ]]
