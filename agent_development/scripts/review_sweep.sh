#!/usr/bin/env bash
# Review a finished RL sweep with the rl-run-review skill (non-interactive, read/bash tools only).
# Usage: agent_development/scripts/review_sweep.sh runs/<sweep>
# Output: agent_development/reviews/<sweep-name>/review.md (model, thinking level and date at the top).
# An existing review.md is kept as review-<timestamp>.md.
set -euo pipefail

MODEL="MIT-parley/bedrock/claude-opus-5-5"
THINKING="high"

[ $# -eq 1 ] || { echo "usage: $0 <sweep_dir>" >&2; exit 2; }
SWEEP="${1%/}"
[ -d "$SWEEP" ] || { echo "not a folder: $SWEEP" >&2; exit 2; }

cd "$(dirname "$0")/../.."
NAME="$(basename "$SWEEP")"
OUT="agent_development/reviews/$NAME"
mkdir -p "$OUT"
STAMP="$(date +%Y%m%d-%H%M%S)"
if [ -f "$OUT/review.md" ]; then mv "$OUT/review.md" "$OUT/review-$STAMP.md"; fi

RAW="$(mktemp)"
trap 'rm -f "$RAW"' EXIT
pi -p --no-session --model "$MODEL" --thinking "$THINKING" --tools read,bash \
   --skill .pi/skills/rl-run-review "Review $SWEEP" > "$RAW"

{
  echo "<!-- model: $MODEL | thinking: $THINKING | date: $(date -u +%Y-%m-%dT%H:%M:%SZ) | sweep: $SWEEP -->"
  echo
  cat "$RAW"
} > "$OUT/review.md"
echo "saved $OUT/review.md"

python agent_development/scripts/check_review.py "$OUT/review.md" "$SWEEP" | tee "$OUT/check.txt"
