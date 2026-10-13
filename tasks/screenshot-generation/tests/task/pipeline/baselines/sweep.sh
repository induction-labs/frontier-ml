#!/bin/bash
# Generate and score all baselines with the OCR, faithfulness and realism metric.
set -u
W=/scratch/frontier-ml-workspace/work/screenshot-generation
D=${1:-/scratch/frontier-ml-workspace/frontier-mle-clean/data/screenshot-generation-internal}
RUNS=${2:-$W/runs}
cd "$(dirname "$0")"
export SCREENSHOT_PROMPT_KIND=screen
mkdir -p "$RUNS/val" "$RUNS/test"
for split in val test; do
  if [ $split = val ]; then J=$D/public/validation.jsonl; else J=$D/private/test.jsonl; fi
  bash run_split.sh "$J" "$RUNS/$split" "real zimage flux2klein compositor html blank" "T L F R" > "$RUNS/$split/summary.txt" 2>&1
done
echo ALLDONE
