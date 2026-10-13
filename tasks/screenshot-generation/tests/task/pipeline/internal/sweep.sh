#!/bin/bash
# Generate and score real, zimage, compositor, html and blank with the four-part screenshot metric.
set -u
W=/scratch/frontier-ml-workspace/work/screenshot-generation
D=${1:-/scratch/frontier-ml-workspace/frontier-mle-clean/data/screenshot-generation-internal}
RUNS=${2:-$W/runs-internal}
B=$(cd "$(dirname "$0")/../baselines" && pwd)
export SCREENSHOT_PROMPT_KIND=screen
mkdir -p $RUNS/val $RUNS/test
for split in val test; do
  if [ $split = val ]; then J=$D/public/validation.jsonl; M="real zimage compositor html blank"
  else J=$D/private/test.jsonl; M="real zimage compositor blank"; fi
  bash "$B/run_split.sh" "$J" "$RUNS/$split" "$M" "T L F R" > "$RUNS/$split/summary.txt" 2>&1
done
echo ALLDONE
