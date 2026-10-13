#!/bin/bash
# Generate and score baselines on a split: run_split.sh SPLIT_JSONL OUT_ROOT "methods..." ["components"]
set -u
cd "$(dirname "$0")"
W=/scratch/frontier-ml-workspace/work/screenshot-generation
SPLIT=$1; ROOT=$2; METHODS=$3; COMPONENTS=${4:-"T L F R"}
mkdir -p "$ROOT"
for m in $METHODS; do
  runner=../py.sh
  if [ "$m" = zimage ] || [ "$m" = flux2klein ]; then runner=$W/py-gen.sh; fi
  if [ ! -f "$ROOT/out-$m/latency.json" ]; then
    bash $runner generate.py "$m" "$SPLIT" "$ROOT/out-$m" > "$ROOT/gen-$m.log" 2>&1 || { echo "gen $m failed"; continue; }
  fi
  bash ../py.sh score_dir.py "$SPLIT" "$ROOT/out-$m" "$ROOT/score-$m.json" --components $COMPONENTS > "$ROOT/score-$m.log" 2>&1 || { echo "score $m failed"; continue; }
  echo "== $m"; sed -n '/^{/,$p' "$ROOT/score-$m.log" | tr -d '\n' | head -c 600; echo
done
