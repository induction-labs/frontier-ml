#!/usr/bin/env bash
set -euo pipefail

SVG_CALIBRATION_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
SVG_CALIBRATION_IMAGE=sha256:0da16852e69dd2aec24078571b82b62a0190d1c9ed81b4e80e0dea118633bc87
SVG_CALIBRATION_NETWORK=none
SVG_CALIBRATION_KEY=()
if [[ "$#" == 5 && "$1" == --execute && "$2" == --approved-plan-sha256 && "$3" =~ ^[0-9a-f]{64}$ && "$4" == --approved-aggregate-cap-usd && "$5" == 8 ]]; then
  SVG_CALIBRATION_NETWORK=bridge
  SVG_CALIBRATION_KEY=(--env OPENAI_API_KEY)
elif [[ "$#" != 0 && !( "$#" == 1 && "$1" == --help ) ]]; then
  printf '%s\n' 'Usage: run_sol_low_calibration.sh [--execute --approved-plan-sha256 SHA256 --approved-aggregate-cap-usd 8]' >&2
  exit 2
fi
mkdir -p "$SVG_CALIBRATION_ROOT/work/svg-generation/judge-calibration-v3-sol-low"
docker run --rm --network "$SVG_CALIBRATION_NETWORK" --read-only \
  --cpus 2 --memory 1g --pids-limit 128 --cap-drop ALL --security-opt no-new-privileges \
  --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,nosuid,nodev,size=256m,mode=1777 \
  --env PYTHONPATH=/support "${SVG_CALIBRATION_KEY[@]}" \
  --mount "type=bind,src=$SVG_CALIBRATION_ROOT/tasks/svg-generation,dst=/seed,readonly" \
  --mount "type=bind,src=$SVG_CALIBRATION_ROOT/frontier_ml,dst=/support/frontier_ml,readonly" \
  --mount "type=bind,src=$SVG_CALIBRATION_ROOT/data/svg-generation,dst=/data,readonly" \
  --mount "type=bind,src=$SVG_CALIBRATION_ROOT/work/svg-generation/judge-calibration-v1,dst=/history/luna,readonly" \
  --mount "type=bind,src=$SVG_CALIBRATION_ROOT/work/svg-generation/judge-calibration-v2-sol,dst=/history/sol,readonly" \
  --mount "type=bind,src=$SVG_CALIBRATION_ROOT/work/svg-generation/judge-calibration-v3-sol-low,dst=/out" \
  "$SVG_CALIBRATION_IMAGE" python -B /seed/pipeline/calibrate_sol_low.py \
  --data-dir /data --prior-dir /history --work-dir /out "$@"
