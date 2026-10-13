#!/usr/bin/env bash
# v4 judge calibration in the pinned official renderer image.
#   run_v4_calibration.sh                      prepare/refresh the frozen plan offline (no key, no network)
#   run_v4_calibration.sh --execute --approved-plan-sha256 SHA256 --approved-cap-usd 5
#                                              PAID: OPENAI_API_KEY from the operator's environment
#   run_v4_calibration.sh --replay             offline wire replay -> work dir wire_replay.json
# Promotion runs on the host: python tasks/svg-generation/pipeline/calibrate_v4.py --promote ...
set -euo pipefail

SVG_CALIBRATION_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
SVG_CALIBRATION_IMAGE=sha256:0da16852e69dd2aec24078571b82b62a0190d1c9ed81b4e80e0dea118633bc87
SVG_CALIBRATION_WORK="$SVG_CALIBRATION_ROOT/work/svg-generation/judge-calibration-v4-semantic-r2"
SVG_CALIBRATION_NETWORK=none
SVG_CALIBRATION_KEY=()
SVG_CALIBRATION_COMMAND=(python -B /seed/pipeline/calibrate_v4.py --data-dir /data --prior-dir /history --work-dir /out)
if [[ "$#" == 5 && "$1" == --execute && "$2" == --approved-plan-sha256 && "$3" =~ ^[0-9a-f]{64}$ && "$4" == --approved-cap-usd && "$5" == 5 ]]; then
  SVG_CALIBRATION_NETWORK=bridge
  SVG_CALIBRATION_KEY=(--env OPENAI_API_KEY)
  SVG_CALIBRATION_COMMAND+=("$@")
elif [[ "$#" == 1 && "$1" == --replay ]]; then
  SVG_CALIBRATION_COMMAND=(python -B /seed/pipeline/replay_calibrated_wire.py --work-dir /out --data-path /data/private/test.jsonl --output /out/wire_replay.json)
elif [[ "$#" != 0 && !( "$#" == 1 && "$1" == --help ) ]]; then
  printf '%s\n' 'Usage: run_v4_calibration.sh [--replay | --execute --approved-plan-sha256 SHA256 --approved-cap-usd 5]' >&2
  exit 2
fi
# The pinned image has only the standard library; frontier_ml.taskspec needs PyYAML from the checkout.
SVG_CALIBRATION_YAML=$(cd "$SVG_CALIBRATION_ROOT" && uv run python -c 'import os, yaml; print(os.path.dirname(yaml.__file__))' 2>/dev/null)
mkdir -p "$SVG_CALIBRATION_WORK"
docker run --rm --network "$SVG_CALIBRATION_NETWORK" --read-only \
  --cpus 2 --memory 1g --pids-limit 128 --cap-drop ALL --security-opt no-new-privileges \
  --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,nosuid,nodev,size=256m,mode=1777 \
  --env PYTHONPATH=/support "${SVG_CALIBRATION_KEY[@]}" \
  --mount "type=bind,src=$SVG_CALIBRATION_ROOT/tasks/svg-generation,dst=/seed,readonly" \
  --mount "type=bind,src=$SVG_CALIBRATION_ROOT/frontier_ml,dst=/support/frontier_ml,readonly" \
  --mount "type=bind,src=$SVG_CALIBRATION_YAML,dst=/support/yaml,readonly" \
  --mount "type=bind,src=$SVG_CALIBRATION_ROOT/data/svg-generation,dst=/data,readonly" \
  --mount "type=bind,src=$SVG_CALIBRATION_ROOT/work/svg-generation/judge-calibration-v1,dst=/history/luna,readonly" \
  --mount "type=bind,src=$SVG_CALIBRATION_ROOT/work/svg-generation/judge-calibration-v2-sol,dst=/history/sol_none,readonly" \
  --mount "type=bind,src=$SVG_CALIBRATION_ROOT/work/svg-generation/judge-calibration-v3-sol-low,dst=/history/sol_low,readonly" \
  --mount "type=bind,src=$SVG_CALIBRATION_ROOT/work/svg-generation/judge-calibration-v4-semantic,dst=/history/v4_semantic,readonly" \
  --mount "type=bind,src=$SVG_CALIBRATION_WORK,dst=/out" \
  "$SVG_CALIBRATION_IMAGE" "${SVG_CALIBRATION_COMMAND[@]}"
