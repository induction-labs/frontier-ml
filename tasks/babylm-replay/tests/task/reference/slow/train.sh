#!/bin/bash
# Replay entry point: trains on $CORPUS_DIR (read-only) and writes everything serve.sh needs to $OUTPUT_DIR.
set -euo pipefail
export REPLAY_START=${REPLAY_START:-$(date +%s.%N)}
export TORCHINDUCTOR_COMPILE_THREADS=8 RAYON_NUM_THREADS=8 OMP_NUM_THREADS=8 TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/inductor" TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton" USER=${USER:-nobody}
cd "$(dirname "$0")"
exec python train.py --recipe slow --seed 0
