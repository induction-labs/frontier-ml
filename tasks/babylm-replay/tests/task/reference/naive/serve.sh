#!/bin/bash
# Serves the replayed weights in $WEIGHTS_DIR on $HOST:$PORT.
set -euo pipefail
cd "$(dirname "$0")"
exec python server.py
