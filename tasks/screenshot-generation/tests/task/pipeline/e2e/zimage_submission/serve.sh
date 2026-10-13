#!/bin/bash
# LOCAL e2e harness only: training-free Z-Image-Turbo baseline (not self-contained; uses the dev venv and weights).
exec /scratch/frontier-ml-workspace/work/screenshot-generation/venv-gen/bin/python "$(dirname "$0")/server.py" zimage
