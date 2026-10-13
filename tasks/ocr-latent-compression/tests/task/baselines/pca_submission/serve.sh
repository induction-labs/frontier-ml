#!/bin/bash
# Organizer baseline. Local development only: uses $OCR_PYTHON (a Python with torch,
# transformers==4.46.3, easydict, einops, addict, safetensors, pillow). A real submission must
# vendor its dependencies for the offline inference image.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_MODULES_CACHE="${TMPDIR:-/tmp}/hf-modules-$$" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "${OCR_PYTHON:-python3}" "$HERE/server.py"
