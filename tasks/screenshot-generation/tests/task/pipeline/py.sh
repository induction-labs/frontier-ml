#!/bin/bash
# Run the task's work venv with its caches and GPU pinned (GPU 4 only).
W=/scratch/frontier-ml-workspace/work/screenshot-generation
export HF_HOME=$W/hf UV_CACHE_DIR=$W/uv-cache PIP_CACHE_DIR=$W/pip-cache TMPDIR=$W/tmp
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4} PLAYWRIGHT_BROWSERS_PATH=$W/ms-playwright FONTCONFIG_FILE=$W/fonts/fonts.conf
export VLLM_CACHE_ROOT=$W/vllm-cache TRITON_CACHE_DIR=$W/triton-cache XDG_CACHE_HOME=$W/xdg-cache
export PYTHONPATH=/scratch/frontier-ml-workspace/frontier-mle-clean/.claude/worktrees/agent-a28fddb20d7b5122b${PYTHONPATH:+:$PYTHONPATH}
exec $W/venv/bin/python "$@"
