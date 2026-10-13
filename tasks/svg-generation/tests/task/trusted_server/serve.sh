#!/usr/bin/env bash
# Trusted svg-generation server (task seed; uploaded to /opt/frontier-serve and run in place of a submission
# serve.sh, see frontier_ml/runtime/inference.py).
#
# Serves the submitted openai/gpt-oss-20b finetune with SGLang 0.5.9, quantizing its bf16 weights to FP8 at load
# (--quantization fp8). Only the submission's safetensors shards, their index and tokenizer.json (byte-identical to
# the base's, checked by svg_weights.py) are read from /submission/model; config.json, generation_config.json, the
# tokenizer config and the chat template are the base model's, from this directory. Sampling settings come only
# from the trusted client's request (--sampling-defaults openai).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# Grading sets neither; to run this server on your own GPU, point SUBMISSION at a directory holding model/ and
# WORKSPACE at a writable directory (SUBMISSION=/workspace/output WORKSPACE=/tmp/svg-serve bash serve.sh).
SUBMISSION=${SUBMISSION:-/submission} WORKSPACE=${WORKSPACE:-/workspace}
MODEL=$WORKSPACE/served-model
# The runtime starts this script with a minimal environment (env -i), so image ENV values are restated here.
# SGLang JIT-compiles its FP8 quantization kernel at startup with the image's CUDA toolkit; openai-harmony reads
# the o200k_base vocabulary baked into the image (Dockerfile.inference) instead of downloading it.
export TMPDIR=$WORKSPACE/tmp HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
if [[ -z ${TIKTOKEN_ENCODINGS_BASE:-} && -d /opt/tiktoken ]]; then export TIKTOKEN_ENCODINGS_BASE=/opt/tiktoken; fi
mkdir -p "$TMPDIR" "$MODEL"
for name in config.json generation_config.json special_tokens_map.json tokenizer_config.json chat_template.jinja; do
  cp "$HERE/$name" "$MODEL/$name"
done
ln -sf "$SUBMISSION"/model/tokenizer.json "$MODEL/tokenizer.json"
ln -sf "$SUBMISSION"/model/model.safetensors.index.json "$MODEL/model.safetensors.index.json"
for shard in "$SUBMISSION"/model/*.safetensors; do
  ln -sf "$shard" "$MODEL/$(basename "$shard")"
done
exec python3 -m sglang.launch_server \
  --model-path "$MODEL" --tokenizer-path "$MODEL" \
  --served-model-name "${MODEL_NAME:-submission}" --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" \
  --quantization fp8 --dtype bfloat16 --context-length 20480 --mem-fraction-static 0.85 \
  --reasoning-parser gpt-oss --sampling-defaults openai --random-seed 0
