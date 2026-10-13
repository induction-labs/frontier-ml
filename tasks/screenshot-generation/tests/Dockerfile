FROM node:24.14.1-bookworm-slim AS frontier_review_sdk
WORKDIR /opt/frontier-review-sdk
COPY lib/frontier_ml/review_agent/package*.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund
# Trusted verifier base for screenshot-generation (task.yaml verifier.base_dockerfile). frontier build inlines
# this stage into tests/Dockerfile (COPY sources are relative to the task seed, which ships as tests/task/)
# and appends the standard verifier layers, so Harbor/Modal build it from source. Same python:3.12-slim base as
# every other verifier; CUDA/cuDNN come from pip (onnxruntime-gpu[cuda,cudnn]) and are loaded by
# onnxruntime.preload_dlls(). The pinned OCR is baked in, so grading needs no network for models:
#   PP-OCRv5 server det/rec (rapidocr 3.9.2, onnxruntime-gpu 1.23.2). Faithfulness and realism use gpt-6-luna.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends bash git ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_TELEMETRY=1
COPY task/scorer/requirements.txt /tmp/scorer-requirements.txt
RUN pip install --no-cache-dir -r /tmp/scorer-requirements.txt
COPY task/screenshot_generation.py /tmp/fml/screenshot_generation.py
RUN cd /tmp/fml && python -c "import screenshot_generation as sg; sg.prepare_models('/opt/screenshot-scorer/models')" \
    && rm -rf /tmp/fml

# Standard verifier layers on top of the seed's scorer/Dockerfile.
RUN apt-get update && apt-get install -y --no-install-recommends bash git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
RUN pip install --no-cache-dir onnx==1.19.0 gguf==0.19.0
RUN pip install --no-cache-dir 'harbor[modal,daytona]==0.22.0' modal==1.5.5 anthropic==1.8.0 httpx==0.28.1 numpy==2.2.6 pillow==11.3.0 pyyaml==6.0.2 safetensors==0.8.0 'transformers>=4.55,<6'
ENV PYTHONDONTWRITEBYTECODE=1
# Harbor builds this image from tests/ and does not upload tests afterwards.
COPY . /tests/
RUN mkdir -p /submission /workspace/output /logs/verifier /tests && chmod 700 /tests /logs/verifier
RUN env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin python -c 'import httpx, yaml' || (echo 'verifier base: python on the default PATH that test.sh runs with lacks the verifier packages' >&2; exit 1)
WORKDIR /tmp
COPY --from=frontier_review_sdk /usr/local/bin/node /usr/local/bin/node
COPY --from=frontier_review_sdk /opt/frontier-review-sdk/node_modules /tests/lib/frontier_ml/review_agent/node_modules
