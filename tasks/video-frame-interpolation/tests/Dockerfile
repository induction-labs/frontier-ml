FROM node:24.14.1-bookworm-slim AS frontier_review_sdk
WORKDIR /opt/frontier-review-sdk
COPY lib/frontier_ml/review_agent/package*.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund
FROM python:3.12-slim
# Weight acquisition happens at image build, never during grading. All downloads
# must match the organizer's committed lock; the draft lock deliberately fails.
COPY task/pipeline/metric_bundle.py /opt/vfi-build/metric_bundle.py
COPY task/scorer/metric-lock.json /opt/vfi-build/metric-lock.json
RUN python /opt/vfi-build/metric_bundle.py restore --lock /opt/vfi-build/metric-lock.json --output /opt/vfi-metrics
# The common verifier layers install torch==2.8.0 (CPU), numpy==2.2.6, Pillow==11.3.0.
# This parent validates hashes and orchestrates the run. Actual quality metrics
# run in Dockerfile.gpu after the candidate H100 has been destroyed.

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
