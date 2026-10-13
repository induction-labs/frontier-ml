FROM node:24.14.1-bookworm-slim AS frontier_review_sdk
WORKDIR /opt/frontier-review-sdk
COPY lib/frontier_ml/review_agent/package*.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_HOME=/opt/frontier-rustup CARGO_HOME=/opt/frontier-cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup.sh \
    && sh /tmp/rustup.sh -y --profile minimal --default-toolchain 1.89.0
ENV PATH=/opt/frontier-cargo/bin:$PATH
COPY task/foul-play-requirements.txt task/setup_foul_play.sh /tmp/foul-play-setup/
RUN sh /tmp/foul-play-setup/setup_foul_play.sh /opt/frontier-foul-play
ENV FOUL_PLAY_PYTHON=/opt/frontier-foul-play/bin/python

# Standard verifier layers on top of the seed's verifier.Dockerfile.
RUN apt-get update && apt-get install -y --no-install-recommends bash git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends nodejs && rm -rf /var/lib/apt/lists/*
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
