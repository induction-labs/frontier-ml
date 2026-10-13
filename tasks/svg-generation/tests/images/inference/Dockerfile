# Operator-controlled inference image for svg-generation's trusted gpt-oss-20b server (trusted_server/serve.sh):
# the official SGLang v0.5.9 image, at a pinned digest with a clean checkout. It carries
# the CUDA toolkit, which SGLang's online FP8 quantization needs to JIT-compile its quantization kernel at startup.
FROM lmsysorg/sglang@sha256:f337dfb36971becc98d12768f356af3c6c12ba57c9aebede0e6948da5ad37da7
RUN test "$(git -C /sgl-workspace/sglang rev-parse HEAD)" = bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2 \
    && test -z "$(git -C /sgl-workspace/sglang status --porcelain)" \
    && command -v setpriv && command -v python3 && test -x /usr/local/cuda/bin/nvcc \
    && git config --system --add safe.directory /sgl-workspace/sglang \
    && git config --system --add safe.directory /sgl-workspace/sglang/.git
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends librsvg2-bin fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
# openai-harmony (SGLang's gpt-oss chat and reasoning handling) loads the o200k_base vocabulary, which it would
# otherwise download at startup; the grading sandbox is offline. Pinned by digest.
RUN mkdir -p /opt/tiktoken \
    && curl -fsSL https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken -o /opt/tiktoken/o200k_base.tiktoken \
    && echo "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d  /opt/tiktoken/o200k_base.tiktoken" | sha256sum -c - \
    && chmod -R a+rX /opt/tiktoken
ENV HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TIKTOKEN_ENCODINGS_BASE=/opt/tiktoken
RUN mkdir -p /submission /workspace /opt/frontier-grading/requests && chmod 777 /workspace
WORKDIR /tmp
