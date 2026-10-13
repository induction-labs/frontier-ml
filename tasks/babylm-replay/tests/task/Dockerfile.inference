# Pinned replay image for babylm-replay: the candidate's train.sh AND serve.sh both run here, offline.
# It provides torch and a few small pure-code libraries; it contains no model weights, datasets or HF caches.
FROM pytorch/pytorch@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385
RUN apt-get update && apt-get install -y --no-install-recommends bash util-linux procps coreutils gcc libc6-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==2.2.6 tokenizers==0.22.1 safetensors==0.5.3 \
    && rm -rf /root/.cache \
    && python -c "import torch, numpy, tokenizers, safetensors; print(torch.__version__)"
# No pretrained weights or HF caches may exist anywhere in the image.
RUN python - <<'PY'
import os
bad = []
for root, dirs, files in os.walk("/", topdown=True):
    dirs[:] = [d for d in dirs if os.path.join(root, d) not in ("/proc", "/sys", "/dev")]
    if os.path.basename(root) == "huggingface":
        bad.append(root)
    bad += [os.path.join(root, f) for f in files if f.endswith((".safetensors", ".gguf")) or f.startswith("pytorch_model")]
assert not bad, bad[:20]
PY
ENV PATH="/opt/conda/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 FRONTIER_MLE_VERIFIER=1
RUN mkdir -p /submission /corpus /replay /workspace /logs/verifier /tests && chmod 700 /tests /logs/verifier
WORKDIR /tmp
