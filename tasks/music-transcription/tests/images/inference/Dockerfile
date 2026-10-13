FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04
# gcc and headers: Triton (used by torch.compile and vLLM) compiles its CUDA driver helper at startup.
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-venv python3-dev gcc libc6-dev bash util-linux iproute2 ca-certificates && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 FRONTIER_MLE_VERIFIER=1
RUN pip install --no-cache-dir httpx==0.28.1 numpy==2.2.6 pillow==11.3.0 safetensors==0.5.3
RUN mkdir -p /submission /workspace/output /logs/verifier /tests && chmod 700 /tests /logs/verifier
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchaudio==2.8.0 soundfile==0.13.1 requests==2.34.2 urllib3==2.8.0 idna==3.20 certifi==2026.7.22 charset-normalizer==3.5.1
WORKDIR /tmp
