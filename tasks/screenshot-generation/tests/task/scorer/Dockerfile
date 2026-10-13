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
COPY scorer/requirements.txt /tmp/scorer-requirements.txt
RUN pip install --no-cache-dir -r /tmp/scorer-requirements.txt
COPY screenshot_generation.py /tmp/fml/screenshot_generation.py
RUN cd /tmp/fml && python -c "import screenshot_generation as sg; sg.prepare_models('/opt/screenshot-scorer/models')" \
    && rm -rf /tmp/fml
