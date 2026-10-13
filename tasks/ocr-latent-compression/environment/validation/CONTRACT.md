# Compress page images into 2 KB of discrete codes that still decode to the full text

Decoding must depend only on the codes and be deterministic. Include training and inference code with the submitted weights. Do not train, fine-tune, calibrate or select on the validation or test split of any public dataset or benchmark; training splits and other public corpora are fine.

## Server interface

POST /encode with {"image": "<base64 PNG>"} returns exactly {"codes": [<int>, ...], "codebook_size": <int>}. Every code is in [0, codebook_size), codebook_size is the same in every response, and len(codes) x ceil(log2(codebook_size)) is at most 16384 bits. codebook_size describes the serialized integer alphabet; scalar levels, learned codebook indices and codes from multiple codebooks may be serialized into it. POST /decode with {"codes": [...], "codebook_size": <int>} returns exactly {"text": "<page text>"}: plain text or markdown, up to 32768 characters. Images are English document pages (1275x1650) and application screenshots (1920x1080). Encoding and decoding run in separate server instances with no shared per-image state, at up to 16 concurrent requests.

## Evaluation

Mean word-level edit similarity between the decoded text and the page text after Unicode, markup and whitespace normalization; case and punctuation count. Malformed responses and code lists over the bit budget score zero for that image. Each endpoint must sustain at least one image per second after readiness.

## Artifacts

DeepSeek-OCR weights, tokenizer and reference code are in /artifacts/deepseek-ocr/weights/ (revision and file hashes in public/STARTING_ARTIFACT.json); you may start from them, but no starting checkpoint is prescribed. Add codebook_size to architecture.json (example in /task/public/SCHEMA.md). The Python 3.12/CUDA 12.8 inference environment includes torch==2.6.0, torchvision==0.21.0, transformers==4.46.3, tokenizers==0.20.3, einops==0.8.2, addict==2.4.0 and easydict==1.13.

Include architecture.json with base_model, base_revision, training_evidence_files and inference_evidence_files. The file lists use paths relative to output/ and include training source, serve.sh and model-loading code. Keep initialization/training logs with the submission as readable text or JSON. Name the source of every training, calibration and selection corpus (in architecture.json or a data manifest) and keep its download commands and logs with the submission. I will check for training-data provenance. Ship compiled dependencies as unmodified PyPI or PyTorch wheels, installed into /tmp at startup. Prewarmed JIT kernel caches (Triton, TorchInductor) may be included.

## Submission

Put serve.sh, code, weights and dependencies in /workspace/output/ as regular files. Relative
symlinks that resolve inside output/ are copied as the regular files they point to. However, absolute, escaping,
dangling or cyclic symlinks fail the submission (for example, use `python -m venv --copies`, or vendor wheels).
The /workspace/output folder is copied to /submission when being evaluated. serve.sh runs as an unprivileged 
user that can write to /submission, /tmp and the preinstalled Python environment (/opt/venv) and nowhere else; 
install vendored wheels there or under /submission. You should package dependencies for offline installation,
since the evaluation server will not have internet access.
The command `bash /submission/serve.sh` receives HOST=0.0.0.0, PORT=8000 and
MODEL_NAME=submission. GET /health must return 200 when ready, within 300 seconds.
Startup and inference run offline on one H100 SXM 80GB, with 8 CPU cores and 80 GiB RAM.
The serving image is Ubuntu 24.04 with the CUDA 12.8 runtime libraries (no CUDA compiler), Python 3.12 in /opt/venv first on PATH (pip, numpy, pillow, safetensors, httpx preinstalled), gcc and Python headers. Also preinstalled: torch==2.6.0, torchvision==0.21.0, transformers==4.46.3, tokenizers==0.20.3, einops==0.8.2, addict==2.4.0, easydict==1.13. Vendor cp312 manylinux wheels for anything else.
Every request must answer HTTP 200 with the documented JSON. A request that fails (non-200, connection
error, invalid JSON) is retried twice on the same payload. If it still fails, or answers in an undocumented shape, 
you get no credit for that item. Up to one evaluation request in twenty may fail that way, but more than that gives 
you no credit for the entire run.

## Development

The development machine is CPU-only. Use `spin --help` to rent compute and access
model APIs, and `spin budget` to check how much allowance you have left.
The API key your coding harness was started with is for the harness alone. Do not use the API key
in your own code, scripts, curl or Spin machines, copy it anywhere, or start extra harness sessions as workers;
AI model calls you make yourself must go through `spin` and spend your allowance. I will check your session transcript for this.

To check your solution locally, run `python /task/validate.py --server-url URL`. See /task/public for examples 
and supporting files.

You can check your solution on the same infrastructure I will evaluate it on using `frontier selfcheck`,
which copies /workspace/output/, starts it as the evaluation does, sends public examples through the same 
response checks and measures the speed requirements on public inputs. Its report gives the stage reached, 
the digest of the files it checked, the GPU, the startup time and each requirement's measured value and 
margin, and the end of the server log. You can check your solution four times.

A check usually takes 10-40 minutes (`frontier selfcheck status` shows this task's limit) and runs while you keep
working. See `frontier selfcheck --help` for more details.

When your solution is final, hand it in with `frontier submit`: it shows what /workspace/output/ holds and asks
you to confirm with `frontier submit --confirm`, which submits the folder as it is and ends your session. Without
a submission, /workspace/output/ is graded as it is when your session ends. See `frontier submit --help`.

Your session ends the moment you finish a turn without a tool call: background commands, scheduled
wake-ups and reminders will not bring you back, and anything not yet in /workspace/output/ is lost.
Wait for long jobs with foreground commands under the tool time limit (for example
`sleep 600; tail -n 3 train.log`, repeated) and stop only when the final files are in place.
