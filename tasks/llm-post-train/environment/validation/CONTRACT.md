# Post-train Qwen3-4B-Base into a strong chat model

The submitted weights must descend from the pinned Qwen3-4B-Base checkpoint alone: do not start from, merge in or serve any other checkpoint's weights (that includes other post-trained versions of this model). Do not train, fine-tune, calibrate or select on the validation or test split of any public dataset or benchmark; training splits and other public corpora are fine. That includes their prompts without answers, rewrites of them, and model answers to them. The examples in /task/public/benchmarks/ are for evaluation only.

## Server interface

OpenAI-compatible POST /v1/chat/completions with model=submission, supporting a 20,480-token context (input plus output). Only message.content is scored: return the model's reasoning in reasoning_content (for example with a reasoning parser) or before the final answer in content.

## Evaluation

First, a weight audit reads the checkpoint in output/model/: it must consist of exactly Qwen3-4B-Base's tensors (the pinned names and shapes, in bf16, fp16 or fp32; the output head may be untied by adding lm_head.weight), or the score is zero. The review then establishes from your training evidence and your session that these weights descend from the pinned base and from no other checkpoint.
The score is the mean of accuracy/pass@1 across four benchmarks, two of which are GPQA-Diamond and LCB v6, each normalized from the untouched Qwen3-4B-Base's performance (0; chance level where the base scores below chance) to the performance of a strong post-trained model of the same size (1). Requests use max_tokens=16384 and temperature=0.7 and arrive up to 32 at a time, and each gets 250 seconds. A reply cut off at max_tokens scores zero for its item, as does a request that times out (it is not retried) or that still fails after two retries; failed requests do not stop the evaluation.

## Artifacts

/artifacts/qwen3-4b-base/ holds Qwen/Qwen3-4B-Base at the revision in /task/public/STARTING_ARTIFACT.json; take architecture.json's base_model and base_revision from there. Put the final trained weights in output/model/ as unquantized bf16, fp16 or fp32 safetensors with the original tensor names and shapes, next to config.json and the tokenizer files; only output/model/*.safetensors are audited. serve.sh may serve those weights directly or a quantized copy derived from them that you also keep in output/.
In your training evidence, record the datasets, splits and data-generation recipes you used, and every model you ran or called to generate, score or filter data.

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
The serving image is Ubuntu 24.04 with the CUDA 12.8 runtime libraries (no CUDA compiler), Python 3.12 in /opt/venv first on PATH (pip, numpy, pillow, safetensors, httpx preinstalled), gcc and Python headers. PyTorch is not preinstalled: vendor cp312 manylinux wheels (and their dependencies).
Every request must answer HTTP 200 with the documented JSON. A request that fails (non-200, connection
error, invalid JSON) is retried twice on the same payload. If it still fails, or answers in an undocumented shape,
you get no credit for that item. However many requests fail, the evaluation goes on.

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
