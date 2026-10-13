# Train a small draft model that speeds up greedy speculative decoding for Qwen3-8B

The drafter only proposes tokens: the verifier holds the target's greedy continuation of every held-out conversation, so you do not serve the target. Answer each request from its tokens and your submitted files; caching across requests (for example a prefix cache) is fine. Do not train, fine-tune, calibrate or select on the validation or test split of any public dataset or benchmark; training splits and other public corpora are fine.

## Server interface

POST /draft with {"tokens": [<int>, ...]}: the target tokenizer's ids for the conversation formatted with its chat template (add_generation_prompt=True, enable_thinking=False), followed by the part of the target's continuation accepted so far. Return {"draft": [<int>, ...]} with 0 to 16 token ids. Conversation prompts are at most 1,536 tokens and continuations at most 256 tokens.

## Evaluation

The held-out set is 200 new conversations like the public examples, in the same eight categories, with 44,314 continuation tokens: five times the 40 public conversations (8,784 tokens). For each one the verifier replays greedy speculative decoding against the target's greedy continuation (at most 256 tokens, ending with <|im_end|> when the target stops): the target accepts the longest prefix of your draft that matches its continuation, then adds one token of its own, and the next request carries everything accepted. The mean acceptance length tau is continuation tokens divided by verification steps over the whole held-out set, and the score is (tau - 4.6) / 2.4, clamped to [0, 1]. Every unfinished conversation has one request in flight at a time, up to 32 at once. Each request must answer within 60 seconds; one that has not is failed without a retry, scores as an empty draft (the target adds its own token and the conversation continues), still counts its 60 seconds against the serving budget, and counts toward the limit of one failed request in twenty. A submission over 1,000,000,000 bytes (every file counted) scores zero before it is started, and so does a server whose total request time over the held-out evaluation exceeds 10,800 seconds; validate.py reports your public serving time scaled to the held-out size. The public examples are not in the held-out set, so memorizing them does not help.

## Artifacts

/artifacts/qwen3-8b/ holds the tokenizer and configuration of the target, Qwen/Qwen3-8B at revision b968826d9c46dd6066d109eabc6255188de91218 (its weights are on Hugging Face at that revision). No drafter checkpoint is prescribed. Record the checkpoint you started from in architecture.json as base_model and base_revision (an immutable commit), or "none" and "from-scratch" if you use no pretrained model. Keep your data download scripts and training logs with the submission: the review checks where your training data came from.

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
The serving image is Ubuntu 24.04 with the CUDA 12.8 runtime libraries (no CUDA compiler), Python 3.12 in /opt/venv first on PATH (pip, numpy, pillow, safetensors, httpx preinstalled), gcc and Python headers. Also preinstalled: torch==2.8.0, transformers==4.57.1. Vendor cp312 manylinux wheels for anything else.
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
