# Detect when a speaker has finished talking so voice assistants reply promptly without interrupting

Do not train, fine-tune, calibrate or select on the validation or test split of any public dataset or benchmark; training splits and other public corpora are fine. However, you are free to use the validation data samples I have provided however you want. Do not use any other copy or derivative of this task's conversations or their labels, beyond the 8 shipped examples.

## Server interface

POST /predict receives {"audio_format": "flac", "speaker_1_audio": "<base64>", "speaker_2_audio": "<base64>"}: two aligned mono 48 kHz, 24-bit FLAC recordings of the whole conversation. Return {"speaker_1": {"hop_seconds": h, "eot_probability": [p_0, p_1, ...]}, "speaker_2": {"hop_seconds": h, "eot_probability": [...]}}, where h is a fixed hop per speaker between 0.02 and 0.20 s. Frame i covers (i*h, (i+1)*h] and is reported at time (i+1)*h: p_i, in [0, 1], is the probability that this speaker's turn just ended, computed from either channel only through (i+1)*h (lookahead means reporting later, never by backdating). Return ceil(duration / h) frames per speaker; the stream must end within one hop of the audio duration.

## Evaluation

Each labeled end of turn at time t gets a positive window [t - 0.25 s, min(t + 1 s, that speaker's next end of turn)], and each mid-turn pause [s, e] gets a negative window [s - 0.25 s, e]. A window's value is the highest probability reported inside it, skipping frames in that speaker's disputed regions (gold.eot_excluded in validation.jsonl) and, for pauses, frames inside a positive window. For a threshold T, recall(T) and FPR(T) are the fractions of positive and pause windows with value >= T, pooled over all held-out conversations; the score is the highest recall(T) with FPR(T) <= 0.05.

Conversations are 10 to 15 minutes long (about 100 MB of FLAC per request) and are sent one at a time. The server must handle the whole held-out pass (5.7 hours of audio) at least 3x faster than real time: at most 6,818 s of server time in total.

## Artifacts

You can start from any checkpoint, or from scratch (base_model "none", base_revision "from-scratch"). Include your data download scripts.

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
