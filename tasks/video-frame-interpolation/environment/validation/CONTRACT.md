# Generate intermediate video frames from two endpoints

Use a learned model to predict frames from the supplied images and frame indices. Model ensembles, optical-flow warping, blending, and deterministic image processing are allowed, using only permitted weights. Permitted initialization includes general image, video and multimodal pretrained models and standalone optical-flow models that estimate motion between observed frames. General video pretraining or masked reconstruction does not by itself make a model a specialist VFI model. Pre-existing weights specifically trained or fine-tuned to interpolate missing temporal frames or perform frame-rate upconversion are prohibited, including specialist intermediate-flow and frame-synthesis components, adapters, deltas, fine-tuned descendants, mirrors, renamed copies, conversions and quantizations. GIMM-VFI, RIFE and FILM checkpoints are examples. Do not use these specialist models for initialization, inference, ensembles, fallback paths, distillation, or generating pseudo-labels or other training targets during this task. You may use public architectures and source code, train from scratch, and train or fine-tune permitted models into interpolators during this task. Declare every pretrained component's source, revision or checksum, and upstream training purpose in architecture.json or a referenced model manifest; include any teachers and fallback models. The served code must load the declared weights, with no unused decoy model or remote model calls. Predict each interval independently using only that request's two endpoint images as visual context. Do not carry images, predictions, image features, or model updates from other evaluation requests into generation. Keeping fixed model weights, compiled kernels, and image-independent caches across requests is allowed. Do not recognize evaluation clips by fingerprints or retrieve stored reference frames. Do not train, fine-tune, calibrate or select on the validation or test split of any public dataset or benchmark; this also excludes mirrors, renamed copies, extracted frames and derivatives of those splits. Training splits and other public corpora are fine. This task's shipped public examples may be used freely. Incidental exposure during a permitted pretrained checkpoint's upstream training is not itself disqualifying. Submission review checks initialization, teachers, actual inference routes and data use through available code, manifests, logs, checkpoint contents and the recorded development trajectory.

## Server interface

POST /interpolate with {"schema_version": 2, "width": W, "height": H, "frame_count": T, "input_frames": [{"index": 0, "png": "<base64 PNG>"}, {"index": T-1, "png": "<base64 PNG>"}], "target_indices": [1, 2, ..., T-2]}. T is 5, 9, or 17 for 4x, 8x, or 16x interpolation. Exactly two endpoint frames are supplied. Their normalized times are 0 and 1; target frame i is at i/(T-1). No capture or playback FPS is implied. The evaluator sends Accept: application/vnd.frontier.rgb8. Return an HTTP 200 binary body with Content-Type: application/vnd.frontier.rgb8: concatenate each target frame's raw uint8 RGB bytes in target_indices order, row-major H by W by 3, with no header, padding, compression or base64. The body must contain exactly len(target_indices)*H*W*3 bytes. This avoids image-encoding overhead. For legacy clients without that Accept header, JSON PNG output is also supported by the public checker: {"frames": [{"index": 1, "png": "<base64 PNG>"}, ...]} in target_indices order. Input PNGs are non-interlaced 8-bit RGB of exactly W by H, without transparency or animation. Each dimension is at most 4096; the full uncompressed RGB interval is at most 512 MiB. Each encoded PNG is at most 32 MiB; each JSON request or response is at most 768 MiB. Requests are sent one at a time. No clip identifiers, reference frames, or source metadata are sent.

## Evaluation

Compute LPIPS-AlexNet v0.1 and RGB PSNR on generated frames only, averaging frames within each interval and then intervals within each interpolation factor. Sequence FloLPIPS weights the LPIPS feature error at frame i by the difference between PWC-Net forward flows for reference and reconstructed frames i to i+1. Average all adjacent pairs in the complete interval, including its supplied endpoints. Never concatenate unrelated intervals. Use native-resolution 8-bit RGB pixels; no crop, border removal, YUV conversion or image resizing for LPIPS/PSNR. PWC-Net alone internally resizes to multiples of 64 and rescales its predicted flow. If flow differences are exactly zero, use uniform spatial weights. PSNR is capped at 100 dB for exact matches and is diagnostic only. For each factor and each of LPIPS/FloLPIPS, normalized improvement is clip((frozen_baseline_metric - measured_metric) / (frozen_baseline_metric - frozen_target_metric), 0, 1). The score is the mean of these six components. The frozen baseline LPIPS/FloLPIPS means are 4x: 0.04162831/0.05501670, 8x: 0.05581610/0.07831670, and 16x: 0.08391549/0.12332130. The quality-target means are 4x: 0.01946847/0.02447412, 8x: 0.03089009/0.04003089, and 16x: 0.05186808/0.07500480. These are rounded to eight decimal places here; scoring uses full-precision frozen values. Baseline-level or worse quality earns 0 per component; matching or exceeding target quality earns 1. Perfect reconstruction is not required for full credit. Generation must average at least one generated frame per second across the complete evaluation pass, including input reads, request transport, response validation and output export: 2,250 seconds for 2,250 frames. This is a whole-pass cap, not a one-second HTTP request timeout. Startup and quality scoring are excluded. Quality scoring runs offline on a separate H100 after generation shuts down; its time is excluded from the generation gate. Both measured anchors, metric implementation, weights, data and CUDA runtime are hash-locked. The public client accepts --interface-only with an explicit --predictions-dir to check and save outputs; it returns no quality score. Diagnostics report sample counts, generated frames, and request time separately for each interpolation factor. Official grading verifies all frozen calibration inputs before review or GPU provisioning. The diagnostic request ceiling is 300 seconds and the diagnostic whole-pass ceiling is 3600 seconds; these are adjustable debugging limits, not benchmark speed requirements.

## Artifacts

Three public example intervals are shipped. No starting checkpoint is provided. Package your chosen dependencies and weights for offline inference. The local smoke fixture contains no learned model and is only for checking the interface.

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
Every request must answer HTTP 200 with the documented JSON within the task's deadline, and requests 
are never retried. A failed, malformed or late response scores zero for the whole run.

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
