# Pretrain a language model from scratch on 10M words

Do not train, fine-tune, calibrate or select on the validation or test split of any public dataset or benchmark; training splits and other public corpora are fine. Here, though, the corpus is the only data allowed, and your code can't contain precomputed weights, vocabularies, word lists, statistics or other data. This task was inspired by NanoGPT Slowrun (https://github.com/qlabs-eng/slowrun). Looking it up is cheating: do not search for it, and do not open, download or read it, its forks or other derivative repositories, or pages about it. I will check your session transcript for this.

## Server interface

GET /vocab returns {"tokens":[...]}: each token's bytes as lowercase hex, or null for special tokens that never appear in text.
POST /score {"text":str} returns {"token_ids":[int],"logprobs":[float]}. The token bytes must concatenate to the exact UTF-8 input, and logprobs[i] = ln p(token_i | tokens_0..i-1).
POST /dist {"prefix_tokens":[int]} returns {"logprobs":[float]} over the entire vocabulary, with |logsumexp| <= 1e-3. An empty prefix starts a document.
/score and /dist must agree within 0.02 nats, with mean(score - dist) <= 0.002 nats. Scoring a truncated text must keep the log-probabilities of its shared token prefix within 0.02 nats. Each request is independent: no caching across requests.
Each request must answer within 120 seconds, and the whole evaluation (every /score request, a server restart, then the /dist and truncation audits, 8 requests in flight) within 3,600 seconds. A request that fails or times out follows the retry rule below and scores its document as uniform bytes (8 bits per byte); a timed-out audit request is skipped.

## Evaluation

Bits per UTF-8 byte from /score: bpb = sum(-logprobs) / ln(2) / total bytes. Breaking any interface rule fails the run.

## Artifacts

In architecture.json, set base_model="none" and base_revision="from-scratch".

Include architecture.json with base_model, base_revision, training_evidence_files and inference_evidence_files. The file lists use paths relative to output/ and include training source, serve.sh and model-loading code. Keep initialization/training logs with the submission as readable text or JSON. Name the source of every training, calibration and selection corpus (in architecture.json or a data manifest) and keep its download commands and logs with the submission. I will check for training-data provenance. Ship compiled dependencies as unmodified PyPI or PyTorch wheels, installed into /tmp at startup. Prewarmed JIT kernel caches (Triton, TorchInductor) may be included.

## Submission

Put train.sh, serve.sh and their source code in /workspace/output/, which is mounted read-only at /submission.
`bash /submission/train.sh` runs as an unprivileged user with CORPUS_DIR=/corpus (a read-only copy of /task/public/corpus/), OUTPUT_DIR=/replay/output (empty and writable), HOME and TMPDIR on scratch, REPLAY_START (integer epoch seconds, as `date +%s`) and TRAIN_BUDGET_SECONDS=3600. Read data only from CORPUS_DIR and write everything serving needs to OUTPUT_DIR. It must exit 0 within 3600 wall-clock seconds, including tokenization, compilation and saving.
Training and serving each get one H100 80GB SXM, 8 CPU cores and 80 GiB RAM, with Python 3.11 at /opt/conda/bin, PyTorch 2.8.0 (CUDA 12.8), numpy 2.2.6, tokenizers 0.22.1 and safetensors 0.5.3. There are no other Python packages and no network.
After training, OUTPUT_DIR becomes read-only and the corpus and other scratch are removed. `bash /submission/serve.sh` then starts with WEIGHTS_DIR=/replay/output, HOST=0.0.0.0 and PORT=8000, and may load only those training outputs. GET /health must return 200 within 300 seconds, including loading and compilation. The server is also restarted with empty scratch and must answer the same way afterwards.

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
