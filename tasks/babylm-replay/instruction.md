Train a causal language model from scratch, tokenizer included, using only the 10M-word corpus in `/task/public/corpus/`.

Submit code, not weights: a script `train.sh` which trains the model and a script `serve.sh` which serves its normalized next-token probabilities. I'll run `train.sh` myself, offline on one H100, and it has to finish within 60 minutes. The server should be ready within 300 seconds.

Minimize bits per byte.

This task was inspired by NanoGPT Slowrun (https://github.com/qlabs-eng/slowrun). Looking it up is cheating: do not search for it, and do not open, download or read it, its forks or other derivative repositories, or pages about it. I will check your session transcript for this.

Do not train, fine-tune, calibrate or select on the validation or test split of any public dataset or benchmark; training splits and other public corpora are fine.

Submit in `/workspace/output/`. Interface and runtime details: `/task/CONTRACT.md`.
Supporting files: `/task/public/`. Evaluate with `python /task/validate.py --server-url URL`.
Test the finished submission the way it will be run with `frontier selfcheck`.
Hand it in with `frontier submit` when it is final.
Use `spin --help` for compute and model APIs.
