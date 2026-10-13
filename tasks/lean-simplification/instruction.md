Build a system that takes a Lean 4 theorem and its proof and returns a shorter proof of the same theorem. Proofs are checked with Lean v4.15.0 and its matching Mathlib, and you can run Lean at inference time too.

Write a script `serve.sh` which serves it at a `POST /simplify` endpoint. It should run offline on one H100, be ready within 300 seconds, and finish every batch of 8 concurrent requests within 60 seconds.

Optimize the average relative reduction in proof tokens; proofs that don't check count as zero.

Do not train, fine-tune, calibrate or select on the validation or test split of any public dataset or benchmark; training splits and other public corpora are fine.

Submit in `/workspace/output/`. Interface and runtime details: `/task/CONTRACT.md`.
Supporting files: `/task/public/`. Evaluate with `python /task/validate.py --server-url URL`.
Test the finished submission the way it will be run with `frontier selfcheck`.
Hand it in with `frontier submit` when it is final.
Use `spin --help` for compute and model APIs.
