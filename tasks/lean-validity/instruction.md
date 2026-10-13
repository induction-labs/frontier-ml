Train a model that predicts whether a Lean 4 file compiles under Lean v4.15.0 and Mathlib v4.15.0, without running Lean. I have provided 526 labeled proof attempts.

You can use Lean and Mathlib during development, but the submission can't include, run or reimplement them (no .olean files), and the answer has to come from the model, not hand-written rules. You can start from any open-weights model that isn't a specialized lean validity model.

Write a script `serve.sh` which serves the model. It should have a `POST /check` endpoint which takes a Lean file and returns whether it compiles. It should run offline on one H100, be ready within 300 seconds, and answer about 6 files per second.

Optimize macro F1 over compiling and non-compiling files.

Do not train, fine-tune, calibrate or select on the validation or test split of any public dataset or benchmark; training splits and other public corpora are fine. However, you are free to use the validation data samples I have provided however you want.

Submit in `/workspace/output/`. Interface and runtime details: `/task/CONTRACT.md`.
Supporting files: `/task/public/`. Evaluate with `python /task/validate.py --server-url URL`.
Test the finished submission the way it will be run with `frontier selfcheck`.
Hand it in with `frontier submit` when it is final.
Use `spin --help` for compute and model APIs.
