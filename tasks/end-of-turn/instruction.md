Train a model that listens to a two-person conversation and predicts when each speaker has finished their turn, as opposed to pausing or hesitating mid-turn. I have provided 8 labeled example conversations.

The model has to be causal: its output at any moment can only depend on audio up to that moment.

Write a script `serve.sh` which serves the model. It should have a `POST /predict` endpoint which takes both speakers' audio and returns, for each speaker, a per-frame probability that their turn just ended. It should run offline on one H100, be ready within 300 seconds, and process audio at least 3x faster than real time.

Optimize end-of-turn recall at a 5% false-positive rate on mid-turn pauses.

Do not train, fine-tune, calibrate or select on the validation or test split of any public dataset or benchmark; training splits and other public corpora are fine. However, you are free to use the validation data samples I have provided however you want.

Submit in `/workspace/output/`. Interface and runtime details: `/task/CONTRACT.md`.
Supporting files: `/task/public/`. Evaluate with `python /task/validate.py --server-url URL`.
Test the finished submission the way it will be run with `frontier selfcheck`.
Hand it in with `frontier submit` when it is final.
Use `spin --help` for compute and model APIs.
