Train a model that infers keyboard and mouse actions from five-second screen recordings. I have provided one labeled example clip.

The test clips come from `p-doom/idm-eval` on Hugging Face. Don't download, look at or train on its clips or labels, or copies of them. The example clip is from there too; you may use that one clip, but no other clip from it. You can start from any pretrained base model, as long as it isn't a specialist IDM model (e.g., you may not start from p-doom/idm).

Write a script `serve.sh` which serves the model. It should have a `POST /actions` endpoint which takes an MP4 and returns the actions it sees, labeled by frame. It should run offline on one H100, and be ready within 300 seconds.

Optimize action F1, micro-averaged over key presses, clicks, mouse moves and scrolls.

Do not train, fine-tune, calibrate or select on the validation or test split of any public dataset or benchmark; training splits and other public corpora are fine. However, you are free to use the validation data samples I have provided however you want.

Submit in `/workspace/output/`. Interface and runtime details: `/task/CONTRACT.md`.
Supporting files: `/task/public/`. Evaluate with `python /task/validate.py --server-url URL`.
Test the finished submission the way it will be run with `frontier selfcheck`.
Hand it in with `frontier submit` when it is final.
Use `spin --help` for compute and model APIs.
