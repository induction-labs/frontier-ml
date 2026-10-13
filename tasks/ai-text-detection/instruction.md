Train one local model to distinguish AI-generated from human-written passages. It needs to work for Arabic, Chinese, English, French, German, Portuguese, Russian and Spanish. I have provided 200 labeled example passages.

Start from any general pretrained backbone, but not one already trained for AI-text detection.

Write a script `serve.sh` which serves the AI-detector. It should have a `POST /detect` endpoint which takes a passage and returns the probability that the passage was written by an AI model. It should run offline on one H100, be ready within 300 seconds, and score at least 10 passages per second.

Optimize AI-text recall at a 1% false-positive rate, averaged over languages, generators and conditions.

Do not train, fine-tune, calibrate or select on the validation or test split of any public dataset or benchmark; training splits and other public corpora are fine. However, you are free to use the validation data samples I have provided however you want.

Submit in `/workspace/output/`. Interface and runtime details: `/task/CONTRACT.md`.
Supporting files: `/task/public/`. Evaluate with `python /task/validate.py --server-url URL`.
Test the finished submission the way it will be run with `frontier selfcheck`.
Hand it in with `frontier submit` when it is final.
Use `spin --help` for compute and model APIs.
