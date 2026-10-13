Convert the MiniCPM5-2B checkpoint in `/artifacts/minicpm5-2b/` into a text diffusion model that actually generates by iteratively denoising tokens, not by left-to-right decoding.

Write a script `serve.sh` which serves it through an OpenAI-compatible chat endpoint. The server should run offline on one H100, be ready within 300 seconds, and generate over 450 tokens/second on a single stream.

Optimize the mean of its language and coding benchmark scores. Our eval setup lets the model reason before it answers, with up to 32,768 generated tokens per request.

Do not train, fine-tune, calibrate or select on the validation or test split of any public dataset or benchmark; training splits and other public corpora are fine.

Submit in `/workspace/output/`. Interface and runtime details: `/task/CONTRACT.md`.
Supporting files: `/task/public/`. Evaluate with `python /task/validate.py --server-url URL`.
Test the finished submission the way it will be run with `frontier selfcheck`.
Hand it in with `frontier submit` when it is final.
Use `spin --help` for compute and model APIs.
