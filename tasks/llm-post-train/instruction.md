Post-train the pretrained Qwen3-4B-Base checkpoint in `/artifacts/qwen3-4b-base/` into a chat model that solves science, math and coding problems as well as you can make it. How you get there is up to you. The one rule is that the weights you submit must descend from that checkpoint and from no other model's weights.

Write `serve.sh` to serve the result through an OpenAI-compatible chat endpoint offline on one H100, ready within 300 seconds, and put the trained weights in `/workspace/output/model/`.

Optimize the mean of language and coding benchmark scores. Our eval setup lets the model reason before it answers, with up to 16,384 generated tokens per request at temperature 0.7.

Do not train, fine-tune, calibrate or select on the validation or test split of any public dataset or benchmark; training splits and other public corpora are fine.

Submit in `/workspace/output/`. Interface and runtime details: `/task/CONTRACT.md`.
Supporting files: `/task/public/`. Evaluate with `python /task/validate.py --server-url URL`.
Test the finished submission the way it will be run with `frontier selfcheck`.
Hand it in with `frontier submit` when it is final.
Use `spin --help` for compute and model APIs.
