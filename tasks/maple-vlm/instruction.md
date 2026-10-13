Add a vision encoder and connector to Maple, the ternary language model in `/artifacts/maple/`. You can use a pretrained encoder and fine-tune Maple, but keep its language network ternary. Don't add another language model or an image-to-text front end.

Write a script `serve.sh` which serves the model through an OpenAI-compatible `POST /v1/chat/completions` endpoint. The server should work offline on one H100 and be ready within 300 seconds.

Optimize accuracy on college-level questions about images.

Do not train, fine-tune, calibrate or select on the validation or test split of any public dataset or benchmark; training splits and other public corpora are fine. However, you are free to use the validation data samples I have provided however you want.

Submit in `/workspace/output/`. Interface and runtime details: `/task/CONTRACT.md`.
Supporting files: `/task/public/`. Evaluate with `python /task/validate.py --server-url URL`.
Test the finished submission the way it will be run with `frontier selfcheck`.
Hand it in with `frontier submit` when it is final.
Use `spin --help` for compute and model APIs.
