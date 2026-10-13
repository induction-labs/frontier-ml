Quantize Qwen3.5-4B in `/artifacts/qwen3.5-4b/` for text generation only. More than 95% of text model parameters outside the input embedding and output head must be ternary, with roughly one ternary code per original weight (stored parameters may exceed the original count by at most 10%); the embedding, head and activations may remain floating point. You may keep the vision tower and the MTP head in floating point as well, since they will not be used or evaluated.

Write `serve.sh` to serve an OpenAI-compatible chat endpoint offline on one H100, ready within 300 seconds.

Try to preserve as much performance as possible.

Do not train, fine-tune, calibrate or select on the validation or test split of any public dataset or benchmark; training splits and other public corpora are fine.

Submit in `/workspace/output/`. Interface and runtime details: `/task/CONTRACT.md`.
Supporting files: `/task/public/`. Evaluate with `python /task/validate.py --server-url URL`.
Test the finished submission the way it will be run with `frontier selfcheck`.
Hand it in with `frontier submit` when it is final.
Use `spin --help` for compute and model APIs.
