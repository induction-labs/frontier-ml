Train a vision-language model that takes an image of a scientific figure and writes TikZ/LaTeX code to recreate it. I have provided 50 example figures with reference code from SciTikZ-230K.

You can train on these and the rest of SciTikZ-230K, but don't use SciTikZ-Bench at all. The model itself has to write the TikZ: no embedding the input image, pixel tracing, classical OCR or looking up stored answers. You can start from any pretrained base model, as long as it isn't a specialist TikZ model (e.g., you may not start from JackieLin0123/SciTikZer-8B).

Write a script `serve.sh` which serves the model through an OpenAI-style `POST /v1/chat/completions` endpoint. It takes a figure image and returns a complete LaTeX document in a ```latex block. Run offline on one H100, ready within 300 seconds, and finish all 611 test figures within 2 hours.

Optimize how often the output compiles and how closely the rendered figure and code match the reference.

Do not train, fine-tune, calibrate or select on the validation or test split of any public dataset or benchmark; training splits and other public corpora are fine.

Submit in `/workspace/output/`. Interface and runtime details: `/task/CONTRACT.md`.
Supporting files: `/task/public/`. Evaluate with `python /task/validate.py --server-url URL`.
Test the finished submission the way it will be run with `frontier selfcheck`.
Hand it in with `frontier submit` when it is final.
Use `spin --help` for compute and model APIs.
