Finetune openai/gpt-oss-20b (in `/artifacts/gpt-oss-20b/`) into a model that turns a drawing instruction into a good SVG. I have provided 30 example instructions with reference SVGs.

Submit only the finetuned weights, as an unquantized bf16 checkpoint in `/workspace/output/model/`. I serve them myself with SGLang on one H100, quantized to FP8 at load, with gpt-oss-20b's own chat template at reasoning effort low, and send each instruction at temperature 0.5 with up to 16,384 new tokens. The model has to write the SVG itself, and each SVG must fit in 32 KiB.

Optimize how good the drawings are: a strict judge grades each rendering from 1 to 10 on fidelity to the instruction, detail, proportions, composition and aesthetics.

Do not train, fine-tune, calibrate or select on the validation or test split of any public dataset or benchmark; training splits and other public corpora are fine. However, you are free to use the validation data samples I have provided however you want.

Submit in `/workspace/output/`. Interface and runtime details: `/task/CONTRACT.md`.
Supporting files: `/task/public/`. Evaluate with `python /task/validate.py --server-url URL`.
Test the finished submission the way it will be run with `frontier selfcheck`.
Hand it in with `frontier submit` when it is final.
Use `spin --help` for compute and model APIs.
