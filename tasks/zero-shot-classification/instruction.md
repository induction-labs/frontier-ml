Train a zero-shot text classifier that works with any set of label descriptions. I have provided 500 labeled examples for checking your server locally; do not train, calibrate or select on them.

Don't use any split of BTZSC or its source datasets (listed in `/task/public/DATA_POLICY.md`). Start from any pretrained LLM or encoder, but not one fine-tuned on those datasets, and don't use models from open-source reproductions of TypeSafe's JEV.

Write a script `serve.sh` which serves the classifier. It should have a `POST /classify` endpoint which takes a text and a list of label descriptions and returns a probability for each label. It should run offline on one H100, be ready within 300 seconds, and handle at least 50 documents per second.

Optimize macro-F1 * (1 - ECE), averaged over datasets.

Do not train, fine-tune, calibrate or select on the validation or test split of any public dataset or benchmark; training splits and other public corpora are fine.

Submit in `/workspace/output/`. Interface and runtime details: `/task/CONTRACT.md`.
Supporting files: `/task/public/`. Evaluate with `python /task/validate.py --server-url URL`.
Test the finished submission the way it will be run with `frontier selfcheck`.
Hand it in with `frontier submit` when it is final.
Use `spin --help` for compute and model APIs.
