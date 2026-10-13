Train a model that finds the personal identifiers, private contact and account details, credentials and other secrets in a document so they can be redacted. I have provided 39 labeled example documents for validation.

Write a script `serve.sh` which serves the model. It should have a `POST /redact` endpoint which takes a document and returns the character spans to redact, run offline on one H100, be ready within 300 seconds, and handle at least five documents per second. You can start from any open-weights model that isn't a specialized PII redaction model.

Optimize the mean document R-score: redact everything sensitive without touching unrelated text.

Do not train, fine-tune, calibrate or select on the validation or test split of any public dataset or benchmark; training splits and other public corpora are fine. However, you are free to use the validation data samples I have provided however you want.

Submit in `/workspace/output/`. Interface and runtime details: `/task/CONTRACT.md`.
Supporting files: `/task/public/`. Evaluate with `python /task/validate.py --server-url URL`.
Test the finished submission the way it will be run with `frontier selfcheck`.
Hand it in with `frontier submit` when it is final.
Use `spin --help` for compute and model APIs.
