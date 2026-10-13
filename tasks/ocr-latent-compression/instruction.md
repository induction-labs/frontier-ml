Train a discrete visual tokenizer for document pages and application screenshots. It should quantize each image into a sequence of codes that stores at most 2 KB (16,384 bits) of information, and a decoder should recover the page text from those codes alone.

Use a neural encoder to produce the codes and a neural decoder for the text. FSQ, learned codebooks and other discrete quantization methods are allowed. Do not transcribe the image in the encoder: the codes are learned visual codes, not text tokens or a compressed transcript.

Write a script `serve.sh` which serves it with `POST /encode` (image to codes) and `POST /decode` (codes to text). It needs to run offline on one H100,  be ready within 300 seconds, and handle at least one image per second on each endpoint.

Optimize word-level transcription accuracy of the decoded text against the page text.

Do not train, fine-tune, calibrate or select on the validation or test split of any public dataset or benchmark; training splits and other public corpora are fine.

Submit in `/workspace/output/`. Interface and runtime details: `/task/CONTRACT.md`.
Supporting files: `/task/public/`. Evaluate with `python /task/validate.py --server-url URL`.
Test the finished submission the way it will be run with `frontier selfcheck`.
Hand it in with `frontier submit` when it is final.
Use `spin --help` for compute and model APIs.
