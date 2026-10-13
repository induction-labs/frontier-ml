# Discrete visual latent compression

`POST /encode` accepts `{"image": "<base64 PNG>"}` and returns exactly
`{"codes": [<int>, ...], "codebook_size": <int>}`. Every code is an integer in
`[0, codebook_size)`, `codebook_size` is the same in every response (2 to 16,777,216), and
`len(codes) * ceil(log2(codebook_size))` is at most 16,384 bits (2 KB). A list over the
budget scores 0 for its image and is never decoded.

`codebook_size` describes the serialized integer alphabet. FSQ levels, learned codebook
indices and codes from multiple codebooks may be flattened or packed into this alphabet.
For example, 4,096 scalar codes with 16 levels each store exactly 16,384 bits; so do 1,024
indices into a learned codebook with 65,536 entries. Shared quantization rules and learned
codebooks are part of the submitted model; any per-image information must fit in the codes.

`POST /decode` accepts `{"codes": [...], "codebook_size": <int>}` and returns exactly
`{"text": "..."}`, up to 32,768 characters of plain text or markdown. The decoder runs in a
separate instance and receives only the code list; no per-image state is shared. About 5% of
code lists are decoded twice; if the two normalized texts differ, the lower score counts.
Each endpoint must sustain one image per second with up to 16 concurrent requests.

The encoder is a neural network that quantizes the image into a sequence of codes. FSQ, LFQ,
learned codebooks, residual or product quantization, and similar methods are allowed;
quantization levels may be fixed or learned. The codes are learned
visual codes: the encoder must not transcribe the image into characters, words or
text-tokenizer ids, and the code list must not be a text-token sequence or a compressed
transcript. The decoder is a neural network (any language decoder, fine-tuned or with adapters)
that generates the text from the codes through embeddings, scalar reconstruction or another
representation; no stored transcripts or retrieval may produce it.
Anything of that kind fails the architecture review.

## Submission

Include `serve.sh`, the served weights, training and inference code, and `architecture.json`:

```json
{
  "codebook_size": 4096,
  "base_model": "deepseek-ai/DeepSeek-OCR",
  "base_revision": "9f30c71f441d010e5429c532364a86705536c53a",
  "training_evidence_files": ["train.py", "training_config.json"],
  "inference_evidence_files": ["serve.sh", "server.py", "encoder.py", "decoder.py"]
}
```

`codebook_size` must equal the value returned by `/encode`. `base_model` and
`base_revision` name the starting checkpoint and its immutable revision if you used one
(the DeepSeek-OCR weights in `/artifacts/deepseek-ocr/weights/` are available but not
prescribed); a model trained from scratch declares `"none"` and `"from-scratch"`. The
referenced files should describe initialization, training and the code used to serve the
model. Safetensors is the default supported checkpoint format.

## Examples

Rows in `validation.jsonl` have `id`, `image` (a relative PNG path), `text` (the page text
in reading order), `kind` (`document` or `screenshot`) and `bucket` (text density). Document
pages are 1275×1650; screenshots are 1920×1080.

## Metric

The score averages `1 - levenshtein(words(prediction), words(reference)) /
max(len(words(prediction)), len(words(reference)))` over images. `ocr_text.py`
normalizes Unicode, removes markup and collapses whitespace before splitting words;
case and punctuation count. Malformed responses, code lists over the bit budget and
texts over 32,768 characters score 0 for that image. Character-level edit similarity
and word F1 are diagnostics only.
