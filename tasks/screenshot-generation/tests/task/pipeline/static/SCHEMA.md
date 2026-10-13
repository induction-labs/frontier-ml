# Grounded screenshot generation: public validation data

Screens include web pages, desktop applications, code editors, terminals and mobile screens, all 1280x720.

`validation.jsonl` has one sample per line:

| Field | Meaning |
| --- | --- |
| `id` | Sample id. |
| `description` | Text-free description of the screen's visual design and layout (no literal on-screen text). Sent to your server. |
| `screen_type` | `web_page`, `desktop_app`, `ide_terminal`, `mobile` or `other` (metadata; not sent). |
| `texts` | `[{"text", "bbox": [x1, y1, x2, y2]}]`, integer pixels in the 1280x720 canvas, `x2`/`y2` exclusive. Sent to your server. Each entry is one text detection of the pinned OCR (below) on the real screenshot, with the OCR's own string and box (the detector's box is a few pixels larger than the ink). A detection is requested only when a second, independent OCR engine agrees with its string. |
| `size` | Always `1280x720`. Sent to your server. |
| `ignore_regions` | Rectangles of text on the real screen that is not requested (the engines disagreed, vertical or non-Latin text, text inside imagery). They are neutral for the layout score. Not sent to your server. |
| `image` | The real screenshot (reference only; the score never compares against it pixel by pixel). |

The trusted client sends exactly `{"description", "texts", "size"}` to `POST /v1/images/generations`,
one request at a time, and expects `{"data": [{"b64_json": "<PNG>"}]}` with a 1280x720 8-bit RGB PNG.

## Metric (identical in `/task/validate.py` and the final verifier)

Per sample `quality = geomean(T, L, F)` and `score = 2 * quality * R / (quality + R)`, the harmonic mean of
quality and realism (0 when either is 0); the headline is the mean over samples. Text pasted perfectly onto
an unrealistic background saturates T and L but scores less than `2 * R`: quality 1 with R 0.2 scores 0.33,
while quality 0.8 with R 0.8 scores 0.8. A response that is not one 1280x720 8-bit RGB PNG scores 0 for
its sample.

* **T, text accuracy.** For each requested box, the pinned recognizer reads the crop
  `[x1-2, y1+dy, x2+2, y2+dy]` for `dy` in `{0, -s, +s}` with `s = max(1, round(0.2 * box height))`
  and keeps the reading closest to the requested text. Each crop is recognized independently (resized to
  height 48 and padded to a multiple of 32 px), so a reading never depends on other crops. Box score
  = 1 - Levenshtein / max(len) after NFKC and whitespace normalization (case-sensitive). Weighted by
  requested non-space characters. The real screenshot scores T = 1 by construction.
* **L, layout F1.** The pinned detector+recognizer runs on the whole image. Detections without an
  alphanumeric character, within 2 px of the border, or at least half inside an `ignore_regions`
  rectangle are dropped. Precision: each detection's area fraction lying on requested boxes dilated
  by `max(4, 0.5 * box height)` px or on `ignore_regions`, weighted by its recognized characters. Recall:
  each requested box's area fraction covered by detections dilated by 2 px, weighted by its characters.
  L = harmonic mean.
* **F, faithfulness, and R, realism.** The gpt-6-luna judge sees the image with a fixed rubric prompt
  (`FAITHFULNESS_PROMPT` with the description, and `REALISM_PROMPT`, in `screenshot_generation.py`) and
  answers one digit 0-9; the rating is that digit divided by 9. Two judge requests per image, each exactly
  `screenshot_generation.judge_request` (`max_completion_tokens` and `seed` only: gpt-6-luna returns no
  log-probabilities, and `validate.py` sends the same request as the final verifier).

Pinned OCR: PP-OCRv5 server detection and recognition (ONNX, rapidocr 3.9.2, onnxruntime-gpu 1.23.2
CUDA; hashes in `screenshot_generation.py`, shipped next to `validate.py`).
`python /task/validate.py --prepare-models DIR` downloads and verifies it. F and R need an
OpenAI-compatible endpoint serving gpt-6-luna (`--judge-base-url`, `--judge-model`, key in
`$MLE_JUDGE_API_KEY`); each image costs about $0.0003 at list price.

## Diagnostics and final evaluation

`validate.py`'s score is this metric only. It also reports, as a diagnostic that never changes the score,
p95 per-request latency (sequential requests) against the final 30 s requirement. The final evaluation
enforces it, a 120 s ceiling on every single request (the same default as `validate.py`'s
`--request-timeout-seconds`) and a valid image for every request; see `/task/CONTRACT.md`. Latency is
timed by the HTTP client next to the server (in the final evaluation, the trusted client inside the
serving sandbox), so the grading system's own transport overhead never counts.
