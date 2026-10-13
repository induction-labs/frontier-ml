"""Grounded screenshot generation: request template, response checks and the pinned metric.

The same code runs in the public validation script and in the trusted verifier.

Per sample, quality = geomean(T, L, F) and score = the mean of quality and R,
(quality + R) / 2 (operator decision 2026-10-01; a harmonic mean, 2 * quality * R / (quality + R), until then):

* T (text accuracy): pinned OCR recognizes a tight crop of every requested box
  (best of the crop and the crop shifted up/down by 20% of the box height),
  score 1 - normalized edit distance, weighted by requested character count.
  Every crop is recognized independently of the others (batch-independent readings).
* L (layout F1): pinned OCR detects text on the full image. Precision is the
  character-weighted fraction of each detection lying on (dilated) requested boxes;
  recall is the character-weighted fraction of each requested box covered by
  detections. L is their harmonic mean, so missing and extra text both cost.
  Detections touching the image border are ignored, and the sample's
  `ignore_regions` (text on the real screen that is not requested) are neutral.
* F (faithfulness) and R (realism): the gpt-6-luna judge rates the image on a fixed
  0-9 rubric (F also sees the description); the rating is the single digit in its
  reply, divided by 9. The judge is given a generous completion budget (a reasoning
  model's hidden tokens count against it) and bounded retries; when it stays
  unavailable, no score is imputed.

Pasting the exact text onto any background saturates T and L, so accuracy alone must not
carry the score: realism enters with equal weight (quality 1 with R 0.2 scores 0.6, while an
image with quality and realism both at 0.8 scores 0.8), and the reward normalization
(task.yaml score_normalization) puts the organizer compositor, a text paste with no realism,
at 0. The requested texts and boxes are the pinned OCR's own output on the real
screenshot, so a real screenshot scores T = 1 by construction. The headline is the mean over
samples. An image that is not a valid 1280x720 8-bit RGB PNG scores 0 for its sample.
The module also measures p95 per-image latency (a diagnostic here); the trusted verifier
enforces the latency limit.

Heavy dependencies (numpy, Pillow, rapidocr, onnxruntime) are imported lazily so request
construction and parsing need only the standard library.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import math
import re
import struct
import unicodedata
from pathlib import Path

WIDTH, HEIGHT = 1280, 720
SIZE = f"{WIDTH}x{HEIGHT}"
ENDPOINT = "/v1/images/generations"
MAX_TEXTS = 400
CROP_PAD_X = 2           # horizontal context added to every recognition crop, px
REC_BUCKET = 32           # recognition input widths are padded to a multiple of this (batch-independent readings)
REC_BATCH = 16
CROP_SHIFT = 0.2         # recognition is also tried with the crop shifted up/down by this fraction of box height
LAYOUT_TOLERANCE = 0.5   # requested boxes are dilated by max(LAYOUT_MIN_PAD px, 0.5 * box height) for precision
LAYOUT_MIN_PAD = 4
RECALL_DILATION = 2      # detection boxes are dilated by this many px when measuring coverage of requested boxes
BORDER_PX = 2            # detections within this distance of the image border are ignored
IGNORE_COVERAGE = 0.5    # detections at least this much inside a sample's ignore_regions are ignored
LATENCY_P95_SECONDS = 30.0  # final-evaluation requirement (diagnostic here)
METRIC = "mean_accuracy_realism"
JUDGE_MODEL = "gpt-6-luna"
# One chat-completions request per rating; the rating is the single digit 0-9 in the reply text. gpt-6-luna
# returns no token log-probabilities and accepts only its default temperature, so nothing else is asked for;
# `seed` gives best-effort determinism. The agents' Spin judge proxy rejects `logprobs`, `top_logprobs` and
# `reasoning_effort` with HTTP 400 ("unsupported request fields", production rollout of 2026-09-25), so the
# official request carries none of them (JUDGE_PROXY_REJECTED, tested): the model reasons at its default effort
# (about 60 hidden tokens per rating on 2026-09-26 probes), and the public validate.py sends exactly this
# request, so a local run through the proxy exercises the official judge behaviour.
# gpt-6-luna is a reasoning model: any reasoning tokens it spends count against max_completion_tokens, and a
# budget it cannot finish within is rejected with HTTP 400 ("max_tokens or model output limit was reached")
# rather than answered with a truncated reply (incident of 2026-09-25 with a budget of 8). The budget is
# therefore generous (the reply is one digit, so unspent budget costs nothing) and is doubled, up to
# JUDGE_MAX_COMPLETION_TOKENS_CAP, whenever the judge reports it was too small.
JUDGE_MAX_COMPLETION_TOKENS = 4096
JUDGE_MAX_COMPLETION_TOKENS_CAP = 16384
JUDGE_PARAMS = {"max_completion_tokens": JUDGE_MAX_COMPLETION_TOKENS, "seed": 0}
JUDGE_PROXY_REJECTED = ("logprobs", "top_logprobs", "reasoning_effort")  # never sent: the agents' proxy returns 400
JUDGE_IMAGE_DETAIL = "high"
JUDGE_CONCURRENCY = 16
JUDGE_ATTEMPTS = 3   # transient failures (408/409/425/429, 5xx, timeouts, network, non-JSON) back off 1 s, 2 s
JUDGE_BUDGET_ERROR = re.compile(r"max_tokens or model output limit|higher max_(?:completion_)?tokens"
                                r"|output limit was reached", re.IGNORECASE)
JUDGE_TRANSIENT_HTTP = frozenset({408, 409, 425, 429})
RATING_SCALE = 9
REALISM_PROMPT = (
    "You are an expert at telling genuine computer screenshots from synthetic, generated or edited images.\n"
    "Rate how realistic this image is as a genuine, unedited screenshot of a real computer screen (a web page, "
    "desktop application, code editor, terminal or mobile screen).\n"
    "Judge only the image. Real screenshots have crisp, legible, correctly spelled UI text; coherent, consistent "
    "layouts and widgets; plausible icons and imagery. Penalize garbled or invented glyphs, warped or melted shapes, "
    "blur or smearing, inconsistent UI elements, diffusion artifacts, and flat mock-ups with text on an otherwise "
    "empty canvas.\n"
    "Scale: 0 = not a screenshot at all; 2 = obviously fake or a bare mock-up; 4 = clearly synthetic with many "
    "artifacts; 6 = plausible at a glance but with noticeable flaws; 8 = convincing with minor flaws; "
    "9 = indistinguishable from a real screenshot.\n"
    "Answer with a single digit from 0 to 9 and nothing else.")
FAITHFULNESS_PROMPT = (
    "Here is a description of a computer screenshot:\n\n{description}\n\n"
    "Rate how faithfully the image matches this description's visual design and layout: the kind of screen, "
    "window and browser chrome, color scheme and light or dark theme, the arrangement of panels, sidebars, toolbars, "
    "tables and editors, imagery and icons, and the style of controls. Ignore the exact wording of any text.\n"
    "Scale: 0 = unrelated to the description; 2 = same broad kind of screen but most described elements are missing "
    "or wrong; 4 = several described elements present, several missing or wrong; 6 = most described elements present "
    "with some clear mismatches; 8 = nearly everything matches; 9 = every described element is present and matches.\n"
    "Answer with a single digit from 0 to 9 and nothing else.")

# Pinned scorer models. Paths are relative to the scorer model directory.
PINS = {
    "ocr": {
        "package": "rapidocr==3.9.2 onnxruntime-gpu==1.23.2 (CUDA 12 execution provider, cudnn_conv_algo_search=DEFAULT)",
        "files": {
            "ocr/ch_PP-OCRv5_det_server.onnx": "0f8846b1d4bba223a2a2f9d9b44022fbc22cc019051a602b41a7fda9667e4cad",
            "ocr/ch_PP-OCRv5_rec_server.onnx": "e09385400eaaaef34ceff54aeb7c4f0f1fe014c27fa8b9905d4709b65746562a",
        },
        "urls": {
            "ocr/ch_PP-OCRv5_det_server.onnx": "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv5/det/ch_PP-OCRv5_det_server.onnx",
            "ocr/ch_PP-OCRv5_rec_server.onnx": "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_server.onnx",
        },
    },
}


class ScorerUnavailable(RuntimeError):
    """A pinned scorer model, dependency or the judge is missing or malformed; no score is imputed.

    This is the scorer's own failure, never the submission's: the verifier reports it as an evaluation error.
    """


class RatingTruncated(ValueError):
    """The judge ran out of completion budget before its digit (finish_reason "length")."""


# ---------------------------------------------------------------- requests and responses

def build_request(sample, data_root=None):
    texts = [{"text": item["text"], "bbox": [int(v) for v in item["bbox"]]} for item in sample["texts"]]
    return ENDPOINT, {"description": sample["description"], "texts": texts, "size": SIZE}


def _png_header(data):
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError("Response image must be a PNG")
    width, height, depth, color = struct.unpack(">IIBB", data[16:26])
    return width, height, depth, color


def decode_response(response):
    """Validate {"data":[{"b64_json": PNG}]} and return the PNG bytes (exactly 1280x720, 8-bit RGB)."""
    if not isinstance(response, dict) or not isinstance(response.get("data"), list) or len(response["data"]) != 1:
        raise ValueError('Response must be {"data":[{"b64_json": "..."}]} with exactly one image')
    item = response["data"][0]
    if not isinstance(item, dict) or not isinstance(item.get("b64_json"), str) or not item["b64_json"]:
        raise ValueError("Response image must be a nonempty b64_json string")
    try:
        data = base64.b64decode(item["b64_json"], validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("b64_json is not valid base64") from error
    width, height, depth, color = _png_header(data)
    if (width, height) != (WIDTH, HEIGHT):
        raise ValueError(f"Image must be exactly {WIDTH}x{HEIGHT}; got {width}x{height}")
    if depth != 8 or color != 2:
        raise ValueError("Image must be an 8-bit RGB PNG (color type 2, no alpha or palette)")
    return data


def load_png(data):
    """Decode a validated PNG to an RGB PIL image (dimension checks happen before decompression)."""
    from PIL import Image
    image = Image.open(io.BytesIO(data))
    if image.size != (WIDTH, HEIGHT) or image.mode != "RGB":
        raise ValueError("Image must be exactly 1280x720 RGB")
    image.load()
    return image


def validate_sample(sample):
    if not isinstance(sample.get("description"), str) or not sample["description"].strip():
        raise ValueError("Sample needs a description")
    texts = sample.get("texts")
    if not isinstance(texts, list) or not texts or len(texts) > MAX_TEXTS:
        raise ValueError("Sample needs 1..400 texts")
    for item in texts:
        box = item.get("bbox")
        if not isinstance(item.get("text"), str) or not item["text"].strip() or not isinstance(box, list) or len(box) != 4:
            raise ValueError("Each text needs text and bbox [x1,y1,x2,y2]")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= WIDTH and 0 <= y1 < y2 <= HEIGHT):
            raise ValueError("bbox outside the canvas")
    regions = sample.get("ignore_regions", [])
    if not isinstance(regions, list) or any(not isinstance(r, list) or len(r) != 4 or not all(isinstance(v, int) for v in r)
                                            or not (0 <= r[0] < r[2] <= WIDTH and 0 <= r[1] < r[3] <= HEIGHT) for r in regions):
        raise ValueError("ignore_regions must be a list of in-canvas integer [x1,y1,x2,y2] boxes")
    for key in ("id",):
        if key not in sample:
            raise ValueError("Sample needs an id")


# ---------------------------------------------------------------- text metrics

def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def levenshtein(a, b):
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def similarity(prediction, reference):
    prediction, reference = normalize(prediction), normalize(reference)
    return 1.0 - levenshtein(prediction, reference) / max(len(prediction), len(reference), 1)


def weight(text):
    return sum(1 for c in normalize(text) if not c.isspace())


def text_accuracy(recognized, texts):
    """T: character-weighted mean of 1 - NED between OCR of each requested crop and its text."""
    if len(recognized) != len(texts):
        raise ValueError("One recognition per requested box is required")
    total = sum(weight(t["text"]) for t in texts)
    scores = [similarity(r, t["text"]) for r, t in zip(recognized, texts)]
    value = sum(weight(t["text"]) * s for t, s in zip(texts, scores)) / max(total, 1)
    return value, scores


def crop_box(bbox, width=WIDTH, height=HEIGHT, dy=0):
    x1, y1, x2, y2 = (int(v) for v in bbox)
    return max(0, x1 - CROP_PAD_X), max(0, y1 + dy), min(width, x2 + CROP_PAD_X), min(height, y2 + dy)


def crop_offsets(bbox):
    """Vertical crop offsets tried for a box: 0 and +-CROP_SHIFT of its height (the best reading counts)."""
    shift = max(1, int(round(CROP_SHIFT * (int(bbox[3]) - int(bbox[1])))))
    return (0, -shift, shift)


def layout_f1(detections, texts, width=WIDTH, height=HEIGHT, ignore_regions=None):
    """L: character-weighted F1 between OCR detections and requested boxes (area coverage).

    ignore_regions: rectangles of the real screen whose text is not requested (text inside imagery,
    or text whose ground truth is uncertain). They are neutral: detections lying mostly inside them
    are dropped, and the part of any other detection that falls inside them does not count against
    precision.
    """
    import numpy as np
    ignored = np.zeros((height, width), bool)
    for x1, y1, x2, y2 in ignore_regions or ():
        ignored[max(0, int(y1)):min(height, int(y2)), max(0, int(x1)):min(width, int(x2))] = True
    kept, skipped = [], 0
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        if weight(det["text"]) == 0 or not any(c.isalnum() for c in det["text"]):
            continue
        if x1 <= BORDER_PX or y1 <= BORDER_PX or x2 >= width - BORDER_PX or y2 >= height - BORDER_PX:
            continue
        region = ignored[max(0, int(round(y1))):min(height, int(round(y2))), max(0, int(round(x1))):min(width, int(round(x2)))]
        if region.size and region.mean() >= IGNORE_COVERAGE:
            skipped += 1
            continue
        kept.append(det)
    requested = np.zeros((height, width), bool)
    for item in texts:
        x1, y1, x2, y2 = (int(v) for v in item["bbox"])
        pad = max(LAYOUT_MIN_PAD, int(round(LAYOUT_TOLERANCE * (y2 - y1))))
        requested[max(0, y1 - pad):min(height, y2 + pad), max(0, x1 - pad):min(width, x2 + pad)] = True
    neutral = requested | ignored  # precision: detection area on requested boxes or neutral regions
    detected = np.zeros((height, width), bool)
    for det in kept:
        x1, y1, x2, y2 = (int(round(v)) for v in det["bbox"])
        d = RECALL_DILATION
        detected[max(0, y1 - d):min(height, y2 + d), max(0, x1 - d):min(width, x2 + d)] = True
    precision_num = precision_den = 0.0
    for det in kept:
        x1, y1, x2, y2 = (int(round(v)) for v in det["bbox"])
        region = neutral[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
        if region.size == 0:
            continue
        w = weight(det["text"])
        precision_num += w * float(region.mean())
        precision_den += w
    recall_num = recall_den = 0.0
    for item in texts:
        x1, y1, x2, y2 = (int(v) for v in item["bbox"])
        region = detected[y1:y2, x1:x2]
        w = weight(item["text"])
        recall_num += w * (float(region.mean()) if region.size else 0.0)
        recall_den += w
    precision = precision_num / precision_den if precision_den else 0.0
    recall = recall_num / recall_den if recall_den else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return f1, {"precision": precision, "recall": recall, "detections": len(kept), "ignored_detections": skipped}


def geomean(values):
    if any(v <= 0 for v in values):
        return 0.0
    return math.exp(sum(math.log(v) for v in values) / len(values))


def harmonic_mean(a, b):
    """2ab / (a + b); 0 when both are 0. Either side at 0 zeroes the result."""
    return 2 * a * b / (a + b) if a + b > 0 else 0.0


def sample_score(parts):
    """The per-sample metric from its scored parts ({component: value}, a subset of COMPONENTS).

    quality = geomean of the accuracy parts present (T, L, F); score = (quality + R) / 2. Scoring a
    subset (validate.py --components) drops the missing side: without R the score is quality, and with only R
    it is R. Returns (score, quality or None)."""
    accuracy = [parts[c] for c in ACCURACY_COMPONENTS if c in parts]
    if not accuracy and "R" not in parts:
        raise ValueError("At least one metric part is required")
    quality = geomean(accuracy) if accuracy else None
    if quality is None:
        return parts["R"], None
    if "R" not in parts:
        return quality, quality
    return (quality + parts["R"]) / 2, quality


def percentile(values, q):
    import math
    ordered = sorted(values)
    if not ordered:
        raise ValueError("No values")
    rank = max(0, math.ceil(q / 100 * len(ordered)) - 1)
    return ordered[rank]


def latency_summary(latencies):
    """p95 of per-request wall time against the 30 s requirement."""
    p95 = percentile(latencies, 95)
    return {"requests": len(latencies), "mean_seconds": sum(latencies) / len(latencies), "p95_seconds": p95,
            "max_seconds": max(latencies), "required_p95_seconds": LATENCY_P95_SECONDS,
            "meets_requirement": p95 <= LATENCY_P95_SECONDS}


# ---------------------------------------------------------------- pinned models

def verify_files(model_dir, component, pins=None):
    root = Path(model_dir)
    missing, wrong = [], []
    for relative, digest in (pins or PINS)[component]["files"].items():
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 22), b""):
                h.update(block)
        if h.hexdigest() != digest:
            wrong.append(relative)
    if missing or wrong:
        raise ScorerUnavailable(f"Pinned {component} files missing {missing} or with wrong hashes {wrong} under {root}")


class PinnedOCR:
    """PP-OCRv5 server detection + recognition through rapidocr 3.9.2 on the ONNX Runtime CPU provider."""

    def __init__(self, model_dir, threads=8, device="cuda", verify=True, gpu_mem_limit=None):
        if verify:
            verify_files(model_dir, "ocr")
        try:
            import onnxruntime
            from rapidocr import RapidOCR
            from rapidocr.ch_ppocr_rec import TextRecInput
        except ImportError as error:
            raise ScorerUnavailable("Install rapidocr==3.9.2 and onnxruntime-gpu (or onnxruntime for CPU) to score") from error
        onnxruntime.set_default_logger_severity(3)  # per-shape cuDNN fallback warnings are noise
        if device == "cuda":
            if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
                raise ScorerUnavailable("onnxruntime-gpu with CUDAExecutionProvider is required for device=cuda")
            if hasattr(onnxruntime, "preload_dlls"):
                onnxruntime.preload_dlls()
        elif device != "cpu":
            raise ValueError("OCR device must be cuda or cpu")
        root = Path(model_dir) / "ocr"
        self._rec_input = TextRecInput
        extra = {"EngineConfig.onnxruntime.cuda_ep_cfg.gpu_mem_limit": int(gpu_mem_limit)} if gpu_mem_limit else {}
        self.engine = RapidOCR(params={**extra,
            "EngineConfig.onnxruntime.use_cuda": device == "cuda",
            "EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search": "DEFAULT",
            "EngineConfig.onnxruntime.cuda_ep_cfg.arena_extend_strategy": "kSameAsRequested",
            "Global.log_level": "error", "Global.use_cls": False, "Global.text_score": 0.5,
            "Global.use_vertical_padding": False,
            "Det.model_path": str(root / "ch_PP-OCRv5_det_server.onnx"),
            "Rec.model_path": str(root / "ch_PP-OCRv5_rec_server.onnx"),
            "Det.limit_side_len": 1920, "Det.limit_type": "max", "Det.thresh": 0.3, "Det.box_thresh": 0.5,
            "Det.unclip_ratio": 1.6, "Rec.rec_batch_num": 8,
            "EngineConfig.onnxruntime.intra_op_num_threads": int(threads),
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        })

    @staticmethod
    def _bgr(image):
        import numpy as np
        return np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])

    def recognize(self, image, texts):
        """Best reading per requested box over the crop offsets (closest to the requested text)."""
        import numpy as np
        array = self._bgr(image)
        crops, index = [], []
        for i, item in enumerate(texts):
            for dy in crop_offsets(item["bbox"]):
                x1, y1, x2, y2 = crop_box(item["bbox"], array.shape[1], array.shape[0], dy)
                if x2 - x1 >= 2 and y2 - y1 >= 2:
                    crops.append(np.ascontiguousarray(array[y1:y2, x1:x2]))
                    index.append(i)
        readings = [[] for _ in texts]
        for i, text in zip(index, self.recognize_crops(crops)):
            readings[i].append(text or "")
        return [max(r, key=lambda text: similarity(text, item["text"])) if r else ""
                for r, item in zip(readings, texts)]

    def recognize_crops(self, crops):
        """Batch-independent recognition: every crop is resized to height 48 and padded to its own width
        bucket (a multiple of REC_BUCKET px), and only crops with the same bucket share a batch, so a
        reading never depends on which other crops were recognized alongside it."""
        import cv2
        import numpy as np
        rec = self.engine.text_rec
        height = 48
        prepared = []
        for i, crop in enumerate(crops):
            h, w = crop.shape[:2]
            width = max(1, int(math.ceil(height * w / float(h))))
            bucket = int(math.ceil(width / REC_BUCKET)) * REC_BUCKET
            resized = cv2.resize(crop, (width, height)).astype("float32").transpose((2, 0, 1)) / 255
            tensor = np.zeros((3, height, bucket), dtype=np.float32)
            tensor[:, :, :width] = (resized - 0.5) / 0.5
            prepared.append((bucket, i, tensor, width / float(height)))
        texts = [""] * len(crops)
        prepared.sort(key=lambda item: (item[0], item[1]))
        start = 0
        while start < len(prepared):
            bucket = prepared[start][0]
            end = start
            while end < len(prepared) and prepared[end][0] == bucket and end - start < REC_BATCH:
                end += 1
            group = prepared[start:end]
            preds = rec.session(np.stack([g[2] for g in group]))
            lines, _ = rec.postprocess_op(preds, False, wh_ratio_list=[g[3] for g in group], max_wh_ratio=bucket / float(height))
            for (_, i, _, _), (text, _score) in zip(group, lines):
                texts[i] = text
            start = end
        return texts

    def detect(self, image):
        output = self.engine(self._bgr(image), use_det=True, use_cls=False, use_rec=True)
        detections = []
        if getattr(output, "boxes", None) is None or output.txts is None:
            return detections
        for polygon, text, score in zip(output.boxes, output.txts, output.scores):
            xs = [float(p[0]) for p in polygon]
            ys = [float(p[1]) for p in polygon]
            detections.append({"bbox": [min(xs), min(ys), max(xs), max(ys)], "text": text, "score": float(score)})
        return detections


def png_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def judge_request(model, image_url, prompt):
    """One chat-completions payload: the image, then the fixed rubric prompt."""
    return {"model": model, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image_url, "detail": JUDGE_IMAGE_DETAIL}},
        {"type": "text", "text": prompt}]}], **JUDGE_PARAMS}


def _digit(text):
    word = text.strip()
    word = word[:-1] if word.endswith(".") else word
    return int(word) if len(word) == 1 and word.isdigit() else None


def parse_rating(response):
    """Strict: the reply must be exactly one digit 0-9 (surrounding space and one final period aside); that digit
    over RATING_SCALE is the rating. gpt-6-luna returns no log-probabilities, so nothing else is read.
    Returns {"rating" in [0,1], "reply", "digit"}."""
    try:
        choice = response["choices"][0]
        reply = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("judge response has no message content") from error
    digit = _digit(reply) if isinstance(reply, str) else None
    if digit is None:
        if isinstance(choice, dict) and choice.get("finish_reason") == "length":
            raise RatingTruncated("judge reply was cut off by the completion budget")
        raise ValueError(f"judge reply is not a single digit: {str(reply)[:80]!r}")
    return {"rating": digit / RATING_SCALE, "reply": reply, "digit": digit}


def judge_error_kind(error):
    """Classify a failed judge call and read its body once.

    Returns ("budget", body) for an HTTP 400 saying the completion budget was too small (retried with a larger
    max_completion_tokens), ("transient", body) for 408/409/425/429, 5xx, timeouts, network and non-JSON
    failures (bounded retries with backoff), or ("permanent", body) for any other 4xx (no retry)."""
    body = ""
    if hasattr(error, "read"):
        try:
            body = error.read()[:600].decode(errors="replace")
        except Exception:  # noqa: BLE001 - an unreadable body is simply absent from the message
            body = ""
    code = getattr(error, "code", None)
    if not isinstance(code, int):
        return "transient", body
    if code == 400 and JUDGE_BUDGET_ERROR.search(body):
        return "budget", body
    if code in JUDGE_TRANSIENT_HTTP or code >= 500:
        return "transient", body
    return "permanent", body


def _grow_budget(payload):
    """Double max_completion_tokens up to JUDGE_MAX_COMPLETION_TOKENS_CAP; False once the cap is reached."""
    current = int(payload.get("max_completion_tokens") or JUDGE_MAX_COMPLETION_TOKENS)
    if current >= JUDGE_MAX_COMPLETION_TOKENS_CAP:
        return False
    payload["max_completion_tokens"] = min(JUDGE_MAX_COMPLETION_TOKENS_CAP, 2 * current)
    return True


def judge_ratings(judge, jobs, model=JUDGE_MODEL, concurrency=JUDGE_CONCURRENCY, attempts=JUDGE_ATTEMPTS,
                  sleep=None, on_complete=None):
    """jobs: list of (PIL image, description). Two judge calls per image (realism, faithfulness).
    judge(**payload) returns an OpenAI chat-completions body. A too-small completion budget (HTTP 400 or a
    reply cut off by finish_reason "length") is retried at once with a doubled budget up to the cap;
    transient failures are retried `attempts` times with backoff; a malformed reply is never retried (the
    answer must not be re-rolled), and a judge that stays unavailable raises ScorerUnavailable (no score
    imputed). Returns per job {"R": {...}, "F": {...}} as parse_rating results."""
    import time
    from concurrent.futures import ThreadPoolExecutor
    sleep = sleep or time.sleep
    calls = []
    for j, (image, description) in enumerate(jobs):
        url = png_data_url(image)
        calls.append((j, "R", judge_request(model, url, REALISM_PROMPT)))
        calls.append((j, "F", judge_request(model, url, FAITHFULNESS_PROMPT.format(description=description))))

    def ask(payload):
        payload = dict(payload)
        failures = 0
        while True:
            try:
                response = judge(**payload)
            except Exception as error:  # HTTP, network or JSON failure
                kind, body = judge_error_kind(error)
                if kind == "budget" and _grow_budget(payload):
                    continue
                failures += 1
                if kind != "transient" or failures >= attempts:
                    raise ScorerUnavailable(f"Judge {model} unavailable; no score imputed: "
                                            f"{type(error).__name__}: {error} {body}".rstrip()) from error
                sleep(2 ** (failures - 1))
                continue
            try:
                return parse_rating(response)
            except RatingTruncated as error:
                if _grow_budget(payload):
                    continue
                raise ScorerUnavailable(f"Judge {model} unavailable; no score imputed: {error} even at "
                                        f"max_completion_tokens={payload['max_completion_tokens']}") from error
            except ValueError as error:
                raise ScorerUnavailable(f"Malformed judge response; no score imputed: {error}") from error
    with ThreadPoolExecutor(max(1, int(concurrency))) as pool:
        answers = []
        for answer in pool.map(ask, [payload for _, _, payload in calls]):
            answers.append(answer)
            if on_complete is not None:
                on_complete(len(answers))
    results = [{} for _ in jobs]
    for (owner, part, _), answer in zip(calls, answers):
        results[owner][part] = answer
    return results


# ---------------------------------------------------------------- scoring

COMPONENTS = ("T", "L", "F", "R")
ACCURACY_COMPONENTS = ("T", "L", "F")  # geomean -> quality; the score is its mean with R


def score_images(samples, pngs, config):
    """Score validated PNGs against their samples with the pinned OCR and the judge.

    Per record: quality = geomean(T, L, F), score = (quality + R) / 2 (see sample_score); the result's
    score is the mean over records.
    pngs: validated PNG bytes, or None for a sample whose response was invalid (that sample scores 0).
    config: scorer_models_dir (required for T and L), components (default T, L, F, R), ocr_threads, and for
    F and R the trusted `judge` callable (judge(**payload) -> chat-completions body), judge_model and
    judge_concurrency.
    """
    from frontier_ml.validation.progress import progress
    if len(samples) != len(pngs) or not samples:
        raise ValueError("Exactly one image per sample is required")
    components = set(config.get("components") or COMPONENTS)
    if not components <= set(COMPONENTS):
        raise ValueError(f"components must be a subset of {COMPONENTS}")
    model_dir = config.get("scorer_models_dir")
    if components & {"T", "L"} and not model_dir:
        raise ScorerUnavailable("scorer_models_dir is required (see --prepare-models)")
    judge, judge_model = config.get("judge"), config.get("judge_model") or JUDGE_MODEL
    records = [{"id": s["id"]} for s in samples]
    images = []
    for data, record in zip(pngs, records):
        image = None
        if data is None:
            record["invalid"] = "invalid response"
        else:
            try:
                image = load_png(data)
            except (ValueError, OSError) as error:  # corrupt PNG body: invalid for this sample
                record["invalid"] = f"{type(error).__name__}: {error}"
        images.append(image)
        if image is None:
            record.update({c: 0.0 for c in components})
    live = [(s, im, r) for s, im, r in zip(samples, images, records) if im is not None]
    if components & {"F", "R"} and live and not callable(judge):  # before any scorer loads
        raise ScorerUnavailable(f"Faithfulness and realism need the {JUDGE_MODEL} judge; no score imputed")
    if components & {"T", "L"} and live:
        progress(config, "ocr_loading")
        ocr = PinnedOCR(model_dir, threads=config.get("ocr_threads", 8))
        progress(config, "ocr", completed=0, total=len(live))
        for index, (sample, image, record) in enumerate(live, 1):
            if "T" in components:
                recognized = ocr.recognize(image, sample["texts"])
                record["T"], per_box = text_accuracy(recognized, sample["texts"])
                record["worst_boxes"] = sorted(({"text": t["text"], "ocr": r, "similarity": round(s, 3)}
                                                for t, r, s in zip(sample["texts"], recognized, per_box)),
                                               key=lambda x: x["similarity"])[:5]
            if "L" in components:
                record["L"], record["layout"] = layout_f1(ocr.detect(image), sample["texts"],
                                                         ignore_regions=sample.get("ignore_regions"))
            progress(config, "ocr", completed=index, total=len(live))
        del ocr
        import gc
        gc.collect()
    if components & {"F", "R"} and live:
        progress(config, "judge", completed=0, total=2 * len(live))
        ratings = judge_ratings(judge, [(image, s["description"]) for s, image, _ in live],
                                model=judge_model, concurrency=config.get("judge_concurrency", JUDGE_CONCURRENCY),
                                on_complete=lambda n: progress(config, "judge", completed=n, total=2 * len(live)))
        for (_, _, record), rating in zip(live, ratings):
            for part in ("F", "R"):
                if part in components:
                    record[part] = rating[part]["rating"]
                    record[f"{part}_reply"] = rating[part]["reply"]
    for record in records:
        record["score"], quality = sample_score({c: record[c] for c in COMPONENTS if c in components})
        if quality is not None:
            record["quality"] = quality
    means = {c: sum(r[c] for r in records) / len(records) for c in COMPONENTS if c in components}
    score = sum(r["score"] for r in records) / len(records)
    return {"score": score, "means": means, "records": records, "images": images,
            "invalid_samples": sum(1 for r in records if "invalid" in r),
            "components": [c for c in COMPONENTS if c in components],
            "complete_metric": components == set(COMPONENTS),
            **({"judge_model": judge_model} if components & {"F", "R"} else {})}


def prepare_models(model_dir, components=("ocr",), pins=None):
    """Download the pinned OCR files (ModelScope) and verify their hashes."""
    import urllib.request
    root = Path(model_dir)
    pins = pins or PINS
    for component in components:
        pin = pins[component]
        if component == "ocr":
            for relative, url in pin["urls"].items():
                target = root / relative
                if not target.is_file():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with urllib.request.urlopen(url, timeout=600) as response, open(str(target) + ".part", "wb") as out:
                        while block := response.read(1 << 22):
                            out.write(block)
                    Path(str(target) + ".part").rename(target)
        else:
            raise ValueError(f"Unknown scorer component {component}")
        verify_files(root, component, pins)
    return str(root)
