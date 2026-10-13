"""End-of-turn probability-stream scoring and conversation HTTP adapter.

The gold event semantics (annotated ends of turn, believable mid-turn pauses, disputed regions, the
0.25 s early tolerance and the cap at the speaker's next end of turn) follow the MIT-licensed reference
implementation at revision 76ccd045f121ccfa921abac2ad3107027e736911. Its one-to-one matcher over
committed detection times is replaced here by window maxima over a causal per-frame probability
stream and a threshold sweep, so the operating point is the scorer's, not the server's; the late window
is 1 s instead of 3 s. NumPy's default linear percentile is implemented with the standard library.

MIT License

Copyright (c) 2026 Freeman Jiang, Ramon Sanabria, Soham Deshmukh, Bandhav Veluri, Simon Michael Vuch Williams, Elliott K. Suen, Garreth Lee, Kevin Yoonho Choi, Takuya Umeki, Riku Kubo, Sathvik Udupa, Chien-yu Huang, Shih-Yun Shan Kuan, Zhuoyan Tao, Satyapriya Krishna, Sefik Emre Eskimez, Yu Tsao, Hung-yi Lee, Shinji Watanabe

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
from __future__ import annotations
import base64
import bisect
from dataclasses import dataclass
import json
import math
from pathlib import Path

METRIC = "eot_recall_at_5pct_fpr"
TAU_PRE_S = 0.25  # early tolerance before an annotated end of turn or pause: the gold boundary is annotation-exact
TAU_MAX_S = 1.0  # late window after an annotated end of turn
MIN_HOP_S, MAX_HOP_S = 0.02, 0.20  # accepted frame hops, inclusive
FPR_TARGET = 0.05  # the score is recall at this false-positive rate
FPR_REPORTED = (0.05, 0.10, 0.15)  # operating points reported alongside
EPSILON = 1e-9  # slack for hop-multiple arithmetic on inclusive window boundaries


def _percentile(values, p):
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100
    lower = math.floor(position)
    fraction = position - lower
    if lower + 1 == len(ordered):
        return float(ordered[lower])
    # NumPy switches the algebra at .5 to reduce cancellation.
    left, right = ordered[lower], ordered[lower + 1]
    return float(left + (right - left) * fraction if fraction < .5 else right - (right - left) * (1 - fraction))


@dataclass(frozen=True)
class Stream:
    """One speaker's causal probability stream: frame i covers (i*hop, (i+1)*hop] and is reported at (i+1)*hop."""
    hop_s: float
    probabilities: list[float]

    def frame_time(self, index: int) -> float:
        return (index + 1) * self.hop_s

    def frames_in(self, start_s: float, end_s: float) -> range:
        """Indices of the frames reported inside [start_s, end_s] (both inclusive), in time order."""
        first = max(0, math.ceil(start_s / self.hop_s - 1 - EPSILON))
        last = min(len(self.probabilities) - 1, math.floor(end_s / self.hop_s - 1 + EPSILON))
        return range(first, last + 1)


@dataclass(frozen=True)
class Window:
    """The scoreable frames (report time, probability) inside one gold window; anchor_s is the annotated time."""
    anchor_s: float
    frames: tuple[tuple[float, float], ...]

    @property
    def statistic(self) -> float:
        """The window's statistic: its highest scoreable probability, 0 when it holds no scoreable frame."""
        return max((probability for _, probability in self.frames), default=0.0)

    def first_crossing(self, threshold: float) -> float | None:
        return next((time_s for time_s, probability in self.frames if probability >= threshold), None)


def speaker_windows(gold, speaker, stream, *, tau_pre_s=TAU_PRE_S, tau_max_s=TAU_MAX_S):
    """Positive and pause windows of one speaker over that speaker's stream.

    A positive window is [t - tau_pre_s, min(t + tau_max_s, the same speaker's next end of turn)]; a pause window
    is [start - tau_pre_s, end]. Frames reported inside a disputed (excluded) region of this speaker are scoreable
    in neither kind of window, and frames inside one of this speaker's positive windows are not scoreable in a
    pause window: a correct detection is never also a false positive. The other speaker never shrinks a window.
    """
    frames = len(stream.probabilities)
    excluded = [False] * frames
    for interval in gold["eot_excluded"]:
        if interval["speaker"] == speaker:
            for index in stream.frames_in(interval["start"], interval["end"]):
                excluded[index] = True
    anchors = sorted(event["time_s"] for event in gold["eot_positive_events"] if event["speaker"] == speaker)
    in_positive = [False] * frames
    positives = []
    for position, anchor in enumerate(anchors):
        following = anchors[position + 1] if position + 1 < len(anchors) else math.inf
        indices = stream.frames_in(anchor - tau_pre_s, min(anchor + tau_max_s, following))
        for index in indices:
            in_positive[index] = True
        positives.append(Window(anchor, tuple((stream.frame_time(i), stream.probabilities[i])
                                              for i in indices if not excluded[i])))
    pauses = []
    for span in gold["eot_negative_spans"]:
        if span["speaker"] == speaker:
            indices = stream.frames_in(span["start"] - tau_pre_s, span["end"])
            pauses.append(Window(span["start"], tuple((stream.frame_time(i), stream.probabilities[i])
                                                      for i in indices if not excluded[i] and not in_positive[i])))
    return positives, pauses


def sweep(positives, pauses):
    """(threshold, recall, fpr) at every observed window statistic, ascending; a rate is None without its windows."""
    positive_statistics = sorted(window.statistic for window in positives)
    pause_statistics = sorted(window.statistic for window in pauses)
    curve = []
    for threshold in sorted(set(positive_statistics) | set(pause_statistics)):
        recall = (len(positive_statistics) - bisect.bisect_left(positive_statistics, threshold)) / len(positive_statistics) \
            if positive_statistics else None
        fpr = (len(pause_statistics) - bisect.bisect_left(pause_statistics, threshold)) / len(pause_statistics) \
            if pause_statistics else None
        curve.append((threshold, recall, fpr))
    return curve


def operating_point(curve, fpr_limit):
    """The threshold with the highest recall whose FPR is within the limit (ties: lower FPR, then higher threshold).

    Without one (no threshold reaches the limit, or a rate is undefined) nothing fires: threshold None, recall 0.
    """
    qualifying = [(recall, -fpr, threshold) for threshold, recall, fpr in curve
                  if recall is not None and fpr is not None and fpr <= fpr_limit]
    if not qualifying:
        return {"threshold": None, "recall": 0.0, "fpr": 0.0}
    recall, negated_fpr, threshold = max(qualifying)
    return {"threshold": threshold, "recall": recall, "fpr": -negated_fpr}


def load_rows(path, *, require_gold=True):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Expected nonempty conversations with unique IDs")
    for row in rows:
        if not isinstance(row.get("id"), str) or row.get("conversation_id") != row["id"]:
            raise ValueError("Conversation IDs must be matching strings")
        duration = row.get("duration_s")
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Conversation duration must be positive and finite")
        if not isinstance(row.get("audio"), dict) or set(row["audio"]) != {"speaker_1", "speaker_2"}:
            raise ValueError("Both speaker channels are required")
        gold = row.get("gold")
        if gold is None:
            if require_gold:
                raise ValueError("Missing EOT annotations; audio-only rows cannot be scored")
            continue
        expected = {"eot_positive_events", "eot_negative_spans", "eot_excluded"}
        if not isinstance(gold, dict) or set(gold) != expected:
            raise ValueError("Invalid EOT gold schema")
        for key in expected:
            if not isinstance(gold[key], list):
                raise ValueError("Gold event sets must be lists")
            for event in gold[key]:
                fields = {"speaker", "time_s"} if key == "eot_positive_events" else {"speaker", "start", "end"}
                if not isinstance(event, dict) or set(event) != fields or event["speaker"] not in (1, 2):
                    raise ValueError("Invalid gold event")
                values = [event[k] for k in fields - {"speaker"}]
                if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
                    raise ValueError("Gold timestamps must be finite nonnegative seconds")
                if "start" in event and event["start"] > event["end"]:
                    raise ValueError("Gold interval is reversed")
    return rows


def audio_path(reference, root):
    root = Path(root).resolve()
    path = (root / reference).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Audio reference escapes the dataset directory")
    return path


def build_request(row, root):
    channels = {speaker: base64.b64encode(audio_path(reference, root).read_bytes()).decode()
                for speaker, reference in row["audio"].items()}
    return "/predict", {"audio_format": "flac", "speaker_1_audio": channels["speaker_1"],
                        "speaker_2_audio": channels["speaker_2"]}


def frame_count(duration_s, hop_s):
    """Frames of hop_s covering duration_s of audio (the last frame may run past the end by less than a hop)."""
    return math.ceil(duration_s / hop_s - EPSILON)


def silent_stream(duration_s, hop_s=MAX_HOP_S):
    """All-zero probabilities covering the audio: nothing fires, so every end of turn is missed and no pause flagged."""
    return {"hop_seconds": hop_s, "eot_probability": [0.0] * frame_count(duration_s, hop_s)}


def parse_response(row, response):
    if not isinstance(response, dict) or set(response) != {"speaker_1", "speaker_2"}:
        raise ValueError("Response must contain exactly speaker_1 and speaker_2")
    prediction = {"id": row["id"]}
    for speaker in ("speaker_1", "speaker_2"):
        block = response[speaker]
        if not isinstance(block, dict) or set(block) != {"hop_seconds", "eot_probability"}:
            raise ValueError("Each speaker must contain exactly hop_seconds and eot_probability")
        hop = block["hop_seconds"]
        if type(hop) not in (int, float) or not math.isfinite(hop) or not MIN_HOP_S - EPSILON <= hop <= MAX_HOP_S + EPSILON:
            raise ValueError(f"hop_seconds must be a number between {MIN_HOP_S} and {MAX_HOP_S} seconds")
        probabilities = block["eot_probability"]
        if not isinstance(probabilities, list) or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
                                                      for p in probabilities):
            raise ValueError("eot_probability must be a list of finite numbers in [0, 1]")
        if abs(len(probabilities) * hop - row["duration_s"]) > hop + EPSILON:
            raise ValueError(f"eot_probability must cover the audio: {len(probabilities)} frames of {hop} s "
                             f"do not end within one hop of {row['duration_s']:.3f} s")
        prediction[speaker] = {"hop_seconds": hop, "eot_probability": probabilities}
    return prediction


def score(rows, predictions, config=None):
    if not rows or len(rows) != len(predictions):
        raise ValueError("Expected one prediction per conversation")
    conversations = []
    for row, prediction in zip(rows, predictions):
        if prediction.get("id") != row["id"]:
            raise ValueError("Prediction order or conversation ID mismatch")
        parsed = parse_response(row, {s: prediction[s] for s in ("speaker_1", "speaker_2")})
        gold = row.get("gold")
        if gold is None:
            raise ValueError("Missing EOT annotations; audio-only rows cannot be scored")
        positives, pauses = [], []
        for speaker in (1, 2):
            block = parsed[f"speaker_{speaker}"]
            found, quiet = speaker_windows(gold, speaker, Stream(block["hop_seconds"], block["eot_probability"]))
            positives += found
            pauses += quiet
        conversations.append((row["id"], positives, pauses))
    positives = [window for _, found, _ in conversations for window in found]
    pauses = [window for _, _, quiet in conversations for window in quiet]
    # Pooled over every conversation: the threshold is one operating point for the whole held-out pass.
    curve = sweep(positives, pauses)
    points = {f"{limit:.2f}": operating_point(curve, limit) for limit in FPR_REPORTED}
    chosen = operating_point(curve, FPR_TARGET)
    threshold = chosen["threshold"] if chosen["threshold"] is not None else math.inf  # nothing fires
    fires = lambda window: window.statistic >= threshold
    per_conversation = [{"id": cid, "tp": sum(map(fires, found)), "fn": sum(not fires(w) for w in found),
                         "fp": sum(map(fires, quiet)), "tn": sum(not fires(w) for w in quiet)}
                        for cid, found, quiet in conversations]
    totals = {key: sum(record[key] for record in per_conversation) for key in ("tp", "fn", "fp", "tn")}
    latencies = [(window.first_crossing(threshold) - window.anchor_s) * 1000.0 for window in positives if fires(window)]
    percentiles = {f"p{p}": _percentile(latencies, p) if latencies else None for p in (10, 50, 90)}
    return {"score": chosen["recall"] if chosen["threshold"] is not None else 0.0, "details": {
        "metric": METRIC, "fpr_target": FPR_TARGET, "threshold": chosen["threshold"],
        "recall": totals["tp"] / len(positives) if positives else None,
        "fpr": totals["fp"] / len(pauses) if pauses else None,
        "operating_points": points, "positives": len(positives), "pauses": len(pauses), **totals,
        "latency_ms": percentiles,
        "curve": [{"threshold": t, "recall": r, "fpr": f} for t, r, f in curve],
        "conversations": per_conversation}}


def submission(predictions):
    """Export the probability streams as returned by the server (schema_version 2)."""
    return {"schema_version": 2, "predictions": [
        {"conversation_id": p["id"], **{s: p[s] for s in ("speaker_1", "speaker_2")}} for p in predictions]}
