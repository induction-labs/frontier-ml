"""babylm-replay public metric: bits per UTF-8 byte of the candidate server on a set of documents.

The agent's validate.py and the trusted verifier share this request template and metric.

HTTP interface of the candidate server used here:
  GET  /vocab                    -> {"tokens": [hex-encoded bytes | null, ...]}   (null = special token)
  POST /score {"text": str}      -> {"token_ids": [int], "logprobs": [float]}      (natural log)
The concatenated bytes of the returned tokens must equal the UTF-8 encoding of the text, and every
log-prob must be finite and <= 0; otherwise bits per byte cannot be computed and the run fails with the
offending document named. logprobs[i] = ln p(token_i | token_0..token_{i-1}).
bpb = sum(-logprobs) / ln 2 / total UTF-8 bytes; reward = reward(bpb), the leaderboard's linear scale in bpb.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

# The reward is linear in bits per byte, clamped to [0, 1]: REWARD_FLOOR_BPB scores 0 and REWARD_CEILING_BPB
# scores 1. These two numbers mirror task.yaml's score_normalization (a test keeps them equal), so the
# number validate.py prints is the number the leaderboard records.
REWARD_FLOOR_BPB = 1.461
REWARD_CEILING_BPB = 1.25


def reward(bpb):
    """The leaderboard reward for a bits-per-byte value: linear between the floor and the ceiling, clamped."""
    return min(1.0, max(0.0, (bpb - REWARD_FLOOR_BPB) / (REWARD_CEILING_BPB - REWARD_FLOOR_BPB)))

from frontier_ml.validation.errors import VerificationFailure
from frontier_ml.validation.evaluation import FailedRequests, RequestFailure, attempt_request

MAX_VOCAB = 1 << 20


def parse_vocab(response):
    tokens = response.get("tokens") if isinstance(response, dict) else None
    if not isinstance(tokens, list) or not 2 <= len(tokens) <= MAX_VOCAB:
        raise VerificationFailure(f"/vocab must return 2..{MAX_VOCAB} tokens")
    vocab = []
    for item in tokens:
        if item is None:
            vocab.append(None)
            continue
        if not isinstance(item, str) or not item or len(item) % 2 or len(item) > 1024:
            raise VerificationFailure("/vocab tokens must be non-empty hex strings (<=512 bytes) or null")
        try:
            vocab.append(bytes.fromhex(item))
        except ValueError as exc:
            raise VerificationFailure("/vocab token is not hex") from exc
    return vocab


def finite_logprob(value, *, allow_neg_inf=False):
    if type(value) not in (int, float):
        return False
    if value != value or value > 1e-6:
        return False
    return allow_neg_inf or value > -1e30


def parse_score(text, response, vocab):
    if not isinstance(response, dict):
        raise VerificationFailure("/score must return an object")
    ids, logprobs = response.get("token_ids"), response.get("logprobs")
    if not isinstance(ids, list) or not isinstance(logprobs, list) or len(ids) != len(logprobs):
        raise VerificationFailure("/score must return equal-length token_ids and logprobs")
    pieces = []
    for token in ids:
        if type(token) is not int or not 0 <= token < len(vocab) or vocab[token] is None:
            raise VerificationFailure("/score returned an invalid or special token id")
        pieces.append(vocab[token])
    if b"".join(pieces) != text.encode("utf-8"):
        raise VerificationFailure("/score tokens do not concatenate to the exact UTF-8 text")
    if not all(finite_logprob(v) for v in logprobs):
        raise VerificationFailure("/score log-probabilities must be finite and <= 0")
    return ids, [float(v) for v in logprobs]



def load_documents(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise VerificationFailure("Evaluation dataset is empty")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)) or any(not isinstance(r.get("text"), str) or not r["text"] for r in rows):
        raise VerificationFailure("Documents need unique ids and non-empty text")
    return rows


def parallel(items, function, concurrency, deadline):
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="babylm-http") as pool:
        futures = [pool.submit(function, item) for item in items]
        results = []
        try:
            for future in futures:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise VerificationFailure("Evaluation deadline exceeded")
                results.append(future.result(timeout=remaining))
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        return results


# A document whose /score request produced no usable response (transport failure after the shared retries, or a
# timeout) scores as uniform bytes, 8 bits per byte: the task's zero for that item, as the shared contract promises.
# A malformed response is an interface violation and still fails the run.
UNIFORM_BYTE_NATS = math.log(256)


def fetch_vocab(transport, request_timeout=120):
    response = attempt_request(transport, "GET", "/vocab", timeout=min(request_timeout, 60))
    if isinstance(response, RequestFailure):
        raise VerificationFailure(f"GET /vocab produced no usable response: {response.error}")
    return parse_vocab(response)


def score_documents(transport, documents, vocab, *, concurrency=8, request_timeout=120, deadline=None, failures=None):
    """Score every document; a request that fails under the shared retry policy scores None (see bits_per_byte)."""
    deadline = deadline or time.monotonic() + 3600
    failures = failures if failures is not None else FailedRequests(len(documents))

    def one(row):
        timeout = max(1e-3, min(request_timeout, deadline - time.monotonic()))
        response = attempt_request(transport, "POST", "/score", {"text": row["text"]}, timeout=timeout, failures=failures)
        if isinstance(response, RequestFailure):
            failures.record(row["id"], response)  # fails closed above the shared limit
            return None
        try:
            return parse_score(row["text"], response, vocab)
        except VerificationFailure as exc:
            raise VerificationFailure(f"document {row['id']}: {exc}") from exc

    return parallel(documents, one, concurrency, deadline)


def document_nll(row, item):
    """Negative log-likelihood of one document in nats; an unscored document counts as uniform bytes."""
    if item is None:
        return len(row["text"].encode("utf-8")) * UNIFORM_BYTE_NATS
    return -math.fsum(item[1])


def bits_per_byte(documents, scored):
    nll = math.fsum(document_nll(row, item) for row, item in zip(documents, scored))
    total_bytes = sum(len(row["text"].encode("utf-8")) for row in documents)
    return nll / math.log(2) / total_bytes, nll, total_bytes




def score_split(transport, data_path, config):
    """Fetch the vocabulary and score every document. Returns (documents, vocab, scored, details)."""
    config = dict(config or {})
    documents = load_documents(data_path)
    concurrency = int(config.get("concurrency", 8))
    request_timeout = float(config.get("request_timeout_seconds", 120))
    started = time.monotonic()
    deadline = started + float(config.get("inference_timeout_seconds", 3600))
    vocab = fetch_vocab(transport, request_timeout)
    failures = FailedRequests(len(documents))
    scored = score_documents(transport, documents, vocab, concurrency=concurrency,
                             request_timeout=request_timeout, deadline=deadline, failures=failures)
    bpb, nll, total_bytes = bits_per_byte(documents, scored)
    genres = {}
    for row, item in zip(documents, scored):
        g = genres.setdefault(row.get("genre", "all"), {"bytes": 0, "nll": 0.0, "tokens": 0, "documents": 0})
        g["bytes"] += len(row["text"].encode("utf-8"))
        g["nll"] += document_nll(row, item)
        g["tokens"] += len(item[0]) if item else 0
        g["documents"] += 1
    per_genre = {k: {"bpb": v["nll"] / math.log(2) / v["bytes"], "bytes": v["bytes"], "documents": v["documents"],
                     "tokens": v["tokens"]} for k, v in sorted(genres.items())}
    details = {"metric": "bits_per_utf8_byte", "bpb": bpb, "reward": reward(bpb), "nll_nats": nll,
               "bytes": total_bytes, "documents": len(documents), "tokens": sum(len(i[0]) for i in scored if i),
               "unscored_documents": sum(1 for i in scored if i is None),
               "vocab_size": len(vocab), "per_genre": per_genre,
               "failed_requests": {"scoring": failures.summary()},
               "timing": {"scoring_seconds": time.monotonic() - started}}
    return documents, vocab, scored, details


def run(transport, data_path, config):
    """Bits per byte over every document in data_path; score = reward(bpb), linear in bpb."""
    _, _, _, details = score_split(transport, data_path, config)
    return {"score": details["reward"], "details": details}
