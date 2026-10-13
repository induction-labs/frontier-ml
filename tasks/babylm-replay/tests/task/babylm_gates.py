"""babylm-replay trusted gates (never shipped to the agent).

- scan_submission: entry-point and filesystem checks of the replayed source tree (run in prescreen).
- run_verified: the public bits-per-byte pass, an optional server restart that wipes candidate state,
  then the /dist normalization, /score-vs-/dist agreement, bias and causal-truncation audits.
"""
from __future__ import annotations

import bisect
import math
from pathlib import Path
import random
import time

from babylm import fetch_vocab, finite_logprob, parallel, parse_score, score_split
from frontier_ml.validation.errors import VerificationFailure
from frontier_ml.validation.evaluation import FailedRequests, RequestFailure, attempt_request

# ------------------------------------------------------------------------------------------
# Submission entry points and filesystem integrity
# ------------------------------------------------------------------------------------------


def scan_submission(root) -> dict:
    """Check the replay interface, not statistical properties of source text.

    Whether code embeds external training data is established by review and the
    isolated replay, not by file sizes, suffixes, entropy or literal counts.
    """
    root = Path(root)
    violations, files, total = [], [], 0
    if not root.is_dir():
        return {"ok": False, "violations": ["submission directory is missing"], "files": [], "totals": {"bytes": 0}}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            violations.append(f"{rel}: symlinks are not allowed")
        elif path.is_dir():
            continue
        elif not path.is_file():
            violations.append(f"{rel}: special files are not allowed")
        else:
            files.append(rel)
            total += path.stat().st_size
    for name in ("train.sh", "serve.sh"):
        if not (root / name).is_file():
            violations.append(f"missing required {name}")
    return {"ok": not violations, "violations": violations, "files": files, "totals": {"bytes": total}}


# ------------------------------------------------------------------------------------------
# Distribution audits
# ------------------------------------------------------------------------------------------

LOGSUMEXP_TOLERANCE = 1e-3     # nats
MATCH_TOLERANCE = 0.02         # nats, per audited position
BIAS_TOLERANCE = 0.002         # nats, mean(score - dist) over audited positions


def parse_dist(response, vocab):
    values = response.get("logprobs") if isinstance(response, dict) else None
    if not isinstance(values, list) or len(values) != len(vocab):
        raise VerificationFailure("/dist must return one log-probability per vocabulary entry")
    if not all(finite_logprob(v, allow_neg_inf=True) for v in values):
        raise VerificationFailure("/dist log-probabilities must be <= 0 and not NaN")
    top = max(values)
    lse = top + math.log(math.fsum(math.exp(v - top) for v in values if v > -1e30))
    return values, lse



def _token_offsets(ids, vocab):
    offsets, position = [], 0
    for token in ids:
        position += len(vocab[token])
        offsets.append(position)
    return offsets


def audit_documents(transport, documents, scored, vocab, *, seed, positions=500, causal_documents=16,
                    concurrency=8, request_timeout=120, deadline=None):
    """Phase-B audits. Operations on one document are issued in increasing prefix length, so the
    server never sees a later part of a document before it is asked about an earlier position."""
    deadline = deadline or time.monotonic() + 3600
    rng = random.Random(seed)
    lengths = [len(item[0]) if item else 0 for item in scored]  # unscored documents (failed /score) are not audited
    total = sum(lengths)
    flat = rng.sample(range(total), min(positions, total))
    starts, acc = [], 0
    for length in lengths:
        starts.append(acc)
        acc += length
    plan = {}
    for index in flat:
        d = bisect.bisect_right(starts, index) - 1
        while lengths[d] == 0:  # bisect lands on the last zero-length document sharing this start
            d -= 1
        plan.setdefault(d, []).append((index - starts[d], "dist"))
    eligible = [d for d, item in enumerate(scored) if item and len(item[0]) >= 2]
    for d in rng.sample(eligible, min(causal_documents, len(eligible))):
        ids = scored[d][0]
        offsets = _token_offsets(ids, vocab)
        text = documents[d]["text"].encode("utf-8")
        cut_tokens = [k + 1 for k in range(len(ids) - 1)
                      if offsets[k] < len(text) and text[offsets[k]:offsets[k] + 1] in (b" ", b"\n", b"\t")]
        if cut_tokens:
            k = rng.choice(cut_tokens)
            plan.setdefault(d, []).append((k, "cut"))

    failures = FailedRequests(max(1, sum(len(ops) for ops in plan.values())))

    def run_document(item):
        d, operations = item
        ids, lps = scored[d]
        text = documents[d]["text"].encode("utf-8")
        results = []
        for length, kind in sorted(operations):
            timeout = max(1e-3, min(request_timeout, deadline - time.monotonic()))
            if kind == "dist":
                response = attempt_request(transport, "POST", "/dist", {"prefix_tokens": ids[:length]}, timeout=timeout,
                                           failures=failures)
                if isinstance(response, RequestFailure):
                    failures.record(f"{documents[d]['id']}:dist@{length}", response)
                    continue
                values, lse = parse_dist(response, vocab)
                results.append({"kind": "dist", "doc": d, "position": length, "logsumexp": lse,
                                "dist_logprob": float(values[ids[length]]), "score_logprob": lps[length]})
            else:
                offset = sum(len(vocab[t]) for t in ids[:length])
                truncated = text[:offset].decode("utf-8")
                response = attempt_request(transport, "POST", "/score", {"text": truncated}, timeout=timeout,
                                           failures=failures)
                if isinstance(response, RequestFailure):
                    failures.record(f"{documents[d]['id']}:cut@{length}", response)
                    continue
                t_ids, t_lps = parse_score(truncated, response, vocab)
                common = 0
                while common < min(len(t_ids), len(ids)) and t_ids[common] == ids[common]:
                    common += 1
                results.append({"kind": "cut", "doc": d, "position": length, "truncated_tokens": len(t_ids),
                                "common_prefix": common,
                                "differences": [t_lps[i] - lps[i] for i in range(common)]})
        return results

    items = sorted(plan.items())
    nested = parallel(items, run_document, concurrency, deadline)
    return [r for group in nested for r in group], failures.summary()


def audit_report(results):
    """The audit statistics judge_audits holds to the tolerances above (the self-check shows them as gates)."""
    dist = [r for r in results if r["kind"] == "dist"]
    cuts = [r for r in results if r["kind"] == "cut"]
    worst_lse = max((abs(r["logsumexp"]) for r in dist), default=0.0)
    diffs = [r["score_logprob"] - r["dist_logprob"] for r in dist]
    worst_match = max((abs(x) for x in diffs), default=0.0)
    bias = math.fsum(diffs) / len(diffs) if diffs else 0.0
    cut_diffs = [x for r in cuts for x in r["differences"]]
    worst_cut = max((abs(x) for x in cut_diffs), default=0.0)
    return {"dist_positions": len(dist), "max_abs_logsumexp": worst_lse, "max_abs_score_minus_dist": worst_match,
            "mean_score_minus_dist": bias, "causal_probes": len(cuts),
            "causal_probes_without_shared_tokens": sum(r["common_prefix"] == 0 for r in cuts),
            "causal_positions": len(cut_diffs), "max_abs_causal_difference": worst_cut}


def judge_audits(results):
    """Pure function: the audit verdict. Raises VerificationFailure on any violation."""
    report = audit_report(results)
    worst_lse, worst_match = report["max_abs_logsumexp"], report["max_abs_score_minus_dist"]
    bias, worst_cut = report["mean_score_minus_dist"], report["max_abs_causal_difference"]
    if worst_lse > LOGSUMEXP_TOLERANCE:
        raise VerificationFailure(f"/dist is not normalized: |logsumexp| = {worst_lse:.3g} > {LOGSUMEXP_TOLERANCE}")
    if worst_match > MATCH_TOLERANCE:
        raise VerificationFailure(f"/score disagrees with /dist by {worst_match:.3g} nats (> {MATCH_TOLERANCE})")
    if bias > BIAS_TOLERANCE:
        raise VerificationFailure(f"/score is systematically above /dist by {bias:.3g} nats (> {BIAS_TOLERANCE})")
    # Compare probabilities only where the tokens and their histories match.
    # Retokenization of a text prefix is not evidence of noncausal probabilities.
    if worst_cut > MATCH_TOLERANCE:
        raise VerificationFailure(f"Causal probe: truncated-text log-probs differ by {worst_cut:.3g} nats (> {MATCH_TOLERANCE})")
    return report



def run_verified(transport, data_path, config, *, restart=None):
    """Score all documents (phase A), optionally restart the server, then audit (phase B)."""
    config = dict(config or {})
    started = time.monotonic()
    documents, vocab, scored, details = score_split(transport, data_path, config)
    expected = config.get("expected_samples")
    if expected is not None and len(documents) != expected:
        raise VerificationFailure(f"Expected {expected} documents; found {len(documents)}")
    concurrency = int(config.get("concurrency", 8))
    request_timeout = float(config.get("request_timeout_seconds", 120))
    deadline = started + float(config.get("inference_timeout_seconds", 3600))
    if restart is not None:
        details["restart"] = restart()
        if fetch_vocab(transport, request_timeout) != vocab:
            raise VerificationFailure("Vocabulary changed across the server restart")
    seed = config.get("audit_seed")
    if seed is None:
        seed = random.SystemRandom().getrandbits(64)
    audit_started = time.monotonic()
    results, failed_audits = audit_documents(transport, documents, scored, vocab, seed=seed,
                                             positions=int(config.get("audit_positions", 500)),
                                             causal_documents=int(config.get("causal_documents", 16)),
                                             concurrency=concurrency, request_timeout=request_timeout, deadline=deadline)
    audit = judge_audits(results)
    audit.update(seed=str(seed), seconds=time.monotonic() - audit_started, restarted=restart is not None)
    details["audit"] = audit
    details.setdefault("failed_requests", {})["audits"] = failed_audits
    details["timing"]["total_seconds"] = time.monotonic() - started
    return {"score": details["reward"], "details": details}
