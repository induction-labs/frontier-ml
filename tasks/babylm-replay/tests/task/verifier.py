"""Trusted verifier for babylm-replay.

prescreen(): the entry-point and filesystem check (trusted tasks/babylm-replay/babylm_gates.py), before the
paid review. The shared engine then runs the LLM code review (review.py; weightless, from-scratch policy).
preflight(): requires the passing receipt, before any GPU.
launch(): training replay (frontier_ml.runtime.replay): train.sh runs offline on the pinned corpus under a
hard wall-clock limit, then serve.sh is launched on its outputs.
verify(): the receipt is re-bound to the snapshot; the public scoring pass over the private held-out set,
a server restart that wipes all candidate-writable state, then the /dist and causal audits (secret seed).
selfcheck(): the agent's `frontier selfcheck` (frontier_ml/selfcheck.py) on public inputs: the prescreen scan, a
bounded probe of the same training replay with a shorter TRAIN_BUDGET_SECONDS, and serve.sh, a /score on text with
characters outside the public data, a bounded sample of the evaluation's requests and the audit verdict only when
train.sh exits 0 within the probe.
"""
from contextlib import contextmanager
import functools
from pathlib import Path
import random
import statistics
import time

from frontier_ml.runtime import replay
from frontier_ml.runtime.replay import replay_submission
from frontier_ml.runtime.server import VerificationFailure
from frontier_ml.scoring.architecture import verify_review
from frontier_ml.taskspec import sibling
from frontier_ml.validation.errors import EvaluationError
from frontier_ml.validation.evaluation import FailedRequests, RequestFailure, attempt_request
import babylm
import babylm_gates

REVIEW_CHECKS = list(sibling(__file__, "review.py").POLICY["checks"])

# BabyLM-community/BabyLM-2026-Strict-Small @ c92ab16b4f08858304b0815706065b3354d8fc0a
CORPUS_SHA256 = {
    "bnc_spoken.train.txt": "8f3042f4104cf356052fda784682d8e8170fd1fd8c795a1c3e172f9eb2ebfa3c",
    "childes.train.txt": "c898c0a44f4380ce7a66766acd01e70eebbbd9410a1701ddf89452ea6f5aea45",
    "gutenberg.train.txt": "4ddcf5774a7947293ceb56dc76774f553b534231e7ac636c071559229cf2cb9b",
    "open_subtitles.train.txt": "9b21adecf57dd97f611d9442a2f993d2f843278eb24fc98f4165322c1ba5670a",
    "simple_wiki.train.txt": "0ac614aa069ea3908c07c9755bf80bb500d4cee15cd972e9f93cccf89639e9a7",
    "switchboard.train.txt": "89c99649777be2f76f5054e9f1aa364b7e1a62844e219fe810205be6d99399b5",
}

# Self-check: the public corpus is the pinned corpus the replay checks, so train.sh runs on the same bytes.
SELFCHECK_PUBLIC = tuple(f"corpus/{name}" for name in CORPUS_SHA256) + ("validation.jsonl",)
# How long the self-check watches train.sh (selfcheck_train_seconds overrides it). It runs train.sh as grading does
# but with TRAIN_BUDGET_SECONDS = this window minus the grace, so a train.sh that sizes its run from its budget exits
# in time for serve.sh and the gates; one that trains for a fixed hour is stopped. The contract cannot change, so the
# train.sh check's detail and the report's notes state both.
SELFCHECK_TRAIN_SECONDS = 300  # five minutes: long enough to catch a train.sh that crashes at startup or early on
SELFCHECK_TRAIN_GRACE_SECONDS = 60
# One /score beyond the public data. The corpus and validation.jsonl lack about 60 of the byte values UTF-8 text can
# hold (Cyrillic, Arabic and CJK lead bytes among them) and held-out documents do not. A tokenizer built only from
# the bytes it saw (a byte-level BPE trained without the full initial alphabet) passes every public document, then
# fails grading at the first held-out document with an unseen byte. This text holds every byte of printable UTF-8:
# all printable ASCII and tab, each continuation byte (the Latin-1 letters) and each lead byte 0xC2-0xF4; it is NFC,
# as held-out documents are.
UNSEEN_CHECK = "POST /score, characters outside the corpus"
UNSEEN_TEXT = "\n".join((
    "Русский: Съешь же ещё этих мягких французских булок, да выпей чаю.",
    "Українська, қазақша, тоҷикӣ, монгол: ґ є ї ә ғ қ ң ө ұ ү һ ӣ ӯ ҷ.",
    "Ελληνικά: Γαζέες καὶ μυρτιὲς δὲν θὰ βρῶ πιὰ στὸ χρυσαφὶ ξέφωτο; ͱ ͻ.",
    "Հայերեն Արարատ, עברית, العربية, فارسی: کتاب بزرگ, Jawi ݢ, ܣܘܪܝܝܐ, ދިވެހި, ߒߞߏ.",
    "हिन्दी, ไทย, ქართული, 中文：天地玄黄，宇宙洪荒。日本語：いろは、カタカナ。한국어: 다람쥐 헌 쳇바퀴에 타고파.",
    "Latin: ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ ąłőş ƒǎ șț ɐʃˈ q̃ ©.",
    "Symbols: — “quoted” € → ！ 😀 𐌰 \ue000 \U000E0041 \U000F0000 \U00100000; unassigned \U00040000 \U00080000.",
    "ASCII:\t!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~",
)) + "\n"
# The evaluation is sampled, not run in full: its /score requests and audits in the grader's proportions, at most
# this much of the evaluation deadline's worth (selfcheck_sample_seconds overrides it), and never more than the
# check's time budget (frontier_ml.selfcheck.selfcheck_seconds, from the sandbox's start) leaves after the first
# requests, less RESTART_RESERVE_SECONDS: 120 s for the restart's kill and wipe (root execs that take seconds; their
# 120 s and 300 s exec timeouts would leave no sample, and the sandbox's lifetime margin absorbs them), its launch
# exec, its startup deadline, GET /vocab and the first /dist at their timeouts, then the final /health probe.
SELFCHECK_SAMPLE_SECONDS = 600
RESTART_RESERVE_SECONDS = 120 + 30 + replay.STARTUP_SECONDS + 60 + 120 + 10
# run_verified's audit verdict (judge_audits), one gate per published tolerance.
AUDIT_GATES = (("/dist normalization", "max_abs_logsumexp", babylm_gates.LOGSUMEXP_TOLERANCE,
                "max |logsumexp| over {dist_positions} audited /dist positions"),
               ("/score vs /dist", "max_abs_score_minus_dist", babylm_gates.MATCH_TOLERANCE,
                "max |score - dist| over the same positions"),
               ("/score vs /dist bias", "mean_score_minus_dist", babylm_gates.BIAS_TOLERANCE,
                "mean(score - dist) over the same positions"),
               ("truncated text", "max_abs_causal_difference", babylm_gates.MATCH_TOLERANCE,
                "max difference over {causal_positions} shared positions of {causal_probes} truncated texts"))


def default_config():
    return {"train_seconds": 3600, "corpus_sha256": CORPUS_SHA256, "audit_positions": 500, "causal_documents": 16,
            "concurrency": 8, "request_timeout_seconds": 120, "inference_timeout_seconds": 3600,
            "verifier_lifetime_seconds": 10800}


def scan_or_fail(snapshot):
    scan = babylm_gates.scan_submission(snapshot)
    if not scan["ok"]:
        raise VerificationFailure("Submission interface check failed: " + "; ".join(scan["violations"][:10]))
    return scan


def prescreen(config, data_path):
    """Free static scan of the frozen snapshot, before the paid code review."""
    scan_or_fail(config["submission_path"])


def preflight(config, data_path):
    """After the shared code review, before any GPU: receipt present and static scan clean."""
    review = config.get("architecture_review_path")
    if not review or not Path(review).is_file():
        raise VerificationFailure("Trusted code review receipt (architecture_review_path) is required before provisioning")
    scan_or_fail(config["submission_path"])


@contextmanager
def launch(snapshot, config, data_path):
    scan = scan_or_fail(snapshot)  # idempotent; also guards direct operator use of launch()
    corpus = Path(data_path).parent / "corpus"
    with replay_submission(snapshot, corpus, config) as server:
        server["scan"] = {"totals": scan["totals"], "files": len(scan["files"])}
        yield server


def verify(server, data_path, config):
    if "restart" not in server or "replay" not in server:
        raise VerificationFailure("babylm-replay must be launched through its training-replay launcher")
    review = verify_review(server["snapshot"], config["architecture_review_path"], "babylm-replay", REVIEW_CHECKS)
    result = babylm_gates.run_verified(server["transport"], data_path, config, restart=server["restart"])
    details = result["details"]
    replay = dict(server["replay"])
    details["replay"] = replay
    details["gates"] = {"scan": server.get("scan"),
                        "code_review": {"reviewer": review.get("reviewer"), "checks": review.get("checks")},
                        "train_seconds": replay.get("train_seconds"), "train_limit_seconds": replay.get("limit_seconds"),
                        "serve_startup_seconds": server.get("startup_elapsed")}
    return result


def selfcheck(check, public, config):
    """The agent's self-check: no review, held-out document or score. The prescreen scan; train.sh replayed on the
    public corpus as grading replays it, but with a shorter TRAIN_BUDGET_SECONDS, for selfcheck_train_seconds; a
    train.sh still running then passes and is stopped. One that exited 0 gets serve.sh on its outputs, one public
    document and UNSEEN_TEXT through the grader's parsers and a sample of the evaluation's request pattern, sized to
    the check's time budget, for its deadlines (projected to the full counts) and audit verdict."""
    from frontier_ml.selfcheck import selfcheck_seconds
    public = Path(public)
    try:
        documents = babylm.load_documents(public / "validation.jsonl")
    except VerificationFailure as error:
        raise EvaluationError(f"The public validation documents are unusable: {error}") from error
    check.static("entry points and files", prescreen, config, None)
    check.stage("provisioning")
    finished = False
    try:
        with replay.replay_resource(config) as resource:
            ends = time.monotonic() + selfcheck_seconds(config)  # the time one check was promised to hold the GPU
            train_and_serve(check, resource, public, documents, config, ends)
            finished = True
    except VerificationFailure as error:
        # The hook owns this sandbox, not check.launch: the network monitor's verdict at its teardown lands here.
        if finished and not isinstance(error, EvaluationError):
            check.fail("offline", str(error))
        raise


def train_and_serve(check, resource, public, documents, config, ends):
    """The probe on the hook's replay sandbox, then serve.sh on the same sandbox when train.sh exited 0."""
    window = config.get("selfcheck_train_seconds", SELFCHECK_TRAIN_SECONDS)
    budget = max(1, window - SELFCHECK_TRAIN_GRACE_SECONDS)
    given = f"TRAIN_BUDGET_SECONDS={budget}"
    check.stage("training")
    try:
        record = replay.probe_training(resource, check.snapshot, public / "corpus", config, window, budget)
    except EvaluationError:
        raise
    except VerificationFailure as error:
        check.fail("train.sh", f"With {given}: {error}", kind="smoke")
    on = f" on {record['gpu']['name']}" if record.get("gpu") else ""
    log = "\nEnd of its log:\n" + (record["log_tail"][-1000:].strip() or "(empty)")
    limit = record["limit_seconds"]
    if record["running"]:
        check.record("train.sh", True, kind="smoke", measured=round(record["probe_seconds"], 1), unit="s",
                     detail=f"Given {given}, still running without error{on} after {record['probe_seconds']:.0f} s; "
                            "stopped there, so serve.sh, the startup deadline, the evaluation deadlines and the audits "
                            "were not exercised. A train.sh that sizes its run from TRAIN_BUDGET_SECONDS exits in time "
                            "for them." + log)
        check.note(f"train.sh, given {given}, was still running when the self-check stopped it after "
                   f"{record['probe_seconds']:.0f} s, so serve.sh, the restart, the speed gates and the audits were "
                   "not exercised; a train.sh that sizes its run from TRAIN_BUDGET_SECONDS exits in time for them.")
        return
    seconds = record["train_seconds"]
    over = (f" That is {seconds - budget:.0f} s past its budget: if it times its run from TRAIN_BUDGET_SECONDS, the "
            f"same overrun past {limit} s fails grading." if seconds > budget else "")
    check.record("train.sh", True, kind="smoke",
                 detail=f"Given {given}, exited 0{on} after {seconds:.0f} s, writing {len(record['output_files'])} "
                        f"files ({record['output_bytes']} bytes) to OUTPUT_DIR.{over}" + log)
    check.gate("train.sh wall clock", round(seconds, 1), limit, higher_is_better=False, unit="s",
               detail=f"Grading's hard limit, with TRAIN_BUDGET_SECONDS={limit}; this run had {given}.")
    with check.launch(launcher=functools.partial(replay.serve_trained, resource, record=record)) as server:
        # One timed transport and one clock from the first request on: grading's cold first requests count toward
        # the per-request and evaluation deadlines, and here the smoke requests are the cold ones.
        transport, started = Timed(server["transport"]), time.monotonic()
        answers = smoke(check, {**server, "transport": transport}, documents[0], config)
        check.stage("gates")
        evaluation(check, server, transport, started, answers, documents, config, ends)


def smoke(check, server, document, config):
    """GET /vocab, one public document through POST /score, UNSEEN_TEXT through POST /score and a prefix of the
    document through POST /dist, each built and parsed as the grader builds and parses them, the /dist answer held
    to the audit's per-position limits. Returns the vocabulary, which the evaluation then uses as grading uses its
    first GET /vocab, and the /dist prefix."""
    timeout = float(config.get("request_timeout_seconds", 120))
    parsed = {}

    def vocab(sample, response):
        parsed["vocab"] = babylm.parse_vocab(response)
        return {"tokens": len(parsed["vocab"]), "special": parsed["vocab"].count(None)}

    def score(sample, response):
        parsed["ids"], parsed["logprobs"] = babylm.parse_score(sample["text"], response, parsed["vocab"])
        return {"id": sample["id"], "bytes": len(sample["text"].encode("utf-8")), "tokens": len(parsed["ids"])}

    def unseen(sample, response):
        try:
            ids, _ = babylm.parse_score(sample, response, parsed["vocab"])
        except VerificationFailure as error:
            diagnosis = unseen_diagnosis(sample, response, parsed["vocab"])
            raise VerificationFailure(f"{error}. {diagnosis}" if diagnosis else str(error)) from None
        return {"bytes": len(sample.encode("utf-8")), "tokens": len(ids)}

    def dist(sample, response):
        values, logsumexp = babylm_gates.parse_dist(response, parsed["vocab"])
        position = len(sample)
        difference = parsed["logprobs"][position] - float(values[parsed["ids"][position]])
        # judge_audits fails the run on any one position past these; the mean bias needs the audit's many positions.
        if abs(logsumexp) > babylm_gates.LOGSUMEXP_TOLERANCE:
            raise VerificationFailure(f"/dist is not normalized: |logsumexp| = {abs(logsumexp):.3g} > "
                                      f"{babylm_gates.LOGSUMEXP_TOLERANCE}")
        if abs(difference) > babylm_gates.MATCH_TOLERANCE:
            raise VerificationFailure(f"/score disagrees with /dist by {abs(difference):.3g} nats "
                                      f"(> {babylm_gates.MATCH_TOLERANCE}) at token {position}")
        return {"prefix_tokens": position, "logprobs": len(values), "logsumexp": logsumexp,
                "score_minus_dist": difference}

    check.request(server, "/vocab", method="GET", name="GET /vocab", parse=vocab, timeout=min(timeout, 60))
    check.request(server, "/score", {"text": document["text"]}, parse=score, sample=document, timeout=timeout)
    check.request(server, "/score", {"text": UNSEEN_TEXT}, name=UNSEEN_CHECK, parse=unseen, sample=UNSEEN_TEXT,
                  timeout=timeout)
    prefix = parsed["ids"][:len(parsed["ids"]) // 2]
    check.request(server, "/dist", {"prefix_tokens": prefix}, parse=dist, sample=prefix, timeout=timeout)
    return {"vocab": parsed["vocab"], "prefix": prefix}


def unseen_diagnosis(text, response, vocab):
    """Why a /score answer for `text` does not concatenate to it, for the agent: the text's bytes no /vocab token
    holds, else where the returned tokens part from the text. Empty when the answer was rejected for another reason."""
    want = text.encode("utf-8")
    absent = sorted(set(want) - set().union(*(token for token in vocab if token)))
    if absent:
        examples = {}
        for character in text:
            for byte in character.encode("utf-8"):
                examples.setdefault(byte, character)
        found = (f"No /vocab token holds the bytes {', '.join(f'{b:02x} ({examples[b]!r})' for b in absent[:16])}"
                 f"{f' and {len(absent) - 16} more' if len(absent) > 16 else ''}, so no tokenization reproduces text "
                 "containing them.")
    else:
        ids = response.get("token_ids") if isinstance(response, dict) else None
        if not isinstance(ids, list) or not all(type(t) is int and 0 <= t < len(vocab) and vocab[t] for t in ids):
            return ""
        got = b"".join(vocab[t] for t in ids)
        if got == want:
            return ""
        at = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
        found = (f"The tokens hold {len(got)} bytes for the text's {len(want)} and part from it at byte {at}, in "
                 f"{want[max(0, at - 24):at + 24].decode('utf-8', 'replace')!r}.")
    return (found + " The token bytes must concatenate to any UTF-8 input, and held-out documents are not limited to "
            "the characters of the public data.")


class Timed:
    """The grading transport, timing every answered request: the per-request deadline's measurements, and the
    count a phase that ran out of its deadline is projected from. Failed and timed-out requests are grading's
    per-item failures, counted separately."""

    def __init__(self, transport):
        self.transport, self.seconds = transport, []

    def request(self, method, path, payload=None, **kwargs):
        started = time.monotonic()
        response = self.transport.request(method, path, payload, **kwargs)
        self.seconds.append(time.monotonic() - started)
        return response


def sample_sizes(window, config, total):
    """How much of the evaluation's request pattern fits `window` seconds at the grader's required rate, as
    {phase: (sent, full)}: all of it when the whole evaluation deadline fits, else each phase in proportion but at
    least twice the concurrency and 8 requests (docs/SELFCHECK.md's rule for bounded gates); the truncation probes
    keep their share of the audit requests, at least one."""
    limit = config.get("inference_timeout_seconds", 3600)
    positions, causal = int(config.get("audit_positions", 500)), int(config.get("causal_documents", 16))
    minimum = max(2 * int(config.get("concurrency", 8)), 8)

    def size(full):
        return full if window >= limit else min(full, max(minimum, int(full * window / limit)))
    audits = size(positions + causal)
    cuts = min(causal, max(1, audits * causal // max(1, positions + causal)))
    return {"score": (size(total), total), "dist": (audits - cuts, positions), "cut": (cuts, causal)}


def evaluation(check, server, transport, started, answers, documents, config, ends):
    """run_verified's requests and audit verdict without its score, on a sample sized to what the check's time
    budget leaves: /score requests on public documents (resampled under unique IDs), the server restart, then the
    /dist and truncation audits, `concurrency` in flight, each sampled phase against its share of the evaluation
    deadline. Records the evaluation deadline as the full evaluation's projected time, the per-request deadline, the
    restart and the audit gates. `answers` is smoke()'s: the vocabulary and the /dist prefix."""
    concurrency = int(config.get("concurrency", 8))
    request_timeout = float(config.get("request_timeout_seconds", 120))
    limit = config.get("inference_timeout_seconds", 3600)
    vocab = answers["vocab"]
    # Grading's clock starts at its first, cold requests, sent once: the smoke requests stand in for them, unscaled.
    cold = time.monotonic() - started
    left = ends - time.monotonic() - RESTART_RESERVE_SECONDS
    window = min(config.get("selfcheck_sample_seconds", SELFCHECK_SAMPLE_SECONDS), left)
    sizes = sample_sizes(window, config, config.get("expected_samples") or len(documents))
    (count, total), (dists, positions), (cuts, causal) = sizes["score"], sizes["dist"], sizes["cut"]
    full = {"score": total, "audit": positions + causal}
    sent = {"score": count, "audit": dists + cuts}  # what each phase's time covers; after a miss, the answered ones
    score_scale, audit_scale = total / count, full["audit"] / max(1, sent["audit"])
    needed = limit / min(score_scale, audit_scale)  # the longest the sampled phases can take within their shares
    if needed > left:
        check.note(f"After training, serve.sh's startup and the first requests, {max(0, left):.0f} s of the check's "
                   f"time was left for a sample of the evaluation that needs {needed:.0f} s, so the evaluation "
                   "deadline, the restart and the audits were not exercised.")
        return
    sampled = score_scale > 1 or audit_scale > 1
    rng = random.Random()
    picks = rng.sample(documents, count) if count <= len(documents) else rng.choices(documents, k=count)
    batch = [{**row, "id": f"{row['id']}#{index}"} for index, row in enumerate(picks)]
    restart, restarted, warmed = server.get("restart"), None, False
    scoring, lost = FailedRequests(len(batch)), []
    spent = {"score": 0.0, "restart": 0.0, "audit": 0.0}
    labels = {"score": "/score requests", "audit": "/dist and truncation requests"}
    sample = (f"The evaluation deadline was projected from a sample: {count} of {total} /score requests and "
              f"{dists + cuts} of {positions + causal} audit requests")

    def projected():
        """The full evaluation's time: each phase's time per request scaled to its full count, the rest as measured."""
        return cold + spent["restart"] + sum(spent[name] * full[name] / max(1, sent[name]) for name in full)

    def missed(detail, measured=None):
        """Fail the evaluation deadline; a projected one names its sample in the notes, as a passing one does."""
        if sampled:
            check.note(sample + ".")
        check.fail("evaluation deadline", detail, measured=measured, required=limit, unit="s", higher_is_better=False)

    def phase(name, step, share, run):
        """`run(deadline)` within `share` seconds, the sample's part of what is left of the evaluation deadline.
        Grading fails the run on an exception from these requests or their parsers. Past the deadline, that is the
        cause, and the full evaluation is projected from the requests answered by then."""
        began, before = time.monotonic(), len(transport.seconds)
        try:
            result = run(began + share)
        except EvaluationError:
            raise
        except Exception as error:  # noqa: BLE001 - parsers and the deadline, on the candidate's answers
            spent[name] = time.monotonic() - began
            if spent[name] < share:
                check.fail(step, f"{type(error).__name__}: {error}", kind="smoke")
            planned, answered = sent[name], len(transport.seconds) - before
            ran_out = (f"the sample's {step} ran out of their {max(0, share):.0f} s share of the {limit} s deadline"
                       if sampled else f"the {step} ran out of the {limit} s deadline")
            if not answered:
                missed(f"In the evaluation, {ran_out} with none of {planned} answered, so it would take more than "
                       f"{limit} s")
            sent[name] = answered
            measured = projected()
            scope = ("the evaluation" if name == "audit"
                     else "the first requests and the /score requests, before the restart and the audits,")
            if sampled:
                basis = (f"{answered} of {total} {labels['score']}" if name == "score" else
                         f"{count} of {total} {labels['score']} and {answered} of {full['audit']} {labels['audit']}")
                detail = (f"Projected from {basis} at concurrency {concurrency}: {ran_out} with {answered} of "
                          f"{planned} answered, so")
            else:
                detail = (f"In the evaluation, {ran_out} with {answered} of {planned} answered; at that pace, "
                          f"{concurrency} in flight,")
            missed(f"{detail} {scope} would take about {measured:.0f} s", round(measured, 1))
        spent[name] = time.monotonic() - began
        return result

    scored = phase("score", "scoring requests", (limit - cold) / score_scale, lambda deadline: babylm.score_documents(
        transport, batch, vocab, concurrency=concurrency, request_timeout=request_timeout, deadline=deadline,
        failures=scoring))
    if callable(restart):
        # The kill and wipe are root execs over what the candidate's processes left behind; once candidate code
        # has run, their failing or timing out (RuntimeError, TimeoutError) is the submission's, as serve.sh not
        # coming back is. Otherwise a submission could end every check in a refunded error.
        began = time.monotonic()
        try:
            restarted = restart()
        except EvaluationError:
            raise
        except (VerificationFailure, RuntimeError, TimeoutError) as error:
            check.fail("restart", f"{type(error).__name__}: {error}", kind="smoke")
        try:
            same = babylm.fetch_vocab(transport, request_timeout) == vocab
        except EvaluationError:
            raise
        except Exception as error:  # noqa: BLE001 - the candidate's answer
            check.fail("restart", f"GET /vocab after the restart: {type(error).__name__}: {error}", kind="smoke")
        if not same:
            check.fail("restart", "Vocabulary changed across the server restart", kind="smoke")
        remaining = limit - cold - spent["score"] * score_scale - (time.monotonic() - began)
        if audit_scale > 1 and remaining > 0:
            # Grading's first audit requests meet the restarted server's cold start (a lazy compile, say) once; in
            # the audit sample it would be scaled to the full count. The smoke /dist, sent again, takes it unscaled.
            warmed = True
            warm = attempt_request(transport, "POST", "/dist", {"prefix_tokens": answers["prefix"]},
                                   timeout=min(request_timeout, remaining))
            if isinstance(warm, RequestFailure):
                lost.append(f"the first /dist after the restart: {warm.error}")
            else:
                try:
                    babylm_gates.parse_dist(warm, vocab)
                except EvaluationError:
                    raise
                except Exception as error:  # noqa: BLE001 - the candidate's answer, which the audits would reject
                    check.fail("audit requests", f"The first /dist after the restart was rejected: "
                               f"{type(error).__name__}: {error}", kind="smoke")
        spent["restart"] = time.monotonic() - began
    restart_text = (f"the restart{' and its first /dist' if warmed else ''} ({spent['restart']:.0f} s)"
                    if callable(restart) else "no restart")
    if projected() >= limit:
        scaled = (f"Projected from {count} of {total} /score requests at concurrency {concurrency}: they scale to "
                  f"{spent['score'] * score_scale:.0f} s" if sampled
                  else f"The /score requests took {spent['score']:.0f} s")
        missed(f"{scaled}, which with the first requests ({cold:.1f} s) and {restart_text} leaves no time of the "
               f"{limit} s deadline for the audits", round(projected(), 1))
    results, audits = phase("audit", "audit requests", (limit - projected()) / audit_scale,
                            lambda deadline: babylm_gates.audit_documents(
                                transport, batch, scored, vocab, seed=rng.getrandbits(64), positions=dists,
                                causal_documents=cuts, concurrency=concurrency, request_timeout=request_timeout,
                                deadline=deadline))
    startup = restarted.get("startup_seconds") if isinstance(restarted, dict) else None
    if sampled:
        detail = (f"Projected from {count + dists + cuts} of {total + positions + causal} requests at concurrency "
                  f"{concurrency}: {count} of {total} /score requests on public documents in {spent['score']:.1f} s "
                  f"and {dists + cuts} of {positions + causal} /dist and truncation requests in "
                  f"{spent['audit']:.1f} s, each scaled to its full count, plus the first requests ({cold:.1f} s) "
                  f"and {restart_text} as measured")
        check.note(sample + "; the audit gates hold that sample to grading's tolerances.")
    else:
        detail = (f"From the first smoke request: {count} /score requests on public documents, {restart_text}, "
                  f"then {audits['requests']} /dist and truncation requests, {concurrency} in flight")
    check.gate("evaluation deadline", round(projected(), 1), limit, higher_is_better=False, unit="s", detail=detail)
    check.gate("slowest request", round(max(transport.seconds), 2), request_timeout, higher_is_better=False, unit="s",
               detail=f"{len(transport.seconds)} answered requests, median "
                      f"{statistics.median(transport.seconds):.2f} s")
    if isinstance(startup, (int, float)):
        check.gate("restart startup", round(startup, 1), replay.STARTUP_SECONDS, higher_is_better=False, unit="s",
                   detail="GET /health after the kill and the wipe of candidate state")
    # Grading's per-item policy: a request that still fails after the retries costs its document (scored as uniform
    # bytes) or its audit request; only more than 5% of a phase's requests fails the run. Each sampled phase
    # already held itself to that share (FailedRequests); the first /dist after the restart counts as an audit.
    score_failures = scoring.summary()
    failed = score_failures["total"] + audits["total"] + len(lost)
    requests = score_failures["requests"] + audits["requests"] + warmed
    within = (audits["total"] + len(lost)) * 100 <= audits["limit_percent"] * (audits["requests"] + warmed)
    errors = (lost + score_failures["errors"] + audits["errors"])[:3]
    if failed:
        check.note(f"{failed} of {requests} requests failed or timed out after the retries: grading scores such a "
                   "document as uniform bytes (8 bits per byte) and skips such an audit request, at a cost to the "
                   "score.")
    check.record("requests answered", within, kind="smoke",
                 detail=f"{failed} of {requests} requests failed or timed out after the retries: {errors}. Grading "
                        f"scores such a document as uniform bytes (8 bits per byte), skips such an audit request and "
                        f"fails the run only when more than {audits['limit_percent']}% of a phase's requests fail."
                        if failed else "Every request was answered.")
    report = babylm_gates.audit_report(results)
    passed = [check.gate(name, report[key], tolerance, higher_is_better=False, unit="nats",
                         detail=text.format(**report))["passed"] for name, key, tolerance, text in AUDIT_GATES]
    try:
        babylm_gates.judge_audits(results)
    except VerificationFailure as error:
        if all(passed):  # grading's verdict, should it ever check more than these gates
            check.fail("audits", str(error))
