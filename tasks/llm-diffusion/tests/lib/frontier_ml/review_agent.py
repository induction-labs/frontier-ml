"""Run a fresh Claude Code or Codex SDK review with paginated static evidence."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from .review_evidence import documents

BACKENDS = {"claude-code", "codex", "api"}
# Session endings a fresh session cannot improve: the model finished its turn or a
# configured review limit was reached. Any other ending without a verdict is treated
# as a transient harness or API failure (API errors, safety-classifier refusals of
# the request, 5xx/rate-limit/connection failures after the SDK's own retries, and
# harness crashes with no result) and may be retried in a new session.
FINAL_TERMINAL_REASONS = {"completed", "no_verdict", "max_turns", "budget_exhausted",
                          "structured_output_retry_exhausted", "prompt_too_long",
                          "malformed_tool_use_exhausted", "blocking_limit", "rapid_refill_breaker",
                          "error_max_turns", "error_max_budget_usd"}
MIN_RETRY_SECONDS = 5
# Model API errors (5xx, overload, a failing credential check) back off exponentially from retry_backoff_seconds up to
# this cap over max_attempts attempts, so a review rides out an outage of roughly half an hour (a 25-minute Anthropic
# credential outage on 2026-09-29 exhausted three attempts 2 s apart). Other retryable failures retry quickly.
API_RETRY_CAP_SECONDS = 600
QUICK_RETRY_SECONDS = 2
# A safeguard refusal of the request is deterministic for a given prompt (reproduced on
# claude-opus-5 for the OCR policy), so it is final: the review is incomplete on the configured
# model, and there is no fallback model. (FML_REVIEW_FALLBACK_MODEL is no longer read.)
PROMPT = ("Review this frozen benchmark submission against every check in the operator policy below. "
          "The evidence is a participant's own submitted source code, configuration, logs and checkpoint "
          "inventories, exposed as read-only documents; judge only what that submitted code does. "
          "Use the evidence MCP tools to list, search and read documents as needed. "
          "Start with source architecture.json and serve.sh, follow the training and inference code paths, "
          "and inspect weight inventories, trusted audits and uninspected-file records. "
          "Read further pages whenever needed; filenames and candidate claims are not proof. "
          "Cite original submission paths, not document IDs. Preserve findings and unresolved checks "
          "across context compaction. Return incomplete only when evidence the submission contains could not be "
          "retrieved within the review budget; decide every check the shipped evidence allows. "
          "When trajectory documents are listed, read the one at path trajectory for its index, then search the "
          "session (for example for downloads, training runs and copies into the output directory) to test the "
          "shipped files' provenance, initialization and training claims. Cite steps as trajectory#<step_id>.\n")
# Reading a trajectory adds evidence to the review; these defaults apply only when one is supplied, so reviews of
# runs without a trajectory keep their limits. Operator options override both.
TRAJECTORY_LIMITS = {"timeout_seconds": 1800, "max_tool_calls": 384, "max_turns": 192, "max_budget_usd": 10}
_now, _sleep = time.monotonic, time.sleep


REVIEW_PROVIDERS = {"anthropic", "bedrock"}
BEDROCK_DEFAULT_REGION = "us-east-1"


def review_provider(options, backend="claude-code"):
    """Where the Claude Code reviewer's model API lives: the Anthropic API (ANTHROPIC_API_KEY or FML_REVIEW_API_KEY,
    the default) or Amazon Bedrock (FML_REVIEW_PROVIDER=bedrock with AWS_BEARER_TOKEN_BEDROCK). Other backends
    always use their own vendor."""
    provider = (options.get("provider") or os.environ.get("FML_REVIEW_PROVIDER") or "anthropic").strip().lower()
    if provider not in REVIEW_PROVIDERS:
        raise ValueError("architecture_review.provider must be anthropic or bedrock")
    if provider == "bedrock" and backend != "claude-code":
        raise ValueError("The Bedrock review provider needs the claude-code review backend")
    return provider


def bedrock_model_id(model):
    """Bedrock's cross-region inference profile for an Anthropic model name; Bedrock IDs pass through unchanged."""
    return "us.anthropic." + model if model.startswith("claude-") else model


def review_backend(options):
    default = "api" if options.get("base_url") or os.environ.get("FML_REVIEW_BASE_URL") else "claude-code"
    backend = options.get("backend") or os.environ.get("FML_REVIEW_BACKEND") or default
    if backend not in BACKENDS:
        raise ValueError("architecture_review.backend must be claude-code, codex, or api")
    return backend


def attempt_paths(trace_path, attempt):
    """Attempt 1 keeps the receipt-adjacent names; later attempts add an .attemptN infix."""
    trace_path = Path(trace_path)
    stem = trace_path.name[:-len(trace_path.suffix)] if trace_path.suffix else trace_path.name
    if attempt > 1:
        stem += f".attempt{attempt}"
    return {"transcript": trace_path.with_name(stem + trace_path.suffix),
            "tool_log": trace_path.with_name(stem + ".tools.jsonl"),
            "stderr": trace_path.with_name(stem + ".stderr.txt")}


def attempt_outcome(events, code):
    """Summarize one SDK session from its transcript, including whether a fresh session may retry it.

    A verdict, a finished model turn and every configured limit are final. The Claude
    harness reports API failures as a synthetic assistant frame (is_api_error_message)
    and a result with terminal_reason api_error, whether the SDK returns that result or
    rethrows its text; either transient ending qualifies for a retry.
    """
    verdict = any(e.get("type") == "review_result" for e in events)
    error = next((e for e in reversed(events) if e.get("type") == "review_error"), None)
    sdk = [e["event"] for e in events if e.get("type") == "sdk_event" and isinstance(e.get("event"), dict)]
    result = next((e for e in reversed(sdk) if e.get("type") == "result"), None)
    api_error = any(e.get("type") == "assistant" and e.get("is_api_error_message") for e in sdk)
    refusal = next((e for e in sdk if e.get("type") == "system" and e.get("subtype") == "model_refusal_no_fallback"), None)
    if verdict:
        reason = "completed"
    elif result is None:
        # Codex reports no result event: a failed turn is transient, a completed turn without text is not.
        kinds = {e.get("type") for e in sdk}
        reason = ("turn_failed" if "turn.failed" in kinds else
                  "no_verdict" if "turn.completed" in kinds else "sdk_crash")
    elif result.get("terminal_reason"):
        reason = result["terminal_reason"]
    elif api_error or result.get("stop_reason") == "refusal":
        reason = "api_error"
    else:
        reason = "no_verdict" if result.get("subtype") == "success" else result.get("subtype") or "unknown"
    outcome = {"exit_code": code, "verdict": verdict, "terminal_reason": reason,
               "retryable": not verdict and reason not in FINAL_TERMINAL_REASONS,
               "api_error": api_error or reason == "api_error",
               "refused": bool(refusal) or bool(result and result.get("stop_reason") == "refusal"),
               "refusal_category": refusal.get("api_refusal_category") if refusal else None}
    if result:
        outcome.update(stop_reason=result.get("stop_reason"), num_turns=result.get("num_turns"),
                       cost_usd=result.get("total_cost_usd"), session_id=result.get("session_id"))
    if error:
        outcome["error"] = error.get("error")
    return outcome


class SDKReviewer:
    def __init__(self, backend, options):
        self.backend = backend
        self.options = options
        self.metadata = {"backend": backend, "harness": "sdk", "version": "review-agent-v2"}

    def __call__(self, *, model, system, policy, evidence, response_format, trace_path):
        defaults = {"timeout_seconds": 900, "max_tool_calls": 256, "max_turns": 128, "max_budget_usd": 5}
        if evidence.get("trajectory"):
            defaults.update(TRAJECTORY_LIMITS)
        timeout = self.options.get("timeout_seconds", defaults["timeout_seconds"])
        max_calls = self.options.get("max_tool_calls", defaults["max_tool_calls"])
        max_turns = self.options.get("max_turns", defaults["max_turns"])
        budget = self.options.get("max_budget_usd", defaults["max_budget_usd"])
        max_attempts = self.options.get("max_attempts", 6)
        backoff = self.options.get("retry_backoff_seconds", 60)
        for key, value in [("timeout_seconds", timeout), ("max_tool_calls", max_calls), ("max_turns", max_turns),
                           ("max_budget_usd", budget), ("max_attempts", max_attempts)]:
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Review {key} must be positive and finite")
        if type(max_calls) is not int or type(max_turns) is not int or type(max_attempts) is not int:
            raise ValueError("Review tool, turn and attempt limits must be integers")
        if type(backoff) not in (int, float) or not math.isfinite(backoff) or backoff < 0:
            raise ValueError("Review retry_backoff_seconds must be finite and nonnegative")
        if self.options.get("refusal_fallback_model") is not None:
            raise ValueError("Review refusal_fallback_model is not supported: the reviewer never falls back to "
                             "another model")
        provider = review_provider(self.options, self.backend)
        if provider == "bedrock":
            key_name = "AWS_BEARER_TOKEN_BEDROCK"
            key = os.environ.get(key_name)
            if not key:
                raise ValueError("Bedrock review needs AWS_BEARER_TOKEN_BEDROCK")
            region = (self.options.get("aws_region") or os.environ.get("FML_REVIEW_AWS_REGION")
                      or os.environ.get("AWS_REGION") or BEDROCK_DEFAULT_REGION)
            model = bedrock_model_id(model)
            self.metadata.update(provider="bedrock", aws_region=region)
        else:
            key_name = "ANTHROPIC_API_KEY" if self.backend == "claude-code" else "OPENAI_API_KEY"
            key = (os.environ.get("FML_REVIEW_API_KEY")
                   or (os.environ.get("CODEX_API_KEY") if self.backend == "codex" else None)
                   or os.environ.get(key_name))
            if not key:
                raise ValueError(f"{self.backend} review needs FML_REVIEW_API_KEY or {key_name}")
            if self.backend == "claude-code":
                self.metadata.update(provider="anthropic")
        node = shutil.which("node")
        if not node:
            raise ValueError("Review SDKs need Node.js; rebuild the verifier image")
        worker = Path(__file__).with_name("review_agent") / "worker.mjs"
        trace_path = Path(trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        base_url = self.options.get("base_url") or os.environ.get("FML_REVIEW_BASE_URL")
        with tempfile.TemporaryDirectory(prefix="frontier-review-") as directory:
            root = Path(directory)
            docs = documents(evidence)
            evidence_path = root / "evidence.json"
            evidence_path.write_text(json.dumps(docs, ensure_ascii=False))
            prompt = PROMPT + json.dumps({"policy": policy, "evidence_documents": len(docs)}, ensure_ascii=False)
            self.metadata.update(initial_prompt_bytes=len(prompt.encode()), document_count=len(docs), attempts=[],
                                 limits={"timeout_seconds": timeout, "max_tool_calls": max_calls,
                                         **({"max_turns": max_turns, "max_budget_usd": budget}
                                            if self.backend == "claude-code" else {}),
                                         "max_attempts": max_attempts, "retry_backoff_seconds": backoff})
            deadline = _now() + timeout
            current = model
            for attempt in range(1, max_attempts + 1):
                # Every attempt is a fresh session: new config homes, workspace, transcript and tool log.
                home = root / f"attempt{attempt}"
                cwd = home / "workspace"
                for name in ("workspace", "home", "claude", "codex", "tmp"):
                    (home / name).mkdir(parents=True)
                paths = attempt_paths(trace_path, attempt)
                env = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
                       "HOME": str(home / "home"), "TMPDIR": str(home / "tmp"),
                       "CLAUDE_CONFIG_DIR": str(home / "claude"), "CODEX_HOME": str(home / "codex"),
                       "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_AUTOUPDATER": "1",
                       "PYTHONDONTWRITEBYTECODE": "1", key_name: key}
                if provider == "bedrock":
                    env.update(CLAUDE_CODE_USE_BEDROCK="1", AWS_REGION=region)
                elif base_url and self.backend == "claude-code":
                    env["ANTHROPIC_BASE_URL"] = base_url
                request = {"backend": self.backend, "model": current, "baseUrl": base_url, "system": system,
                           "prompt": prompt, "schema": response_format["json_schema"]["schema"], "cwd": str(cwd),
                           "python": sys.executable,
                           "mcpArgs": [str(Path(__file__).with_name("review_evidence.py")), str(evidence_path),
                                       str(paths["tool_log"]), str(max_calls)],
                           "maxTurns": max_turns, "maxBudgetUsd": budget}
                request_path = home / "request.json"
                request_path.write_text(json.dumps(request))
                started, code, expired = _now(), None, None
                with paths["transcript"].open("x") as trace, paths["stderr"].open("x") as stderr:
                    process = subprocess.Popen([node, str(worker), str(request_path)], cwd=cwd, env=env,
                                               stdout=trace, stderr=stderr, start_new_session=True)
                    try:
                        code = process.wait(timeout=max(deadline - started, 0))
                    except subprocess.TimeoutExpired as error:
                        expired = error
                    finally:
                        # Includes the SDK's CLI and its MCP subprocess, even on timeout.
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                if expired is not None:
                    # The deadline covers every attempt's working time, so a timeout is never retried.
                    self._record(attempt, paths, None, started, reviewer_model=current, terminal_reason="timeout")
                    raise ValueError(f"Review agent exceeded {timeout}s on attempt {attempt}; "
                                     "transcripts preserved") from expired
                record, final, calls = self._record(attempt, paths, code, started, reviewer_model=current)
                if any(call.get("exhausted") for call in calls):
                    raise ValueError(f"Review evidence budget exhausted on attempt {attempt}; transcript preserved")
                if final:
                    break
                failure = self._failure(code)
                if not record["retryable"] or attempt == max_attempts:
                    raise ValueError(failure)
                if record["refused"]:
                    record["retryable"] = False
                    raise ValueError(f"{failure}; a safeguard refusal is final and is not retried on another model")
                if record["api_error"]:
                    wait = min(API_RETRY_CAP_SECONDS, backoff * 2 ** (attempt - 1))
                else:  # a crash or missing verdict
                    wait = min(backoff, QUICK_RETRY_SECONDS) * attempt
                if deadline - _now() < MIN_RETRY_SECONDS:
                    raise ValueError(f"{failure}; no time left in the {timeout}s review deadline to retry")
                _sleep(wait)
                deadline += wait  # waiting out an outage does not use the review's working time
            self.metadata.update({k: v for k, v in final.items() if k not in {"type", "verdict"}})
            if final["verdict"].get("decision") == "pass" and not any(
                    call.get("ok") and call.get("tool") == "read_evidence" for call in calls):
                raise ValueError("Review returned pass without reading evidence")
            return {"model": final.get("response_model"), "usage": final.get("usage"),
                    "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(final["verdict"])}}]}

    def _record(self, attempt, paths, code, started, **overrides):
        """Summarize an attempt's transcript and tool log into the receipt metadata."""
        def records(path):
            # A killed session may leave a partial final line; every complete record still counts.
            rows = []
            for line in path.read_text().splitlines() if path.exists() else []:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
            return rows
        events, calls = records(paths["transcript"]), records(paths["tool_log"])
        final = next((e for e in events if e.get("type") == "review_result"), None)
        record = {"attempt": attempt, **{k: v.name for k, v in paths.items()},
                  "transcript_sha256": hashlib.sha256(paths["transcript"].read_bytes()).hexdigest(),
                  "tool_calls": len(calls), "duration_seconds": round(_now() - started, 3),
                  **attempt_outcome(events, code), **overrides}
        if overrides.get("terminal_reason"):
            record["retryable"] = False
        attempts = self.metadata["attempts"] + [record]
        self.metadata.update(attempts=attempts, attempt_count=attempt,
                             terminal_reasons=[r["terminal_reason"] for r in attempts],
                             transcript=record["transcript"], tool_log=record["tool_log"], stderr=record["stderr"],
                             transcript_sha256=record["transcript_sha256"], tool_calls=record["tool_calls"])
        # Only the final attempt's failure is the review's error; earlier ones stay in attempts.
        self.metadata.pop("error", None)
        if record.get("error") and final is None:
            self.metadata["error"] = record["error"]
        return record, final, calls

    def _failure(self, code):
        attempts = self.metadata["attempts"]
        summary = "; ".join(
            f"attempt {r['attempt']} ({r.get('reviewer_model')}): {r['terminal_reason']}"
            + (f"/{r['stop_reason']}" if r.get("stop_reason") else "")
            + f" (exit {r['exit_code']}, {r['transcript']})" for r in attempts)
        return (f"Review SDK failed (exit {code}) after {len(attempts)} attempt{'s' if len(attempts) != 1 else ''}: "
                f"{summary}; see the transcripts and .stderr.txt files")
