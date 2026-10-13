"""Agent self-check: the grading path for one submission on public inputs, without the held-out split or review.

`frontier selfcheck` in the agent sandbox asks the Harbor plugin (selfcheck_service.py). The plugin copies
/workspace/output into the trial's submission volume and runs `python /tests/selfcheck.py` (this module's `main`) in a
sandbox built from the task's verifier image, with no private data and no review or judge credentials. The program
snapshots the copy exactly as grading does, then calls the task's hook, which launches the submission the way the
verifier launches it (same inference image, one H100, offline, the contract's startup deadline), sends real public
samples through the task's own response parser and measures the task's gates on public inputs. It never scores the
submission: the report says how far it got, the digest it checked, the GPU, the startup time against its deadline,
and every gate's measured value, requirement and margin.

Hook contract, both optional, in the task's verifier.py:

- ``selfcheck(check, public, config)``: `check` is a SelfCheck (below), `public` the directory holding the fetched
  public files, `config` the task's trusted grading config with self-check settings. A task without a hook gets a
  launch, the startup deadline and a second /health probe.
- ``SELFCHECK_PUBLIC``: glob patterns, relative to the task's public/ folder, of the files the hook reads; only
  those are fetched. Without it every public file is fetched; an empty tuple fetches nothing.

A task whose verifier.py must not change (svg-generation's bytes are bound by its judge calibration) names another
seed module holding both in task.yaml `verifier.selfcheck_module`. For `review: true` tasks the review's own free
prerequisites (architecture.json, serve.sh, the declared base) run before the hook: grading stops on them too.

Outcomes: `passed` (every recorded check passed), `failed` (a check failed; the submission's problem), `error` (the
self-check itself could not finish: provider, network or a bug here; the plugin does not count it against the
agent's allowance) and `cancelled`.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import json
import math
import numbers
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import time

from .runtime import submission_volume
from .runtime.inference import DEFAULT_STARTUP_SECONDS, TRANSFER_MARGIN_SECONDS, serve_submission
from .runtime.server import VerificationFailure, snapshot_submission
from .taskspec import load_task, served_by_verifier
from .validation.errors import EvaluationError
from .validation.evaluation import RequestFailure, attempt_request
from .validation.progress import progress, write_json

SCHEMA_VERSION = 1
# Beyond the startup deadline: one sample, the gates and teardown. Tasks with longer gates set
# selfcheck_timeout_seconds in their verifier config.
WORK_SECONDS = 900
DETAIL_CHARACTERS = 2000
LOG_CHARACTERS = 3000
MAX_CHECKS = 60
TIGHT_MARGIN = 0.15  # a gate passed with less to spare than this may fail on the grading run's hardware or inputs
# The plugin delivers the report into the agent sandbox as one base64 exec argument (Linux caps one at 128 KiB).
MAX_REPORT_BYTES = 64 * 1024
NOTES = ("A self-check reports no score: it sends public inputs, never the held-out split.",
         "Gates were measured on public inputs; held-out requests and grading hardware differ, so keep a margin.")
REVIEW_NOTE = "The architecture review did not run; it can still score this submission zero."


class CheckFailed(VerificationFailure):
    """A failure the self-check has already recorded; it ends the self-check."""


def clip(text, limit=DETAIL_CHARACTERS):
    text = str(text)
    return text if len(text) <= limit else text[:limit - 15] + " ...[clipped]"


def excerpt(value, limit=300):
    try:
        text = json.dumps(value, allow_nan=False, default=str)
    except ValueError:
        text = repr(value)
    return clip(text, limit)


def plain(value):
    """A number as a Python int or float. Hooks often measure with numpy, whose scalars (np.float32, np.int64) are not
    float or int, so rounding, the margin and the non-finite rule would skip them; numpy registers them as numbers."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return value
    return int(value) if isinstance(value, numbers.Integral) else float(value)


def selfcheck_seconds(config):
    """How long one self-check may hold its sandboxes after provisioning: the startup deadline plus the work budget."""
    startup = config.get("startup_timeout_seconds", DEFAULT_STARTUP_SECONDS)
    startup = startup if type(startup) is int and startup > 0 else DEFAULT_STARTUP_SECONDS
    work = config.get("selfcheck_timeout_seconds")
    work = work if type(work) is int and work > 0 else WORK_SECONDS
    return startup + work


class SelfCheck:
    """What a task's selfcheck hook records and launches. Every check lands in the agent's report in order."""

    def __init__(self, config, snapshot, *, launcher=None):
        self.config = config
        self.snapshot = Path(snapshot)
        # Tests and local runs replace every launcher with their own, the task's custom ones included.
        self.override = launcher
        self.results = []
        self.instances = []
        self.stage_name = "static"
        self.log_tail = None
        self.notes = []

    def stage(self, name):
        self.stage_name = name
        progress(self.config, name)

    def record(self, name, passed, *, measured=None, required=None, unit=None, higher_is_better=None,
               detail=None, kind="gate"):
        """One named check. With `measured`, `required` and `higher_is_better`, `margin` is the fraction of the
        requirement to spare (0.2: 20% better than required; negative: short by that much)."""
        entry = {"name": str(name), "kind": kind, "passed": bool(passed)}
        measured, required = plain(measured), plain(required)
        if isinstance(measured, float) and not math.isfinite(measured):
            # JSON has no NaN; a non-finite measurement never meets a requirement.
            entry["passed"] = False
            detail = f"non-finite measurement ({measured}); " + (detail or "")
            measured, higher_is_better = None, None
        if measured is not None:
            entry["measured"] = round(measured, 4) if isinstance(measured, float) else measured
        if required is not None:
            entry["required"] = required
        if unit:
            entry["unit"] = unit
        if (higher_is_better is not None and isinstance(measured, (int, float)) and isinstance(required, (int, float))
                and required and math.isfinite(measured)):
            margin = (measured - required) / required if higher_is_better else (required - measured) / required
            entry["margin"] = round(margin, 4)
        if detail:
            entry["detail"] = clip(detail)
        if len(self.results) < MAX_CHECKS:
            self.results.append(entry)
        elif not entry["passed"]:  # the outcome is read from these entries, so a failure past the cap must show
            self.results[-1] = entry
        return entry

    def note(self, text):
        """A sentence for the report's notes and summary: what this task's self-check did not cover, for example."""
        if len(self.notes) < 10:
            self.notes.append(clip(text, 500))

    def warn(self, name, detail, **measurement):
        """Record something grading does not fail the run for but that costs score: the self-check still passes, the
        agent's report marks the entry WARN and the summary line carries it."""
        entry = self.record(name, True, detail=detail, kind="warning", **measurement)
        entry["warning"] = True
        return entry

    def fail(self, name, detail, *, kind="gate", **measurement):
        """Record a failed check and end the self-check, as the same failure ends grading."""
        self.record(name, False, detail=detail, kind=kind, **measurement)
        raise CheckFailed(f"{name}: {detail}")

    def static(self, name, function, *args, **kwargs):
        """Run one of the task's static (no GPU) constraints and record it; a failure ends the check, as in grading.
        Static checks read the submission's files, so any exception they raise (a malformed header, a NUL byte in a
        path, absurdly nested JSON) counts against the submission; only EvaluationError is the self-check's own."""
        self.stage("static")
        try:
            value = function(*args, **kwargs)
        except (CheckFailed, EvaluationError):
            raise
        except VerificationFailure as error:
            self.fail(name, str(error), kind="static")
        except Exception as error:  # noqa: BLE001 - see above
            self.fail(name, f"{type(error).__name__}: {error}", kind="static")
        self.record(name, True, kind="static")
        return value

    @contextmanager
    def launch(self, launcher=None, **overrides):
        """A fresh offline instance of the submission, launched as grading launches it (the task's own launcher when
        given, `launcher(snapshot, config)`); records the startup gate, and on exit a second /health probe."""
        config = {**self.config, **overrides, "on_server_launch": lambda: self.stage("startup")}
        deadline = config.get("startup_timeout_seconds", DEFAULT_STARTUP_SECONDS)
        self.stage("provisioning")
        try:
            session = (self.override or launcher or serve_submission)(self.snapshot, config)
            server = session.__enter__()
        except (CheckFailed, EvaluationError):
            raise
        except VerificationFailure as error:
            self.fail("startup", str(error), required=deadline, unit="s")
        checked = False
        try:
            with ExitStack() as stack:
                stack.push(session)  # entered above; torn down with the body's exception, if any
                stack.callback(self._keep_log, server)  # before teardown, while the log still exists
                self.instances.append({"startup_seconds": server.get("startup_elapsed"), "gpu": server.get("gpu"),
                                       "backend": server.get("backend"), "channel": server.get("channel"),
                                       "network_isolation": server.get("network_isolation")})
                startup = server.get("startup_elapsed")
                if isinstance(startup, (int, float)):
                    self.record("startup", startup <= deadline, measured=round(startup, 1), required=deadline,
                                unit="s", higher_is_better=False, detail=f"GET /health answered 200 after {startup:.1f} s")
                self.stage("smoke")
                yield server
                self._still_healthy(server)
                checked = True
        except VerificationFailure as error:
            if checked and not isinstance(error, (CheckFailed, EvaluationError)):  # teardown: the relay's monitor
                self.fail("offline", str(error))
            raise

    def _still_healthy(self, server):
        """GET /health once more; a launcher whose channel cannot carry it supplies `health() -> (status, body)`."""
        try:
            probe = server.get("health")
            status, body = probe() if callable(probe) else server["transport"].raw_request("GET", "/health", timeout=10)
        except EvaluationError:
            raise
        except Exception as error:  # noqa: BLE001 - the candidate's server went away
            self.fail("healthy after requests", f"GET /health failed after the checks: {type(error).__name__}: {error}")
        if status != 200:
            self.fail("healthy after requests", f"GET /health answered HTTP {status} after the checks: {body[:200]!r}")
        self.record("healthy after requests", True, kind="smoke")

    def _keep_log(self, server):
        tail = server.get("server_log_tail") or server.get("server_log")
        if callable(tail):
            try:
                self.log_tail = clip(tail(LOG_CHARACTERS), LOG_CHARACTERS + 20)
            except Exception:  # noqa: BLE001 - diagnostics only
                pass

    def request(self, server, path, payload=None, *, method="POST", name=None, parse=None, sample=None,
                timeout=120, retry=None, **kwargs):
        """One real request through the grading transport, checked with the task's own parser when given
        (`parse(sample, response)`); records it and returns the parsed response. Any failure ends the check.

        Transport failures are retried as grading retries them (evaluation.attempt_request: two more tries after 1 s
        and 3 s, within `timeout`) unless the task's request_failure_policy allows no retries or `retry` is False."""
        name = name or f"{method} {path}"
        if retry is None:
            retry = self.config.get("request_failure_policy", "shared") in ("shared", "per_item")
        started = time.monotonic()
        try:
            if retry:
                response = attempt_request(server["transport"], method, path, payload, timeout=timeout, **kwargs)
                if isinstance(response, RequestFailure):
                    raise RuntimeError(f"{response.reason}: {response.error}")
            else:
                response = server["transport"].request(method, path, payload, timeout=timeout, **kwargs)
        except EvaluationError:
            raise
        except Exception as error:  # noqa: BLE001 - transport errors, timeouts, non-200 or non-JSON answers
            self.fail(name, f"{type(error).__name__}: {error}", kind="smoke")
        elapsed = time.monotonic() - started
        if parse is not None:
            try:
                response = parse(sample, response)
            except EvaluationError:
                raise
            except Exception as error:  # noqa: BLE001 - the grader would reject this answer
                self.fail(name, f"the response was rejected: {type(error).__name__}: {error}; got {excerpt(response)}",
                          kind="smoke")
        self.record(name, True, kind="smoke", measured=round(elapsed, 3), unit="s", detail=excerpt(response))
        return response

    def gate(self, name, measured, required, *, higher_is_better, unit=None, detail=None):
        """A gate with a measured value; passes when the measurement meets the requirement."""
        passed = measured >= required if higher_is_better else measured <= required
        return self.record(name, passed, measured=measured, required=required, unit=unit,
                           higher_is_better=higher_is_better, detail=detail)


def default_selfcheck(check, public, config):
    """A task without a hook: launch, wait for /health within the deadline, probe /health again."""
    with check.launch():
        pass


def fetch_public(tests_dir, patterns, destination):
    """The task's public files named by `patterns` (None: all), from the pinned dataset the agent's copy came from."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if patterns is not None and not patterns:
        return destination
    if (tests_dir / "public").is_dir():  # tasks that prepare their public data at build time ship it here
        return tests_dir / "public"
    from spin.transfer import fetch_hf
    source = json.loads((tests_dir / "public.json").read_text())
    if patterns is not None:
        source["include"] = list(patterns)
    try:
        fetch_hf(source, destination, token=os.environ.get("HF_TOKEN") or None)
    except Exception as error:
        # The message names the dataset repository, which also holds the held-out split: operator logs only.
        print(f"Public data fetch failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise EvaluationError(f"Could not fetch the task's public files ({type(error).__name__})") from None
    return destination


def review_preconditions(task, snapshot):
    """What the review rejects before any reviewer call (review.collect_evidence), which ends grading as an incomplete
    review: architecture.json and serve.sh at the top of the submission, architecture.json a JSON object, and the
    base it declares allowed by the task's review policy."""
    from .review import EvidenceError, check_declared_base
    from .taskspec import review_policy
    snapshot = Path(snapshot)
    policy = review_policy(task.slug, task.path)
    if policy.get("served_by_verifier") is True:  # the task serves the weights itself: no serve.sh is submitted
        if not (snapshot / "architecture.json").is_file():
            raise VerificationFailure("The review needs architecture.json at the top of the submission")
    elif not (snapshot / "architecture.json").is_file() or not (snapshot / "serve.sh").is_file():
        raise VerificationFailure("The review needs architecture.json and serve.sh at the top of the submission")
    try:
        metadata = json.loads((snapshot / "architecture.json").read_text())
    except (ValueError, RecursionError) as error:
        raise VerificationFailure(f"architecture.json is not valid JSON: {error}") from None
    if not isinstance(metadata, dict):
        raise VerificationFailure("architecture.json must be a JSON object")
    try:
        check_declared_base(metadata, policy)
    except EvidenceError as error:
        raise VerificationFailure(str(error)) from None


def hook_module(task):
    """The module holding the task's selfcheck hook and SELFCHECK_PUBLIC: verifier.py, or the seed module task.yaml
    names in verifier.selfcheck_module."""
    name = (task.spec.get("verifier") or {}).get("selfcheck_module")
    if not name:
        return task.module
    from .taskspec import load_module
    if not isinstance(name, str) or "/" in name or not name.endswith(".py"):
        raise ValueError("verifier.selfcheck_module must name a Python file in the task seed")
    return load_module(f"{task.slug}.selfcheck", task.path / name)


def summary(report):
    """One line for the agent: the outcome, the startup margin and the failing check."""
    checks = report.get("checks") or []
    failed = [c for c in checks if not c["passed"]]
    startup = next((c for c in checks if c["name"] == "startup" and "measured" in c), None)
    parts = [report["status"]]
    if startup:
        parts.append(f"startup {startup['measured']} s of {startup['required']} s")
    gates = [c for c in checks if c["kind"] == "gate" and c["name"] != "startup"]
    if gates:
        parts.append(f"{sum(c['passed'] for c in gates)}/{len(gates)} gates passed")
    if failed:
        parts.append(f"failed {failed[0]['name']}: {failed[0].get('detail', '')}"[:400])
    elif report.get("error"):
        parts.append(report["error"][:400])
    for warning in [c for c in checks if c.get("warning")][:2]:
        parts.append(f"WARNING {warning['name']}: {warning.get('detail', '')}"[:400])
    for note in report.get("hook_notes") or []:
        parts.append(note[:300])
    return "; ".join(parts)


def compact(report):
    """Replace non-finite numbers (JSON has none) and values JSON cannot write, then shrink details and the log until
    the report fits MAX_REPORT_BYTES."""
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: finite(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite(item) for item in value]
        return value

    # A value JSON cannot write (numpy outside record's measured and required) would make write_json drop the whole
    # report; as text it is at least shown.
    report.update(json.loads(json.dumps(finite(report), allow_nan=False, default=str)))

    def size():
        return len(json.dumps(report, allow_nan=False, default=str).encode())
    for limit in (500, 200, 80):
        if size() <= MAX_REPORT_BYTES:
            return report
        for entry in report.get("checks") or []:
            if "detail" in entry:
                entry["detail"] = clip(entry["detail"], limit)
        if report.get("server_log_tail"):
            report["server_log_tail"] = report["server_log_tail"][-limit * 4:]
    if size() > MAX_REPORT_BYTES:
        report["checks"] = (report.get("checks") or [])[:20]
    return report


def run(slug, *, tests_dir=Path("/tests"), results_dir, archive=None, submission=None, public=None,
        check_id=None, launcher=None):
    """Self-check one submission copy and return the report. `archive` is the copy's path inside the trial's
    submission volume (the plugin's case); `submission` a local directory (tests and operator use)."""
    tests_dir, results_dir = Path(tests_dir), Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"schema_version": SCHEMA_VERSION, "task": slug, "check_id": check_id, "status": None,
              "stage": "setup", "notes": list(NOTES)}
    config = {"progress_path": str(results_dir / "progress.json")}
    check = None
    progress(config, "setup")
    try:
        task = load_task(slug, tests_dir / "task")
        module = hook_module(task)
        if task.spec.get("review"):
            report["notes"].append(REVIEW_NOTE)
        config = json.loads((tests_dir / "config.json").read_text())
        for key in ("architecture_review_path", "submission_volume", "submission_digest", "submission_archive",
                    "review_trajectory", "speed_prompts"):
            config.pop(key, None)
        # task.yaml holds request_failure_policy, not the verifier config: SelfCheck.request retries only "shared".
        config.update(selfcheck=True, validation_mode=False, predictions_only=False, task_slug=slug,
                      results_dir=str(results_dir), progress_path=str(results_dir / "progress.json"),
                      request_failure_policy=task.spec.get("request_failure_policy", "shared"))
        # The GPU sandbox lives for the startup deadline and the work budget, plus the copy-and-digest margin.
        lifetime = selfcheck_seconds(config) + TRANSFER_MARGIN_SECONDS
        configured = config.get("verifier_lifetime_seconds")
        config["verifier_lifetime_seconds"] = min(configured, lifetime) if type(configured) is int else lifetime
        with tempfile.TemporaryDirectory(prefix="frontier-selfcheck-") as directory:
            directory = Path(directory)
            public = Path(public) if public is not None else fetch_public(
                tests_dir, getattr(module, "SELFCHECK_PUBLIC", None), directory / "public")
            report["stage"] = "snapshot"
            progress(config, "snapshot")
            snapshot = directory / "submission"
            if archive is not None:
                name = os.environ.get(submission_volume.ENV)
                if not name:
                    raise EvaluationError(f"Self-checks of a volume copy need {submission_volume.ENV}")
                path = Path(submission_volume.MOUNT) / submission_volume.safe_archive(archive)
                if not path.is_file():
                    raise EvaluationError(f"The submission copy {archive} is not in volume {name}")
                config.update(submission_volume=name, submission_archive=archive)
            try:
                source = (submission_volume.extract(path, directory / "staging") if archive is not None
                          else Path(submission))
                digest = snapshot_submission(source, snapshot, require_serve_sh=not served_by_verifier(task))
            except VerificationFailure as error:
                report["checks"] = [{"name": "snapshot", "kind": "static", "passed": False, "detail": clip(error)}]
                for path, _, _ in os.walk(snapshot):
                    os.chmod(path, 0o700)
                raise CheckFailed(str(error)) from error
            finally:
                if archive is not None:
                    shutil.rmtree(directory / "staging", ignore_errors=True)
            files = [path for path in snapshot.rglob("*") if path.is_file()]
            report.update(submission_digest=digest, submission_files=len(files),
                          submission_bytes=sum(path.stat().st_size for path in files))
            config.update(submission_path=str(snapshot), submission_digest=digest)
            check = SelfCheck(config, snapshot, launcher=launcher)
            hook = getattr(module, "selfcheck", None)
            try:
                if task.spec.get("review"):
                    check.static("review prerequisites", review_preconditions, task, snapshot)
                (hook if callable(hook) else default_selfcheck)(check, public, config)
            finally:
                for path, _, _ in os.walk(snapshot):  # read-only snapshots must stay removable
                    os.chmod(path, 0o700)
            check.stage("done")
            failed = [entry for entry in check.results if not entry["passed"]]
            report["status"] = "failed" if failed else "passed"
    except CheckFailed:
        report["status"] = "failed"
    except EvaluationError as error:  # the verifier's own infrastructure; never the submission's fault
        report["status"] = "error"
        report["error"] = clip(f"The self-check could not finish ({type(error).__name__}: {error}).")
    except VerificationFailure as error:
        report["status"] = "failed"
        if check is not None and not any(entry["name"] == check.stage_name for entry in check.results):
            check.record(check.stage_name, False, kind="smoke" if check.stage_name in ("smoke", "gates") else "static",
                         detail=str(error))
        report["error"] = clip(f"{type(error).__name__}: {error}")
    except (SystemExit, KeyboardInterrupt):
        report["status"] = "cancelled"
        report["error"] = "The self-check was cancelled before it finished."
    except Exception as error:  # noqa: BLE001 - provider, network or self-check faults: not the submission's
        report["status"] = "error"
        report["error"] = clip(f"The self-check could not finish ({type(error).__name__}: {error}).")
    if check is not None:
        report["stage"] = check.stage_name
        report["checks"] = check.results
        report["instances"] = check.instances
        if check.notes:
            report["hook_notes"] = check.notes
            report["notes"] = check.notes + report["notes"]
        if check.log_tail:
            report["server_log_tail"] = check.log_tail
        tight = [entry["name"] for entry in check.results if entry.get("passed")
                 and isinstance(entry.get("margin"), (int, float)) and entry["margin"] < TIGHT_MARGIN]
        if tight:
            report["notes"].append("Passed with less than 15% to spare, which may not hold on the grading run: "
                                   + ", ".join(tight) + ".")
    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    report["summary"] = summary(report)
    compact(report)
    progress(config, report["status"], message=report.get("error"))
    write_json(results_dir / "report.json", report)
    return report


def main(slug, argv=None):
    parser = argparse.ArgumentParser(prog="selfcheck.py", description="Self-check a submission copy on public inputs")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", help="the copy's path inside the trial's submission volume")
    source.add_argument("--submission", type=Path, help="a local submission directory")
    parser.add_argument("--out", type=Path, required=True, help="directory for progress.json and report.json")
    parser.add_argument("--check-id")
    parser.add_argument("--tests-dir", type=Path, default=Path("/tests"))
    parser.add_argument("--public", type=Path, help="already fetched public files")
    args = parser.parse_args(argv)

    def cancel(signum, frame):
        raise SystemExit(128 + signum)  # unwinds the sandboxes' context managers, which stop them

    signal.signal(signal.SIGTERM, cancel)
    report = run(slug, tests_dir=args.tests_dir, results_dir=args.out, archive=args.archive,
                 submission=args.submission, public=args.public, check_id=args.check_id)
    print(json.dumps({"status": report["status"], "summary": report["summary"]}), flush=True)
    return {"passed": 0, "failed": 1}.get(report["status"], 2)
