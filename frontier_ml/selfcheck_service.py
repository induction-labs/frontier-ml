"""Host side of `frontier selfcheck`: the Harbor plugin services one agent sandbox's self-check requests.

The agent sandbox takes no inbound connections and must never hold Modal or grading credentials, so the channel is a
spool directory inside it (SPOOL), serviced through root execs from the Harbor process, which already holds the
Modal session, the rendered task and the trial's submission volume:

- SPOOL/requests/ is the only directory the agent can write. `frontier selfcheck` drops `<nonce>.json` there; one
  long-lived root watcher deletes it and reports the nonce. Its contents are never read.
- SPOOL/status.json (root-owned, like everything else under SPOOL) holds the allowance, the running check and
  every check's outcome; SPOOL/checks/<id>/ the check's progress.json and, at the end, its report.json.

A check copies /workspace/output into the submission volume (submission_volume.SELFCHECK_COPY_SCRIPT, without
stopping the agent), then runs `python /tests/selfcheck.py` (selfcheck.py) in a sandbox built from the task's
verifier image, given only Modal and Hugging Face credentials: no private data, review or judge keys. One check runs
at a time; `limit` are allowed per trial attempt. A check that ends in `error` (the provider's or the self-check's
fault, not the submission's) is not counted, up to `limit` such refunds. When the agent's session ends the running
check is stopped (its GPU sandbox with it) before the final copy of /workspace/output is taken.

The same watcher carries `frontier submit --confirm`: a request named `submit-<nonce>.json` (SUBMIT_PREFIX) is the
agent's submission. The service answers it in status.json (`submitted_at`), marks the session in submissions.py so
that no harness resumes it, and SUBMIT_GRACE_SECONDS later kills every agent process once, which ends the harness's
command; Harbor then ends the agent phase and the plugin freezes /workspace/output as at any other session end. The
service runs for every Modal trial, also with `limit` 0 (self-checks off), so that the agent can always submit.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from pathlib import Path
import re
import shlex
import time
import uuid

from . import submissions
from .runtime import submission_volume
from .runtime.inference import TRANSFER_MARGIN_SECONDS
from .runtime.resources import START_TIMEOUT_SECONDS
from .validation.progress import write_json

logger = logging.getLogger(__name__)

SPOOL = "/var/lib/frontier/selfcheck"
DEFAULT_LIMIT = 4
AGENT_USER = "actor"
SCHEMA_VERSION = 1
COPY_TIMEOUT_SECONDS = 1800
PUBLIC_FETCH_SECONDS = 900
PROGRESS_SECONDS = 15
CANCEL_GRACE_SECONDS = 120   # SIGTERM unwinds the GPU sandbox before the self-check sandbox is deleted
DELIVER_SECONDS = 150
HEAL_SECONDS = 15   # a write that failed is tried again this often (well within the CLI's 60 s pickup), then less
WATCH_RETRY_SECONDS = 1
# status.json travels as one exec argument (128 KiB at most): the latest request answers and checks, summaries clipped.
MAX_REQUESTS = 100
MAX_CHECKS = 10
MAX_FREE = 10   # checks that stop before any GPU (snapshot, static constraints) are not counted, up to this many
PRE_GPU_STAGES = ("copying", "setup", "snapshot", "static")
SUMMARY_CHARS = 300
STAGE_CHARS = 100
EXEC_OUTPUT = "/tmp"
STATUSES = ("passed", "failed", "error", "cancelled")
NONCE = re.compile(r"[A-Za-z0-9_-]{1,64}")
SUBMIT_PREFIX = "submit-"  # agent_cli.SUBMIT_PREFIX
# Between answering a submission and killing the agent's processes: `frontier submit --confirm` reads the answer
# within a second or two and prints it, so the acceptance reaches the agent's transcript.
SUBMIT_GRACE_SECONDS = 10
STOP_ATTEMPTS = 3
CONTROLLER_LOGS = "/logs/selfcheck"
# What the self-check sandbox may receive from the Harbor host: provider access for the GPU sandbox and the public
# data fetch. Never review, judge or model API keys.
CONTROLLER_ENV = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_ENVIRONMENT", "HF_TOKEN", "FML_GRADING_RESOURCES",
                  "FML_GRADING_CREDENTIALS_SECRET", "FML_INFERENCE_GPU_TYPE", "FML_GRADING_RELAY",
                  "FML_MODAL_GRADING_NETWORK")

SETUP_SCRIPT = f"""set -e
mkdir -p {SPOOL}/requests {SPOOL}/checks
chown root:root /var/lib/frontier {SPOOL} {SPOOL}/checks
chmod 755 /var/lib/frontier {SPOOL} {SPOOL}/checks
chown {AGENT_USER}:{AGENT_USER} {SPOOL}/requests
chmod 700 {SPOOL}/requests
"""

# Request files are deleted unread; only a well-formed name is reported. A watcher whose exec stream dropped keeps
# running in the sandbox, so each new one claims watcher.pid and an older one exits instead of taking requests.
WATCH_SCRIPT = f"""set -u
cd {SPOOL}/requests || exit 3
echo $$ > {SPOOL}/watcher.pid
echo READY
while [ "$(cat {SPOOL}/watcher.pid 2>/dev/null)" = "$$" ]; do
  for f in *.json; do
    [ -e "$f" ] || [ -L "$f" ] || continue
    name=${{f%.json}}
    rm -rf -- "$f" || continue
    case "$name" in ''|*[!A-Za-z0-9_-]*) continue ;; esac
    [ ${{#name}} -le 64 ] && echo "REQUEST $name"
  done
  sleep 1
done
"""

# The temporary name is per exec: a write abandoned on timeout may still be running beside the next one.
WRITE_SCRIPT = """set -e
umask 022
mkdir -p "$(dirname "$1")"
printf %s "$2" | base64 -d > "$1.$$.tmp"
mv -f "$1.$$.tmp" "$1"
"""

# selfcheck.py unwinds its GPU sandbox on SIGTERM. The verifier image has no procps, and Harbor runs root execs as
# `su root -c <command>`: the shells and su carry the path in their command lines too, and su answers SIGTERM with a
# SIGKILL for its child 2 s later. So only processes that have the path as an argument of their own are signalled.
TERMINATE_SCRIPT = """import os, signal, time
def running():
    pids = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as stream:
                if b"/tests/selfcheck.py" in stream.read().split(b"\\0"):
                    pids.append(int(pid))
        except OSError:
            pass
    return pids
for pid in running():
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass
deadline = time.monotonic() + 100
while running() and time.monotonic() < deadline:
    time.sleep(1)
"""
TERMINATE_COMMAND = f"python -c {shlex.quote(TERMINATE_SCRIPT)}"


def saving(script, directory):
    """`script` with its output also kept in `directory`: Modal can lose an exec's stdio stream after the command
    completed (runtime/modal_environment.py), and a write or a copy must not be redone for that. Only root-owned
    directories are cleaned up, an hour on (the longest command is the 30-minute copy)."""
    return (f"find {EXEC_OUTPUT} -maxdepth 1 -name 'frontier-selfcheck-*' -type d -user root -mmin +60 "
            "-exec rm -rf {} + 2>/dev/null\n"
            f"mkdir -m 700 {directory} || exit 125\n"
            f"(\n{script}\n) >{directory}/stdout 2>{directory}/stderr\n"
            "_frontier_status=$?\n"
            f"cat {directory}/stdout\ncat {directory}/stderr >&2\n"
            'exit "$_frontier_status"\n')


def clipped(value, limit):
    return None if value is None else str(value)[:limit]


def controller_command(check_id):
    archive = submission_volume.selfcheck_archive(check_id)
    return (f"cd /tmp && PYTHONPATH=/tests/lib FML_GRADING_RESOURCE_LOG_DIR={CONTROLLER_LOGS}/{check_id}/resources "
            f"python /tests/selfcheck.py --archive {archive} --check-id {check_id} --out {CONTROLLER_LOGS}/{check_id}")


def controller_credentials():
    environment = {name: os.environ[name] for name in CONTROLLER_ENV if os.environ.get(name)}
    try:  # the Harbor host may read these from ~/.modal.toml; the volume lookup needs its Modal environment too
        from modal.config import config
        for name, key in (("MODAL_TOKEN_ID", "token_id"), ("MODAL_TOKEN_SECRET", "token_secret"),
                          ("MODAL_ENVIRONMENT", "environment")):
            if not environment.get(name) and config.get(key):
                environment[name] = config.get(key)
    except Exception:  # noqa: BLE001 - the self-check then reports the missing credential itself
        pass
    return environment


def budget_seconds(grading_config):
    """How long a check may run in its sandbox: the GPU's start allowance, the public fetch, the copy-and-digest
    margin, the startup deadline and the work budget (selfcheck.selfcheck_seconds)."""
    from .selfcheck import selfcheck_seconds
    return START_TIMEOUT_SECONDS + PUBLIC_FETCH_SECONDS + TRANSFER_MARGIN_SECONDS + selfcheck_seconds(grading_config)


class ModalAgentSandbox:
    """Root execs in the trial's agent sandbox (Harbor tags it `<trial>__env`)."""

    def __init__(self, session_id):
        self.session_id = session_id
        self._sandbox = None

    async def sandbox(self):
        if self._sandbox is None:
            self._sandbox = await submission_volume.agent_sandbox(self.session_id)
        return self._sandbox

    async def run(self, script, *args, env=None, timeout=120):
        """Run `script` as root: its exit status, stdout and stderr, read back from the saved copy if the stream is
        lost. The provider's exit status stays authoritative."""
        from .runtime.modal_environment import _control_call
        sandbox = await self.sandbox()
        directory = f"{EXEC_OUTPUT}/frontier-selfcheck-{uuid.uuid4().hex}"
        # Never retried: an ambiguous error here may mean the command did start.
        process = await sandbox.exec.aio("bash", "-c", saving(script, directory), "selfcheck", *args, env=env,
                                         timeout=timeout)
        try:
            output, lost = [await process.stdout.read.aio(), await process.stderr.read.aio()], None
        except Exception as exc:  # noqa: BLE001
            lost = exc
        code = await _control_call(process.wait.aio)
        if lost is not None:
            logger.warning("An agent sandbox exec lost its output stream (%s); reading the saved copy",
                           type(lost).__name__)
            read = sandbox.filesystem.read_text.aio
            output = [await _control_call(lambda path=f"{directory}/{name}": read(path), timeout=30)
                      for name in ("stdout", "stderr")]
        return code, *output

    async def lines(self, script):
        sandbox = await self.sandbox()
        process = await sandbox.exec.aio("bash", "-c", script, bufsize=1)
        async for chunk in process.stdout:
            for line in chunk.splitlines():
                yield line.strip()


class HarborController:
    """The self-check sandbox: the task's verifier image (tests/ build context, which Modal caches with the verifier's
    own build), public networking, the trial's submission volume, and nothing else from the trial."""

    def __init__(self, trial_config, directory, name, lifetime):
        self.trial_config, self.directory, self.name, self.lifetime = trial_config, Path(directory), name, lifetime
        self.environment = None

    async def start(self):
        from harbor.environments.factory import EnvironmentFactory
        from harbor.models.task.config import NetworkPolicy, TaskConfig
        from harbor.models.task.verifier_mode import resolve_effective_verifier_env_config
        from harbor.models.trial.paths import TrialPaths
        task_dir = Path(self.trial_config.task.path)
        task_config = TaskConfig.model_validate_toml((task_dir / "task.toml").read_text())
        settings = resolve_effective_verifier_env_config(task_config, None)
        runtime = self.trial_config.environment.model_copy(deep=True)
        runtime.env = {}  # the agent's wallet credentials stay with the agent
        runtime.kwargs = {**runtime.kwargs, "sandbox_timeout_secs": self.lifetime}  # keeps the submission volume
        self.directory.mkdir(parents=True, exist_ok=True)
        self.environment = EnvironmentFactory.create_environment_from_config(
            config=runtime, environment_dir=task_dir / "tests", environment_name=task_dir.name,
            session_id=self.name, trial_paths=TrialPaths(self.directory), task_env_config=settings,
            network_policy=NetworkPolicy(network_mode="public"), mounts=[])
        await self.environment.start(force_build=False)

    async def exec(self, command, *, env=None, timeout=60):
        return await self.environment.exec(command, env=env, timeout_sec=timeout, user="root")

    async def stop(self):
        if self.environment is not None:
            await self.environment.stop(delete=True)


class TrialSelfChecks:
    """One trial attempt's self-checks and submission: the watcher, the allowance, at most one running check, and
    the agent's `frontier submit`. `session` is the agent environment's Harbor session ID (`<trial>__env`), marked in
    submissions.py when the agent submits."""

    def __init__(self, *, sandbox, controller, trial_dir, volume, grading_config, limit=DEFAULT_LIMIT,
                 credentials=controller_credentials, session=None):
        if type(limit) is not int or limit < 0:
            raise ValueError("limit must be a non-negative integer")
        self.sandbox, self.controller, self.volume = sandbox, controller, volume
        self.directory = Path(trial_dir) / "selfcheck"
        self.grading_config, self.credentials = grading_config, credentials
        self.limit, self.session = limit, session
        from .selfcheck import selfcheck_seconds
        # What one check of this task may hold the GPU for (startup deadline + work budget), for the agent's CLI.
        minutes = -(-selfcheck_seconds(grading_config) // 60)
        self.state = {"schema_version": SCHEMA_VERSION, "available": limit > 0, "limit": limit, "used": 0,
                      "refunds": 0, "free": 0, "check_minutes": minutes, "running": None, "requests": {},
                      "checks": [], "submission_open": True, "submitted_at": None}
        if not limit:
            self.state["closed_reason"] = "Self-checks are off in this run."
        self.watcher = self.healer = None
        self.ending = None    # after a submission: the wait, then the kill of the agent's processes
        self.work = None      # the running check's copy and controller run; stop() cancels it
        self.running = None   # the running check as a whole: its work, then teardown and report, never cancelled
        self.undelivered = {}   # check ID -> its report, until written into the agent sandbox
        self.stale = False      # the agent sandbox's status.json may lack the latest change
        self.lock = asyncio.Lock()      # the allowance and the running check
        self.writing = asyncio.Lock()   # one write into the agent sandbox at a time, so the newest status lands last

    @property
    def remaining(self):
        return max(0, self.limit - self.state["used"])

    @property
    def submitted_at(self):
        """When the agent submitted (epoch seconds), or None."""
        return self.state["submitted_at"]

    async def start(self):
        code, _, stderr = await self.sandbox.run(SETUP_SCRIPT, timeout=60)
        if code:
            raise RuntimeError(f"Could not prepare {SPOOL} in the agent sandbox ({code}): {stderr[-500:]}")
        await self._publish()
        self.watcher = asyncio.create_task(self._watch())
        self.healer = asyncio.create_task(self._heal())

    async def stop(self, reason="The agent's session ended."):
        """Stop the watcher and any running check, and return once its sandboxes are gone; later requests find the
        service closed. A submission's pending kill is dropped: the session has ended without it."""
        # Before cancelling: the report gives the reason.
        self.state.update(available=False, closed_reason=reason, submission_open=False)
        # Everything is cancelled before the first wait, so a caller that cancels stop() leaves no check running.
        loops = [task for task in (self.watcher, self.healer, self.ending) if task is not None]
        for task in loops:
            task.cancel()
        running = self.running
        if self.work is not None:
            self.work.cancel()
        await asyncio.gather(*loops, return_exceptions=True)
        self.watcher = self.healer = self.ending = None
        if running is not None:
            # Only the work is cancelled: the teardown that stops the controller (and the GPU sandbox it started)
            # must finish before the final copy replaces what it reads. asyncio.wait never cancels it, even when
            # stop() itself is cancelled.
            await asyncio.wait({running})
            if not running.cancelled() and running.exception() is not None:
                logger.warning("Self-check teardown for %s failed: %s", self.volume, running.exception())
        try:
            await asyncio.wait_for(self._publish(), 30)
        except Exception:  # noqa: BLE001 - the agent sandbox may already be gone
            pass
        self._persist()

    async def _watch(self):
        delay = WATCH_RETRY_SECONDS
        while True:
            try:
                async for line in self.sandbox.lines(WATCH_SCRIPT):
                    delay = WATCH_RETRY_SECONDS
                    kind, _, nonce = line.partition(" ")
                    if kind == "REQUEST" and NONCE.fullmatch(nonce):
                        await (self.submit(nonce) if nonce.startswith(SUBMIT_PREFIX) else self.request(nonce))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a dropped exec stream: start another watcher
                logger.warning("Self-check watcher for %s stopped (%s: %s); restarting", self.volume,
                               type(exc).__name__, exc)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)

    async def _heal(self):
        """Write again what a failed write left out: a report the agent sandbox lacks (its CLI would follow a check
        that has finished) or a status.json that fell behind (a request's answer, a check's outcome)."""
        delay = HEAL_SECONDS
        while True:
            await asyncio.sleep(delay)
            if not (self.undelivered or self.stale):
                delay = HEAL_SECONDS
                continue
            try:
                await self._flush()
                delay = HEAL_SECONDS
            except Exception as exc:  # noqa: BLE001 - tried again later, less often
                logger.warning("Self-check results for %s still not in the agent sandbox: %s", self.volume, exc)
                delay = min(delay * 2, 600)

    async def request(self, nonce):
        """Start a check for this request, attach it to the running one, or reject it."""
        async with self.lock:
            requests = self.state["requests"]
            if nonce in requests:
                return
            if not self.state["available"]:
                requests[nonce] = {"rejected": "Self-checks are closed: " + self.state.get("closed_reason", "")}
            elif self.state["running"]:
                requests[nonce] = {"check": self.state["running"], "attached": True}
            elif self.remaining <= 0:
                requests[nonce] = {"rejected": f"All {self.limit} self-checks of this run have been used."}
            else:
                check_id = uuid.uuid4().hex
                entry = {"id": check_id, "request": nonce, "state": "running", "stage": "copying",
                         "submitted_at": time.time(), "counted": True}
                self.state["checks"].append(entry)
                self.state["used"] += 1
                self.state["running"] = check_id
                requests[nonce] = {"check": check_id}
                held = {}
                self.work = asyncio.create_task(self._check(entry, held))
                self.running = asyncio.create_task(self._run(entry, self.work, held))
            while len(requests) > MAX_REQUESTS:
                del requests[next(iter(requests))]
            try:
                await self._publish()
            except Exception as exc:  # noqa: BLE001 - _heal or the check's next stage publishes the answer again
                logger.warning("Could not publish self-check status for %s: %s", self.volume, exc)

    async def submit(self, nonce):
        """Accept the agent's submission (`frontier submit --confirm`) and end its session; a repeated request gets the
        first acceptance again."""
        async with self.lock:
            requests = self.state["requests"]
            if nonce in requests:
                return
            if self.state["submitted_at"] is not None:
                requests[nonce] = {"submitted_at": self.state["submitted_at"]}
            elif not self.state["submission_open"]:
                requests[nonce] = {"rejected": "Your session has ended, so nothing more can be submitted."}
            else:
                now = time.time()
                if self.session:  # before the kill: the harness's failed command is the submission, not a fault
                    submissions.mark(self.session)
                self.state.update(submitted_at=now, submission_open=False, available=False,
                                  closed_reason="The solution was submitted.")
                requests[nonce] = {"submitted_at": now}
                self.ending = asyncio.create_task(self._end_session())
                logger.info("The agent submitted in %s; stopping its processes in %d s", self.volume,
                            SUBMIT_GRACE_SECONDS)
            while len(requests) > MAX_REQUESTS:
                del requests[next(iter(requests))]
            try:
                await self._publish()
            except Exception as exc:  # noqa: BLE001 - _heal publishes the answer again
                logger.warning("Could not publish the submission for %s: %s", self.volume, exc)
            self._persist()

    async def _end_session(self):
        """Kill every agent process once `frontier submit --confirm` has had time to print the acceptance. The
        harness's command then fails and Harbor ends the agent phase. Never repeated once it succeeds: harnesses copy
        their session logs as the agent right after their command ends."""
        await asyncio.sleep(SUBMIT_GRACE_SECONDS)
        for attempt in range(1, STOP_ATTEMPTS + 1):
            try:
                code, _, stderr = await self.sandbox.run(submission_volume.STOP_AGENT_SCRIPT, timeout=60)
                if not code:
                    return
                failure = f"exit {code}: {stderr[-300:]}"
            except Exception as exc:  # noqa: BLE001 - a lost exec stream; the kill may still have happened
                failure = f"{type(exc).__name__}: {exc}"
            logger.warning("Stopping the agent after its submission in %s failed (attempt %d): %s", self.volume,
                           attempt, failure)
            if attempt < STOP_ATTEMPTS:
                await asyncio.sleep(SUBMIT_GRACE_SECONDS * attempt)
        logger.error("Could not stop the agent after its submission in %s; its session runs until it ends by itself",
                     self.volume)

    async def _run(self, entry, work, held):
        """Wait for the check's work, then always stop what it started and deliver its report."""
        try:
            report = await work
        except asyncio.CancelledError:  # stop(): the session ended
            report = self._report(entry, "cancelled", entry.get("stage"),
                                  self.state.get("closed_reason") or "The self-check was stopped.")
        except Exception as exc:  # noqa: BLE001 - provider or self-check faults are never the submission's
            report = self._report(entry, "error", entry.get("stage"),
                                  f"The self-check could not finish ({type(exc).__name__}: {str(exc)[:500]}).")
        job, controller = held.get("job"), held.get("controller")
        try:
            try:
                if job is not None:  # the local wait for selfcheck.py; cancelling it leaves the process running
                    job.cancel()
                    await asyncio.gather(job, return_exceptions=True)
            finally:
                if controller is not None:
                    await self._stop_controller(controller, entry["id"],
                                                terminate=report["status"] in ("error", "cancelled"))
        finally:
            await self._finish(entry, report)

    async def _check(self, entry, held):
        """Copy /workspace/output and run selfcheck.py on the copy; return its report. What it starts goes in `held`
        for _run to stop."""
        check_id = entry["id"]
        await self._stage(entry, "copying")
        code, stdout, stderr = await self.sandbox.run(submission_volume.SELFCHECK_COPY_SCRIPT,
                                                      env={"FML_SELFCHECK_ID": check_id},
                                                      timeout=COPY_TIMEOUT_SECONDS)
        if code == 3:
            return self._report(entry, "failed", "copying", "/workspace/output is not a directory.")
        if code == 2:  # tar, run as the agent, could not read a file of its own output
            return self._report(entry, "failed", "copying",
                                "Copying /workspace/output as the agent failed: " + clipped(stderr.strip(), 500))
        if code:
            raise RuntimeError(f"Copying /workspace/output exited {code}: {stderr[-500:]}")
        size, files, tar_status = (int(value) for value in stdout.split()[-3:])
        entry.update(copied_bytes=size, copied_files=files)
        if tar_status:
            entry["copy_note"] = "Files changed while /workspace/output was copied; the check used that copy."
        await self._stage(entry, "starting the self-check sandbox")
        lifetime = budget_seconds(self.grading_config)
        controller = held["controller"] = self.controller(check_id, lifetime)
        await asyncio.wait_for(controller.start(), START_TIMEOUT_SECONDS)
        await self._stage(entry, "setup")
        environment = {**self.credentials(), submission_volume.ENV: self.volume}
        job = held["job"] = asyncio.create_task(controller.exec(controller_command(check_id), env=environment,
                                                                timeout=lifetime))
        seen = None
        while not job.done():
            await asyncio.wait({job}, timeout=PROGRESS_SECONDS)
            if not job.done():
                seen = await self._mirror_progress(entry, controller, seen)
        output = job.result()
        # Operator diagnostics (a failed public fetch names the dataset); never shown to the agent.
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / f"{check_id}.controller.log").write_text(
            f"exit {getattr(output, 'return_code', None)}\n--- stdout\n{getattr(output, 'stdout', '') or ''}"
            f"\n--- stderr\n{getattr(output, 'stderr', '') or ''}")
        result = await controller.exec(f"cat {CONTROLLER_LOGS}/{check_id}/report.json", timeout=60)
        report = json.loads(result.stdout or "")
        if not isinstance(report, dict) or report.get("status") not in STATUSES:
            raise ValueError("The self-check sandbox wrote no valid report")
        return report

    async def _stop_controller(self, controller, check_id, *, terminate):
        try:
            if terminate:  # deleting the self-check sandbox under a running selfcheck.py would orphan its GPU sandbox
                await asyncio.wait_for(controller.exec(TERMINATE_COMMAND, timeout=CANCEL_GRACE_SECONDS),
                                       CANCEL_GRACE_SECONDS + 15)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not stop self-check %s gracefully: %s", check_id, exc)
        finally:  # even when cancelled: an undeleted self-check sandbox lives until its lifetime ends
            try:
                await asyncio.wait_for(controller.stop(), 180)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not delete the sandbox of self-check %s: %s", check_id, exc)

    async def _mirror_progress(self, entry, controller, seen):
        try:
            result = await controller.exec(f"cat {CONTROLLER_LOGS}/{entry['id']}/progress.json 2>/dev/null", timeout=30)
            value = json.loads(result.stdout) if result.stdout else None
        except Exception:  # noqa: BLE001 - progress is best effort
            return seen
        if isinstance(value, dict) and value != seen:
            if isinstance(value.get("stage"), str):
                entry["stage"] = value["stage"][:STAGE_CHARS]
            if not await self._show_progress(entry, value):
                return seen  # shown on the next poll
        return value

    def _report(self, entry, status, stage, error):
        from .selfcheck import summary
        report = {"schema_version": SCHEMA_VERSION, "check_id": entry["id"], "status": status, "stage": stage,
                  "error": error, "checks": []}
        report["summary"] = summary(report)
        return report

    async def _stage(self, entry, stage):
        entry["stage"] = stage
        await self._show_progress(entry, {"stage": stage, "updated_at": time.time()})

    async def _show_progress(self, entry, value):
        try:
            await self._write(f"checks/{entry['id']}/progress.json", value)
            await self._publish()
            return True
        except Exception as exc:  # noqa: BLE001 - best effort; a lost agent sandbox is not the check's fault
            logger.warning("Could not show self-check %s progress: %s", entry["id"], exc)
            return False

    async def _finish(self, entry, report):
        async with self.lock:
            if entry.get("copy_note"):
                notes = report.get("notes") if isinstance(report.get("notes"), list) else []
                report["notes"] = [*notes, entry["copy_note"]]
            entry.update(state=report["status"], stage=clipped(report.get("stage"), STAGE_CHARS),
                         finished_at=time.time(), summary=clipped(report.get("summary"), SUMMARY_CHARS),
                         submission_digest=report.get("submission_digest"))
            notes = report.get("notes") if isinstance(report.get("notes"), list) else []
            if report["status"] == "error" and self.state["refunds"] < self.limit:
                self.state["used"] -= 1
                self.state["refunds"] += 1
                entry["counted"] = False
                report["notes"] = [*notes, "The self-check failed for infrastructure reasons, so it does not count "
                                           "against your self-checks."]
            elif report["status"] == "error":
                report["notes"] = [*notes, f"This run's {self.limit} refunds for infrastructure errors are used up, "
                                           "so this check counted."]
            elif (report["status"] == "failed" and report.get("stage") in PRE_GPU_STAGES
                  and self.state["free"] < MAX_FREE):
                # Stopped by the copy, the snapshot rules or a static constraint: no GPU was used.
                self.state["used"] -= 1
                self.state["free"] += 1
                entry["counted"] = False
                report["notes"] = [*notes, "This check stopped before any GPU was used, so it does not count "
                                           "against your self-checks."]
            report["remaining_selfchecks"] = self.remaining
            self.state["running"] = None
            self.running = self.work = None
            write_json(self.directory / f"{entry['id']}.json", report)
            self.undelivered[entry["id"]] = report
            try:
                await asyncio.wait_for(self._flush(), DELIVER_SECONDS)
            except Exception as exc:  # noqa: BLE001 - _heal tries again; the session may also have ended
                logger.warning("Could not deliver self-check %s to the agent sandbox yet: %s", entry["id"], exc)
            self._persist()

    async def _flush(self):
        """Write the reports the agent sandbox lacks, then the status. A report goes first so that a status naming its
        outcome finds it, but one that cannot be written does not hold back the status."""
        failure = None
        for check_id, report in list(self.undelivered.items()):
            try:
                await self._write(f"checks/{check_id}/report.json", report)
                self.undelivered.pop(check_id, None)  # _heal may flush beside _finish; a rewrite is harmless
            except Exception as exc:  # noqa: BLE001 - raised once the status is out
                failure = failure or exc
        await self._publish()
        if failure is not None:
            raise failure

    async def _publish(self):
        async with self.writing:  # the snapshot is taken in turn, so a slower older one never lands last
            self.stale = True  # until this write lands; _heal writes it again
            await self._put("status.json", {**self.state, "checks": self.state["checks"][-MAX_CHECKS:],
                                            "remaining": self.remaining, "updated_at": time.time()})
            self.stale = False

    async def _write(self, relative, value):
        async with self.writing:
            await self._put(relative, value)

    async def _put(self, relative, value):
        payload = base64.b64encode(json.dumps(value, allow_nan=False, default=str).encode()).decode()
        code, _, stderr = await self.sandbox.run(WRITE_SCRIPT, f"{SPOOL}/{relative}", payload, timeout=60)
        if code:
            raise RuntimeError(f"Writing {relative} in the agent sandbox failed ({code}): {stderr[-300:]}")

    def _persist(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json(self.directory / "state.json", {**self.state, "remaining": self.remaining})


def for_trial(trial_config, trial_dir, volume, *, limit=DEFAULT_LIMIT):
    """A TrialSelfChecks for a Harbor trial on Modal whose agent sandbox mounts `volume`."""
    task_dir = Path(trial_config.task.path)
    grading_config = json.loads((task_dir / "tests/config.json").read_text())
    trial_name = trial_config.trial_name
    trial_dir = Path(trial_dir)

    def controller(check_id, lifetime):
        name = f"{trial_name}__selfcheck__{check_id[:12]}"
        return HarborController(trial_config, trial_dir / "selfcheck" / check_id, name, lifetime)

    session = f"{trial_name}__env"
    return TrialSelfChecks(sandbox=ModalAgentSandbox(session), controller=controller, trial_dir=trial_dir,
                           volume=volume, grading_config=grading_config, limit=limit, session=session)

