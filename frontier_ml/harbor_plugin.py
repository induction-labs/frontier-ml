"""Spin trial lifecycle for Harbor; Harbor owns agents, environments and results.

Harbor 0.22 exposes lifecycle callbacks, but not a pre-construction config hook.
The only version-specific seam is preparing its pending trial configs at job
start. A nonempty trial environment mapping is passed by reference to Harbor's
environment; the START callback fills it before sandbox creation. Integration
tests exercise that contract against the installed Harbor, including retries.
"""
import asyncio
from datetime import datetime, timezone
from functools import partial
from importlib.metadata import version
import json
import logging
import math
import os
from pathlib import Path
import time
import tomllib
from urllib.parse import quote

import httpx
from harbor.models.agent.name import AgentName
from harbor.models.job.plugin import BaseJobPlugin

from spin.wallet import SpinClient, cents, static_pricing as price_table
from . import live_trajectory, review_trajectory, selfcheck_service, submissions
from .agent_surface import package_present
from .runtime import submission_volume
from .runtime.environments import is_modal, modal_profile, use_recoverable_modal
from .validation.progress import write_json

logger = logging.getLogger(__name__)
# Spin can accept a close asynchronously ("closing") until its provider confirms cleanup.
CLOSE_WAIT_SECONDS = 300
CLOSE_POLL_SECONDS = 10
# How long a wallet outlives the trial's agent time when wallet_seconds is not given: setup, the final collection
# and cleanup of leases still running at the end.
WALLET_MARGIN_SECONDS = 7200
# Starting the self-check service is a few root execs into the agent sandbox; one lost Modal exec stream must not
# turn self-checks off for the attempt. The cap bounds how long AGENT_START holds the agent back.
SELFCHECK_START_ATTEMPTS = 3
SELFCHECK_RETRY_SECONDS = 5
SELFCHECK_START_SECONDS = 120


class FrontierPlugin(BaseJobPlugin):
    def __init__(self, budget_usd="50", wallet_seconds=None, keep_submission_volume="0", static_pricing=True,
                 selfchecks=selfcheck_service.DEFAULT_LIMIT, live_trajectory_seconds=live_trajectory.DEFAULT_SECONDS):
        # `--pk budget_usd=0` (`frontier run --budget 0`) is a run with no compute budget: no wallet is created.
        self.budget_cents = 0 if _is_zero(budget_usd) else cents(str(budget_usd))
        # Evaluation wallets spend their budget in fixed GPU prices so a budget buys the same compute whatever
        # the providers charge (the account still pays real cost). The default is Spin's GPU price index;
        # `--pk static_pricing=@table.json` (or a JSON table) pins other prices, `--pk static_pricing=false` uses
        # market prices.
        self.static_pricing = _static_pricing(static_pricing)
        # By default a wallet outlives the trial's agent time by WALLET_MARGIN_SECONDS (see _wallet_seconds).
        self.wallet_seconds = None if wallet_seconds in (None, "") else int(wallet_seconds)
        if self.wallet_seconds is not None and self.wallet_seconds < 1:
            raise ValueError("wallet_seconds must be positive")
        # `--pk keep_submission_volume=1` retains graded submissions so a completed trial can still be
        # regraded after a verifier fix; by default only ungraded submissions are kept.
        self.keep_submission_volume = str(keep_submission_volume).lower() in {"1", "true", "yes"}
        # `frontier selfcheck` allowance per trial attempt (selfcheck_service.py); `--pk selfchecks=0` turns it off.
        # Harbor parses --pk values as JSON, so int() alone would read `true` as 1 and truncate 2.5.
        if isinstance(selfchecks, bool) or not str(selfchecks).strip().isdecimal():
            raise ValueError("selfchecks must be zero or a positive integer")
        self.selfchecks = int(selfchecks)
        # How often a running Claude Code agent's trajectory is refreshed for `harbor view`; 0 turns it off.
        if isinstance(live_trajectory_seconds, bool) or not str(live_trajectory_seconds).strip().isdecimal():
            raise ValueError("live_trajectory_seconds must be zero or a positive integer")
        self.live_trajectory_seconds = int(live_trajectory_seconds)
        self.live_trajectories = {}  # trial attempt ID -> the task refreshing its live trajectory
        self.sessions = {}
        self.selfcheck_services = {}
        self.submitted = {}  # trial attempt ID -> when its agent ran `frontier submit --confirm`, until grading starts

    async def on_job_start(self, job):
        if version("harbor") != "0.22.0":
            raise RuntimeError("Frontier's lifecycle integration is tested with harbor==0.22.0")
        # The verifier and self-check environments forward MODAL_TOKEN_* from this process; a `modal token new`
        # login lives only in ~/.modal.toml.
        os.environ.update(modal_profile())
        for hook, phase in (("on_trial_started", "setup"), ("on_environment_started", "agent_setup"),
                            ("on_agent_started", "agent"), ("on_agent_ended", "collecting"),
                            ("on_verification_started", "grading"), ("on_trial_ended", "ended"),
                            ("on_trial_cancelled", "cancelled")):
            getattr(job, hook)(partial(self._observe, phase=phase))
        # Before Harbor uploads the collected artifacts to the separate verifier; regrades seed agent/ and
        # artifacts/ from the source trial first, so they review the original session too.
        job.on_verification_started(self._stage_trajectory)
        # Covers both the agent environment and Harbor's separate verifier,
        # including regrades, without changing resource or network settings.
        for config in job._remaining_trial_configs:
            config.environment = use_recoverable_modal(config.environment)
        if job.config.is_regrade:
            # Regrading needs neither development wallets nor checkpoints, but a submission frozen into a
            # volume can only be graded from that volume: mount the source trial's again.
            for config in job._remaining_trial_configs:
                self._reattach_submission_volume(config)
            return
        for config in job._remaining_trial_configs:
            path = config.task.path
            if path is None or not package_present(Path(path)):
                raise ValueError("FrontierPlugin needs a local rendered task; run `frontier build` or `frontier run`")
            path = Path(path)
            task_config = tomllib.loads((path / "task.toml").read_text())
            # Add this per trial, after CLI/config overrides are resolved; the rendered
            # task has no fixed time claim that could contradict this actual limit.
            seconds = agent_seconds(config)
            config.extra_instructions = [*config.extra_instructions,
                                         f"You have {seconds:g} seconds to complete the task. "
                                         "Finish all required files in /workspace/output/ within this time limit."]
            if task_config["environment"].get("env") or (path / "environment/docker-compose.yaml").exists():
                raise ValueError("Frontier tasks use a single environment with trial-scoped variables")
            setup = json.loads((path / "environment/setup.json").read_text())
            # Harbor's job templates share these models across trials. Detach
            # them before assigning any per-trial capability or credential.
            config.environment = config.environment.model_copy(deep=True)
            config.environment.env = {**config.environment.env, "FML_HARBOR_TASK": setup["task"]}
            if self.budget_cents == 0:
                # No wallet will be created (`_start`); spin.harbor_setup accepts this in place of a wallet token.
                config.environment.env = {**config.environment.env, "FML_NO_COMPUTE": "1"}
            # A read-only Hugging Face token so the agent can download gated models and datasets. It is a download
            # credential, not a provider account key; publish_trajectories redacts hf_ tokens from published sessions.
            config.environment.env = {**config.environment.env, "HF_TOKEN": agent_hf_token()}
            config.extra_instructions = [*config.extra_instructions, HF_TOKEN_NOTE]
            if any(key in config.environment.env for key in ("SPIN_BROKER_TOKEN", "SPIN_API_KEY")):
                raise ValueError("The plugin creates a separate wallet for every trial; remove supplied Spin credentials")
            if _is_modal(config.environment):
                await self._attach_submission_volume(config)
            # Self-checks copy the output through the submission volume, which only Modal trials have, and their
            # service also takes `frontier submit`. CONTRACT.md describes the default allowance and the submission;
            # say so when this run differs from it.
            if not (self.selfchecks and _is_modal(config.environment)):
                config.extra_instructions = [*config.extra_instructions,
                                             "`frontier selfcheck` is not available in this run."]
            elif self.selfchecks != selfcheck_service.DEFAULT_LIMIT:
                allowance = f"{self.selfchecks} self-check{'' if self.selfchecks == 1 else 's'}"
                config.extra_instructions = [*config.extra_instructions,
                                             f"This run allows {allowance} (`frontier selfcheck`)."]
            if not _is_modal(config.environment):
                config.extra_instructions = [*config.extra_instructions, SUBMIT_UNAVAILABLE]
        job.on_trial_started(self._start)
        # ENVIRONMENT_START fires before Harbor creates the sandbox; AGENT_START once it is up and set up.
        job.on_agent_started(self._start_selfchecks)
        job.on_agent_started(self._start_live_trajectory)
        # Before Harbor's final copy of the agent logs, which runs after the AGENT_END hooks.
        job.on_agent_ended(self._stop_live_trajectory)
        job.on_agent_ended(self._freeze_submission)
        # VERIFICATION_START is after output collection and agent-environment stop. Closing there
        # must not block grading; END waits for Spin to confirm and on_job_end reports failures.
        job.on_verification_started(self._close_before_grading)
        job.on_verification_started(self._settle_submission)
        job.on_trial_ended(self._stop_selfchecks)
        job.on_trial_ended(self._stop_live_trajectory)
        job.on_trial_ended(self._close)
        job.on_trial_ended(self._release_submission_volume)
        job.on_trial_cancelled(self._stop_selfchecks)
        job.on_trial_cancelled(self._stop_live_trajectory)
        job.on_trial_cancelled(self._close)

    async def _observe(self, event, *, phase):
        path = Path(event.config.trials_dir) / event.trial_name / "frontier-progress.json"
        try:
            previous = json.loads(path.read_text())
        except (OSError, ValueError):
            previous = {}
        attempt = str(event.trial_id)
        phases = previous.get("phases", {}) if previous.get("attempt_id") == attempt else {}
        now = time.time()
        phases[phase] = now
        write_json(path, {"attempt_id": attempt, "phase": phase, "updated_at": now, "phases": phases})

    async def _stage_trajectory(self, event):
        trial = Path(event.config.trials_dir) / event.trial_name
        try:
            record = review_trajectory.stage(trial)
        except OSError as exc:
            # A hook failure would abort the trial; grading reviews without the trajectory and records that.
            record = {"status": "stage_failed", "error": f"{type(exc).__name__}: {exc}"}
            logger.warning("Could not stage the agent trajectory for review: %s", exc)
        write_json(trial / "review-trajectory.json", record)
        # Harbor builds the verifier from this trial config after the hook returns. Detach the shared template.
        event.config.verifier = event.config.verifier.model_copy(deep=True)
        event.config.verifier.env = {**event.config.verifier.env,
                                     review_trajectory.ENV: record.get("sha256") or "none"}

    async def _attach_submission_volume(self, config):
        # The agent and the separate verifier sandbox are both built from config.environment. A retry
        # reuses the config, and so the volume; the freeze step clears it before copying.
        name = submission_volume.new_name()
        await submission_volume.create(name)
        self._mount_submission_volume(config, name)

    def _reattach_submission_volume(self, config):
        source = config.source_trial
        path = Path(source.path) / "submission-volume.json" if source is not None and source.path else None
        if path is None or not path.is_file():
            return  # a hub source, or a submission graded from /workspace/output
        record = json.loads(path.read_text())
        if record.get("state") == "deleted":
            logger.warning("%s: submission volume %s was deleted after grading; keep graded submissions with "
                           "--pk keep_submission_volume=1 to regrade them", source.path, record["volume"])
            return
        # The source trial's volume outlives the regrade: this mode registers no release hook.
        config.environment = config.environment.model_copy(deep=True)
        self._mount_submission_volume(config, record["volume"])

    @staticmethod
    def _mount_submission_volume(config, name):
        volumes = {**(config.environment.kwargs.get("volumes") or {}), submission_volume.MOUNT: name}
        config.environment.kwargs = {**config.environment.kwargs, "volumes": volumes}
        config.verifier = config.verifier.model_copy(deep=True)
        config.verifier.env = {**config.verifier.env, submission_volume.ENV: name}

    async def _start_selfchecks(self, event):
        """Start the service behind `frontier selfcheck` and `frontier submit` (also with self-checks off)."""
        name = (event.config.environment.kwargs.get("volumes") or {}).get(submission_volume.MOUNT)
        identifier = str(event.trial_id)
        if name is None or identifier in self.selfcheck_services:
            return
        submissions.forget(f"{event.trial_name}__env")  # a retry reuses the trial's name; its agent has not submitted
        service = None
        try:
            service = selfcheck_service.for_trial(event.config, Path(event.config.trials_dir) / event.trial_name,
                                                  name, limit=self.selfchecks)
            await asyncio.wait_for(_start_service(service, event.trial_name), SELFCHECK_START_SECONDS)
        except Exception as exc:
            # A hook failure would abort the trial; the agent's `frontier selfcheck` and `frontier submit` then report
            # themselves unavailable, and the output is graded as it is when the session ends.
            logger.warning("Self-checks and submission unavailable for %s: %s: %s", event.trial_name,
                           type(exc).__name__, exc)
            if service is not None:  # a status.json already written would otherwise invite requests nobody serves
                try:
                    await asyncio.wait_for(service.stop("Self-checks could not start in this run."), 30)
                except Exception:  # noqa: BLE001 - best effort
                    pass
            return
        self.selfcheck_services[identifier] = service

    async def _stop_selfchecks(self, event):
        service = self.selfcheck_services.pop(str(event.trial_id), None)
        if service is not None:
            try:
                await asyncio.shield(service.stop())
            except Exception as exc:
                logger.warning("Stopping self-checks for %s failed: %s", event.trial_name, exc)

    async def _start_live_trajectory(self, event):
        if (self.live_trajectory_seconds and event.config.agent.name == AgentName.CLAUDE_CODE.value
                and _is_modal(event.config.environment)):
            self.live_trajectories[str(event.trial_id)] = asyncio.create_task(
                live_trajectory.follow(event, self.live_trajectory_seconds))

    async def _stop_live_trajectory(self, event):
        task = self.live_trajectories.pop(str(event.trial_id), None)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _freeze_submission(self, event):
        submitted_at = getattr(self.selfcheck_services.get(str(event.trial_id)), "submitted_at", None)
        if submitted_at is not None:
            self.submitted[str(event.trial_id)] = submitted_at
        if str(event.trial_id) in self.selfcheck_services:
            # The agent's time is up: kill its processes first, so nothing it left running can change the output while
            # a self-check winds down (its GPU stops before the freeze reuses the volume the check reads).
            try:
                await submission_volume.stop_agent(f"{event.trial_name}__env")
            except Exception as exc:
                logger.warning("Could not stop the agent's processes before self-check teardown (%s): %s",
                               event.trial_name, exc)
        await self._stop_selfchecks(event)
        name = (event.config.environment.kwargs.get("volumes") or {}).get(submission_volume.MOUNT)
        if name is None:
            return
        record = {"volume": name}
        try:
            record.update(await submission_volume.freeze(f"{event.trial_name}__env", name), state="in_volume")
        except Exception as exc:
            # /workspace/output is left in place, so Harbor transfers it and the verifier grades that copy.
            record.update(state="copy_failed", error=f"{type(exc).__name__}: {exc}")
            logger.warning("Submission copy into Modal volume %s failed; falling back to Harbor's transfer: %s", name, exc)
        self._record_volume(event, record)

    async def _settle_submission(self, event):
        """Record a session that ended with `frontier submit` in submission.json. The service ended it by killing the
        agent's processes, which the harness reports as a failed command (NonZeroAgentExitCodeError, exit 137): that is
        the submission, not an agent error, so the exception moves from the trial result into the record. A timeout
        that beat the kill stays."""
        submitted_at = self.submitted.pop(str(event.trial_id), None)
        if submitted_at is None:
            return
        trial = Path(event.config.trials_dir) / event.trial_name
        record = {"submitted_at": datetime.fromtimestamp(submitted_at, timezone.utc).isoformat(),
                  "via": "frontier submit"}
        info = event.result.exception_info
        # Only the agent phase has ended by VERIFICATION_START, and it records only timeouts and failed commands.
        if info is not None and info.exception_type != "AgentTimeoutError":
            record["agent_exit"] = {"exception_type": info.exception_type,
                                    "exception_message": info.exception_message[:2000]}
            event.result.exception_info = None
            (trial / "exception.txt").unlink(missing_ok=True)
        write_json(trial / "submission.json", record)

    async def _release_submission_volume(self, event):
        # The volume holds the only copy of the submission: delete it once it has been graded, and keep
        # it for debugging or regrading when grading did not complete (a harbor retry also reuses it).
        name = (event.config.environment.kwargs.get("volumes") or {}).get(submission_volume.MOUNT)
        if name is None:
            return
        trial = Path(event.config.trials_dir) / event.trial_name
        path = trial / "submission-volume.json"
        record = json.loads(path.read_text()) if path.is_file() else {"volume": name}
        try:
            status = json.loads((trial / "verifier/result.json").read_text()).get("status")
        except (OSError, ValueError):
            status = None
        if event.result.verifier_result is None or status != "completed":
            logger.warning("Keeping submission volume %s because grading did not complete", name)
            return
        if self.keep_submission_volume:
            logger.info("Keeping submission volume %s (keep_submission_volume) for possible regrading", name)
            record["state"] = "kept"
            self._record_volume(event, record)
            return
        try:
            await submission_volume.delete(name)
            record["state"] = "deleted"
        except Exception as exc:
            logger.warning("Could not delete submission volume %s: %s", name, exc)
        self._record_volume(event, record)

    def _record_volume(self, event, record):
        path = Path(event.config.trials_dir) / event.trial_name / "submission-volume.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2) + "\n")

    async def _wallet_request(self, path, body, *, key=None):
        account = SpinClient()
        if account.url.endswith("/v1") and path.startswith("/v1/"):
            path = path[3:]
        headers = {"Authorization": "Bearer " + account.token}
        if key:
            headers["Idempotency-Key"] = key
        async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
            response = await client.post(account.url + path, headers=headers, json=body)
            response.raise_for_status()
            return response.json()

    def _wallet_seconds(self, config):
        if self.wallet_seconds is not None:
            return self.wallet_seconds
        return math.ceil(agent_seconds(config)) + WALLET_MARGIN_SECONDS

    def _record(self, session):
        path = session["directory"] / "spin-wallet.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = ("wallet_id", "state", "budget_cents", "static_pricing")
        path.write_text(json.dumps({key: session[key] for key in keys if key in session}, indent=2) + "\n")

    async def _start(self, event):
        identifier = str(event.trial_id)
        if identifier in self.sessions:
            return
        # START runs before Harbor starts the environment. No provider account
        # key is added to config, the sandbox, or a trial artifact.
        if self.budget_cents == 0:
            return  # `--budget 0`: no compute budget, so no wallet and no Spin credentials (the rendered prompt says so)
        session = None
        try:
            body = {"name": "harbor-" + identifier, "budget_cents": self.budget_cents,
                    "expires_in_seconds": self._wallet_seconds(event.config)}
            if self.static_pricing is not None:
                body["static_pricing"] = self.static_pricing
            wallet = await self._wallet_request("/v1/wallets", body, key="harbor-" + identifier)
            session = {"wallet_id": wallet["id"], "state": "active", "budget_cents": self.budget_cents,
                       "static_pricing": self.static_pricing,
                       "directory": Path(event.config.trials_dir) / event.trial_name}
            self.sessions[identifier] = session
            env = wallet["env"]
            if not env.get("SPIN_BROKER_TOKEN") or not broker_url_allowed(env.get("SPIN_BROKER_URL", "")):
                raise ValueError("Spin returned invalid scoped credentials")
            event.config.environment.env.update({key: env[key] for key in ("SPIN_BROKER_URL", "SPIN_BROKER_TOKEN")})
            self._record(session)
        except BaseException:
            if session is not None:
                await asyncio.shield(self._close_session(session))
            raise

    async def _close_session(self, session, *, wait_seconds=0):
        """Close the wallet, polling up to wait_seconds while Spin reports it as "closing"."""
        if session["state"] == "closed":
            return
        deadline = time.monotonic() + wait_seconds
        try:
            while True:
                result = await self._wallet_request("/v1/wallets/" + quote(str(session["wallet_id"]), safe="") + "/close", {})
                if result.get("errors"):
                    raise RuntimeError("Spin reported cleanup errors")
                if result.get("status") == "closed" or result.get("cleanup_complete") is True:
                    session["state"] = "closed"
                    return
                if result.get("status") != "closing":
                    raise RuntimeError("Spin could not confirm resource cleanup")
                session["state"] = "closing"
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                await asyncio.sleep(min(CLOSE_POLL_SECONDS, remaining))
        except Exception:
            session["state"] = "cleanup_failed"
            raise RuntimeError("Spin wallet cleanup failed; see spin-wallet.json and reconcile the wallet") from None
        finally:
            self._record(session)

    async def _close_hook(self, event, wait_seconds):
        session = self.sessions.get(str(event.trial_id))
        if session is None:
            return
        try:
            await asyncio.shield(self._close_session(session, wait_seconds=wait_seconds))
        except RuntimeError as exc:
            # A hook failure would abort the trial; on_job_end retries and reports it instead.
            logger.warning("%s (wallet %s)", exc, session["wallet_id"])

    async def _close_before_grading(self, event):
        await self._close_hook(event, 0)

    async def _close(self, event):
        await self._close_hook(event, CLOSE_WAIT_SECONDS)

    async def on_job_end(self, job_result):
        live_trajectory.shutdown()
        errors = []
        for session in self.sessions.values():
            try:
                await self._close_session(session)
            except Exception as exc:
                errors.append(str(exc))
        pending = [session["wallet_id"] for session in self.sessions.values() if session["state"] == "closing"]
        if pending:
            logger.warning("Spin is still releasing wallets %s; their reservations return once cleanup is confirmed",
                           ", ".join(pending))
        if errors:
            raise RuntimeError("; ".join(errors))


def _is_zero(value) -> bool:
    """A $0 budget, however Harbor parsed `--pk budget_usd=0` (0, 0.0 or the string)."""
    try:
        return float(str(value)) == 0
    except ValueError:
        return False


def _static_pricing(value):
    """The wallet's static_pricing: True (the price index), a price table, or None for market prices.

    Harbor parses --pk values as JSON, so booleans and tables arrive parsed; anything else is a string.
    """
    if isinstance(value, bool):
        return True if value else None
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    if text.lower() in {"", "0", "false", "no", "off", "none", "market"}:
        return None
    if text.lower() in {"1", "true", "yes", "on", "index"}:
        return True
    table = price_table(text)
    if not isinstance(table, dict):
        raise ValueError("static_pricing must be true, false, a price table, or @path to one")
    return table


def agent_seconds(config) -> float:
    """The trial's agent time limit, as Harbor 0.22's _compute_agent_timeout_sec / _resolve_timeout_sec resolve it."""
    task_config = tomllib.loads((Path(config.task.path) / "task.toml").read_text())
    base_seconds = config.agent.override_timeout_sec or task_config["agent"]["timeout_sec"]
    multiplier = (config.agent_timeout_multiplier if config.agent_timeout_multiplier is not None
                  else config.timeout_multiplier)
    return min(base_seconds, config.agent.max_timeout_sec or float("inf")) * multiplier


def broker_url_allowed(url: str) -> bool:
    """Wallet credentials travel over HTTPS, or plain HTTP to a self-hosted Spin when SPIN_ALLOW_HTTP=1."""
    return url.startswith("https://") or (url.startswith("http://") and os.environ.get("SPIN_ALLOW_HTTP") == "1")


def _is_modal(environment):
    return is_modal(environment)


async def _start_service(service, trial_name):
    # TrialSelfChecks.start() only creates its watcher once its setup and status writes succeed, and both can be
    # rerun, so a failed start can simply be tried again.
    for attempt in range(1, SELFCHECK_START_ATTEMPTS + 1):
        try:
            return await service.start()
        except Exception as exc:
            if attempt == SELFCHECK_START_ATTEMPTS:
                raise
            logger.warning("Starting self-checks for %s failed (attempt %d): %s; retrying", trial_name, attempt, exc)
        await asyncio.sleep(SELFCHECK_RETRY_SECONDS * attempt)


HF_TOKEN_NOTE = ("HF_TOKEN in your environment is a read-only Hugging Face token for downloading models and datasets.")


SUBMIT_UNAVAILABLE = ("`frontier submit` is not available in this run: /workspace/output is graded as it is when your "
                      "session ends.")


HF_TOKEN_MISSING = ("HF_TOKEN is not set: every agent sandbox receives a read-only Hugging Face token for downloads. Export "
                    "HF_TOKEN (or FML_AGENT_HF_TOKEN to give agents a different token than the operator's) before starting a run.")


def agent_hf_token() -> str:
    """The Hugging Face token injected into every agent sandbox: FML_AGENT_HF_TOKEN, else HF_TOKEN (the operator's
    .env token, which must be read-only). A run without either fails before any sandbox starts."""
    token = os.environ.get("FML_AGENT_HF_TOKEN", "").strip() or os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise ValueError(HF_TOKEN_MISSING)
    return token
