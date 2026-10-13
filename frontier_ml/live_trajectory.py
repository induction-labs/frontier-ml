"""Live Claude Code trajectories for `harbor view` while the agent is still working.

Harbor copies the agent's logs out of its sandbox and converts them to agent/trajectory.json only after the agent
ends, so the viewer shows nothing for the 12 hours a session can run. While a Claude Code agent runs on Modal, this
fetches the bytes its session logs gained since the last poll and rewrites agent/trajectory.json with Harbor's own
converter. It stops at AGENT_END and removes what it wrote, before Harbor's final copy and conversion.

Converting a long session takes a few hundred MB and seconds of CPU. Every trial's conversion runs in one worker
process, one at a time, so a job of 21 trials neither multiplies that memory inside Harbor nor holds its GIL. The worker
is replaced every few conversions to hand the memory back, and it offers itself to the kernel's OOM killer first: a
first live run on a 7.6 GB machine had Harbor itself killed, which orphans every sandbox and wallet of the job.
"""
import asyncio
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import contextlib
import logging
import os
from pathlib import Path
import shutil

from harbor.agents.installed.claude_code import ClaudeCode
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

from .runtime import submission_volume

logger = logging.getLogger(__name__)
DEFAULT_SECONDS = 120  # 21 conversions of a long session take about two minutes in the one worker
CONVERSIONS_PER_WORKER = 10
LIVE_DIR = ".live-agent"  # in the trial directory: the copied logs, removed when the agent ends
# One poll's fetch per file: more than an agent writes in a minute, small enough to hold in memory.
CHUNK_BYTES = 64 << 20

# Lists the session logs and the stream log Harbor reads the cost from. For each one longer than the copy this side
# holds (LIVE_OFFSETS: "path<TAB>bytes" lines) it prints "path<TAB>offset<TAB>length\n" and then those bytes. A file
# shorter than its copy was replaced, so it is fetched again from the start.
FETCH_SCRIPT = r"""set -u
cd "$LOGS" 2>/dev/null || exit 0
{ find sessions -type f -name '*.jsonl' 2>/dev/null; [ -f claude-code.txt ] && echo claude-code.txt; } |
while IFS= read -r f; do
  size=$(stat -c %s "$f") || continue
  have=$(printf '%s\n' "$LIVE_OFFSETS" | awk -F'\t' -v f="$f" '$1 == f { print $2 }')
  have=${have:-0}
  [ "$size" -lt "$have" ] && have=0
  [ "$size" -eq "$have" ] && continue
  length=$((size - have))
  [ "$length" -gt "$CHUNK" ] && length=$CHUNK
  printf '%s\t%s\t%s\n' "$f" "$have" "$length"
  tail -c +$((have + 1)) "$f" | head -c "$length"
done
"""


def held(live):
    """The size of each copied log, keyed by its path under the sandbox's agent logs directory."""
    return {path.relative_to(live).as_posix(): path.stat().st_size
            for path in live.rglob("*") if path.is_file() and path.name != "trajectory.json"}


def apply_chunks(live, data):
    """Write each fetched range into its copy; return how many ranges arrived."""
    position, count = 0, 0
    while position < len(data):
        end = data.index(b"\n", position)
        name, offset, length = data[position:end].decode().split("\t")
        offset, length = int(offset), int(length)
        target = live / name
        if not target.resolve().is_relative_to(live.resolve()):
            raise ValueError(f"Fetched path escapes the live directory: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "r+b" if offset and target.exists() else "wb") as handle:
            handle.truncate(offset)
            handle.seek(offset)
            handle.write(data[end + 1:end + 1 + length])
        position = end + 1 + length
        count += 1
    return count


def convert(live, trial, model_name):
    """Harbor's post-run conversion over the copy, published to agent/trajectory.json in one rename."""
    ClaudeCode(logs_dir=live, model_name=model_name).populate_context_post_run(AgentContext())
    written, published = live / "trajectory.json", trial / "agent" / "trajectory.json"
    if not written.is_file():
        return None
    published.parent.mkdir(exist_ok=True)
    os.replace(written, published)
    return fingerprint(published)


def fingerprint(path):
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def _volunteer_for_oom_kill():
    with contextlib.suppress(OSError):
        Path("/proc/self/oom_score_adj").write_text("1000")


_worker = None


async def convert_in_worker(live, trial, model_name):
    global _worker
    if _worker is None:
        _worker = ProcessPoolExecutor(1, initializer=_volunteer_for_oom_kill, max_tasks_per_child=CONVERSIONS_PER_WORKER)
    try:
        return await asyncio.get_running_loop().run_in_executor(_worker, convert, live, trial, model_name)
    except BrokenProcessPool:
        _worker = None  # killed, for one: the next conversion starts a new worker
        raise


def shutdown():
    global _worker
    if _worker is not None:
        _worker.shutdown(wait=False, cancel_futures=True)
        _worker = None


async def follow(event, seconds):
    """Poll until cancelled. Cancelling waits for a conversion in progress, then removes what this wrote, so a partial
    trajectory never outlives the agent: Harbor's final conversion writes the only one that remains. A trajectory that
    changed since this published it is Harbor's, and stays."""
    trial = Path(event.config.trials_dir) / event.trial_name
    live = trial / LIVE_DIR
    shutil.rmtree(live, ignore_errors=True)  # a retried attempt's session starts again
    live.mkdir(parents=True)
    sandbox, failures, conversion, published = None, 0, None, None
    try:
        while True:
            await asyncio.sleep(seconds)
            try:
                if sandbox is None:
                    sandbox = await submission_volume.agent_sandbox(f"{event.trial_name}__env")
                offsets = "\n".join(f"{name}\t{size}" for name, size in held(live).items())
                process = await sandbox.exec.aio("bash", "-c", FETCH_SCRIPT, text=False, timeout=max(30, seconds),
                                                 env={"LOGS": str(EnvironmentPaths().agent_dir),
                                                      "LIVE_OFFSETS": offsets, "CHUNK": str(CHUNK_BYTES)})
                data = await process.stdout.read.aio()
                await process.wait.aio()
                if apply_chunks(live, data):
                    # A conversion keeps running when its awaiting task is cancelled: hold on to it for the cleanup.
                    conversion = asyncio.ensure_future(convert_in_worker(live, trial, event.config.agent.model_name))
                    published = await asyncio.shield(conversion) or published
                failures = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the live view must never disturb the trial
                sandbox, failures = None, failures + 1
                if failures in (1, 10):
                    logger.warning("Live trajectory for %s: %s: %s", event.trial_name, type(exc).__name__, exc)
    finally:
        if conversion is not None:
            with contextlib.suppress(Exception):
                published = await conversion or published
        shutil.rmtree(live, ignore_errors=True)
        path = trial / "agent" / "trajectory.json"
        with contextlib.suppress(OSError):
            if published is not None and fingerprint(path) == published:
                path.unlink()
