"""A per-trial Modal Volume that carries the submission from the agent sandbox to grading.

Without it Harbor tars /workspace/output, downloads it to the host, uploads it to the separate
verifier, and the verifier uploads it again to the GPU sandbox; each hop is a single-core gzip.
On Modal the plugin mounts one fresh volume per trial at MOUNT in the agent and verifier sandboxes.
At AGENT_END a trusted root exec stops every actor process and tars the output into the volume,
leaving only MARKER in /workspace/output, so Harbor moves a few bytes. The verifier extracts the tar
to its own disk and snapshots that as usual; the GPU sandbox extracts it from its own mount, checked
against the snapshot digest. One uncompressed tar reads back at ~180 MB/s, while copying a directory
of small files file by file from a volume is bound by per-file latency (>15 min for a torch install).

Self-checks (selfcheck_service.py) use the same volume while the agent is still working: a root exec tars
/workspace/output into SELFCHECK_DIR/<check id>/ without stopping anything, and the self-check sandbox and its GPU
sandbox read that copy the same way. Only the newest copy is kept; the freeze deletes it with everything else.
"""
from contextlib import contextmanager
from pathlib import Path
import os
import re
import shutil
import tarfile
import uuid

from ..validation.errors import VerificationFailure

MOUNT = "/mnt/fml-submission"
ENV = "FML_SUBMISSION_VOLUME"  # set by the plugin in the verifier environment; never read from the agent
MARKER = ".fml-submission-volume"
ARCHIVE = "output.tar"
COMPLETE = ".complete"
SELFCHECK_DIR = "selfcheck"

# The agent has finished: nothing it started may change the output during or after the copy.
STOP_AGENT_SCRIPT = """pkill -KILL -u actor || true
for _ in $(seq 50); do pgrep -u actor >/dev/null || break; sleep 0.2; done
"""
FREEZE_SCRIPT = f"""set -euo pipefail
{STOP_AGENT_SCRIPT}# A self-check copy (root) cancelled mid-way may still be reading; the bracket keeps this line from matching itself.
pkill -KILL -f 'fml-submission/selfch[e]ck' || true
find {MOUNT} -mindepth 1 -delete
files=$(find /workspace/output -type f | wc -l)
tar -cf {MOUNT}/{ARCHIVE} -C /workspace/output .
touch {MOUNT}/{COMPLETE}
sync {MOUNT}
mv /workspace/output /workspace/.output-in-volume
mkdir /workspace/output
printf '%s\\n' "$FML_SUBMISSION_VOLUME" > /workspace/output/{MARKER}
echo "$(stat -c %s {MOUNT}/{ARCHIVE}) $files"
"""


# A self-check copy while the agent keeps working: nothing is stopped or moved, and only the newest copy is kept.
# tar runs as the agent, so the copy holds nothing the agent could not read itself (a symlinked output folder is
# refused). It exits 1 when a file changed while it was read (the agent is still writing; the snapshot then checks
# the copy) and 2 when it could not read a file.
SELFCHECK_COPY_SCRIPT = f"""set -uo pipefail
[ -d /workspace/output ] && [ ! -L /workspace/output ] || {{ echo "/workspace/output is not a directory" >&2; exit 3; }}
directory={MOUNT}/{SELFCHECK_DIR}/$FML_SELFCHECK_ID
rm -rf {MOUNT}/{SELFCHECK_DIR} && mkdir -p "$directory" || exit 4
files=$(find /workspace/output -type f | wc -l)
su actor -s /bin/sh -c 'exec tar -cf - -C /workspace/output .' > "$directory/{ARCHIVE}"
status=$?
[ $status -le 1 ] || exit $status
sync {MOUNT}
echo "$(stat -c %s "$directory/{ARCHIVE}") $files $status"
"""


def selfcheck_archive(check_id):
    """The volume-relative path of a self-check's copy."""
    if not re.fullmatch(r"[0-9a-f]{12,64}", check_id or ""):
        raise ValueError("Self-check IDs are lowercase hex")
    return f"{SELFCHECK_DIR}/{check_id}/{ARCHIVE}"


def safe_archive(relative):
    """A volume-relative archive path the GPU sandbox may extract: ARCHIVE or a self-check copy."""
    if relative != ARCHIVE and not re.fullmatch(rf"{SELFCHECK_DIR}/[0-9a-f]{{12,64}}/{ARCHIVE}", relative or ""):
        raise VerificationFailure(f"Unexpected submission archive {relative!r}")
    return relative


def new_name():
    return "fml-submission-" + uuid.uuid4().hex


async def create(name):
    import modal
    await modal.Volume.objects.create.aio(name, version=2)


async def delete(name):
    import modal
    await modal.Volume.objects.delete.aio(name, allow_missing=True)


async def agent_sandbox(session_id):
    """The running Modal sandbox Harbor tagged with this session (the agent environment is `<trial>__env`)."""
    import modal
    sandboxes = [sandbox async for sandbox in modal.Sandbox.list.aio(tags={"harbor.session_id": session_id})]
    if len(sandboxes) != 1:
        raise RuntimeError(f"Expected one running sandbox tagged {session_id}, found {len(sandboxes)}")
    return sandboxes[0]


async def stop_agent(session_id, *, timeout=60):
    """Kill every process of the agent user (FREEZE_SCRIPT's first step), before a self-check is torn down."""
    sandbox = await agent_sandbox(session_id)
    process = await sandbox.exec.aio("bash", "-c", STOP_AGENT_SCRIPT, timeout=timeout)
    await process.wait.aio()


async def freeze(session_id, name, *, timeout=1800):
    """Copy the agent sandbox's /workspace/output into the volume; return its size and file count."""
    sandbox = await agent_sandbox(session_id)
    process = await sandbox.exec.aio("bash", "-c", FREEZE_SCRIPT, env={ENV: name}, timeout=timeout)
    code = await process.wait.aio()
    stdout, stderr = await process.stdout.read.aio(), await process.stderr.read.aio()
    if code:
        raise RuntimeError(f"Copy into {name} exited {code}: {stderr[-1000:]}")
    size, files = stdout.split()[-2:]
    return {"bytes": int(size), "files": int(files)}


@contextmanager
def staged(submission, config, staging):
    """Yield the directory to snapshot: the submission itself, or the volume's tar extracted to staging."""
    submission = Path(submission)
    if not (submission / MARKER).is_file():
        yield submission
        return
    name = os.environ.get(ENV)
    if not name:
        raise VerificationFailure(f"The submission was moved into a Modal volume; grading it needs {ENV} "
                                  f"and that volume mounted at {MOUNT}")
    if not (Path(MOUNT) / COMPLETE).is_file():
        raise VerificationFailure(f"Submission volume {name} is not mounted or its copy is incomplete")
    staging = extract(Path(MOUNT) / ARCHIVE, staging)
    config["submission_volume"] = name
    try:
        yield staging
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def extract(archive, staging):
    """Unpack a submission tar into the new directory `staging` and return it."""
    staging = Path(staging)
    staging.mkdir(mode=0o700)
    try:
        with open(archive, "rb", buffering=16 << 20) as stream, tarfile.open(fileobj=stream, mode="r:") as tar:
            # The data filter rejects absolute paths, escapes, devices and links out of staging;
            # snapshot_submission then rejects any remaining symlinks or special files.
            tar.extractall(staging, filter="data")
    except (tarfile.TarError, OSError) as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise VerificationFailure(f"Unsafe or unreadable submission archive: {exc}") from exc
    return staging
