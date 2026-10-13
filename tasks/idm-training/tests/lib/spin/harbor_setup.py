"""Public environment preparation run by Harbor before the agent starts.

/task/setup.json names pinned Hugging Face folders, or hashes for public data already
prepared on the operator's machine. Starting artifacts download into /workspace/artifacts.
Once all files are verified, setup.json is deleted and a marker allows health-check retries.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath

from .transfer import fetch_hf


PREPARED = ".prepared"


def prepared_files(directory: Path, expected: dict | None = None) -> dict[str, str]:
    """Hash a prepared data tree, rejecting links and, when supplied, any inventory mismatch."""
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Prepared data must be a regular directory")
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("Prepared data must contain only regular files and directories")
        if path.is_file():
            with path.open("rb") as stream:
                files[path.relative_to(directory).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    if not files:
        raise ValueError("Prepared data is empty")
    if expected is not None and files != expected:
        raise ValueError("Prepared data changed; rebuild the task image")
    return files


def artifact_target(artifacts_dir: Path, to: str) -> Path:
    """Destination under /workspace/artifacts; "" is the artifacts directory itself (the task's own artifacts/ folder)."""
    if to == "":
        return artifacts_dir
    path = PurePosixPath(to)
    if path.is_absolute() or "\\" in to or any(part in ("", ".", "..") for part in to.split("/")):
        raise ValueError(f"Starting artifact destination must be a relative path: {to!r}")
    return artifacts_dir.joinpath(*path.parts)


def prepare(task_dir=Path("/task"), artifacts_dir=Path("/workspace/artifacts")):
    # The plugin passes a wallet's scoped token, or FML_NO_COMPUTE=1 for a run with no compute budget (--budget 0).
    if not os.environ.get("SPIN_BROKER_TOKEN") and os.environ.get("FML_NO_COMPUTE") != "1":
        raise ValueError("A Spin wallet is required; use --plugin frontier_ml.harbor_plugin:FrontierPlugin")
    if importlib.util.find_spec("frontier_ml") is not None:
        raise ValueError("The agent image must not install the operator harness")
    task_dir, artifacts_dir = Path(task_dir), Path(artifacts_dir)
    if not (task_dir / "setup.json").exists() and (task_dir / PREPARED).is_file():
        return  # Already prepared by an earlier run of this health check.
    manifest = json.loads((task_dir / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        path = task_dir / name
        if not path.resolve().is_relative_to(task_dir.resolve()) or path.is_symlink():
            raise ValueError("Unsafe validation file")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Public validation code changed; rebuild the task image")
    setup = json.loads((task_dir / "setup.json").read_text())
    if "public_prepared" in setup:
        if "public" in setup or not isinstance(setup["public_prepared"], dict):
            raise ValueError("Prepared public data excludes a download source")
        prepared_files(task_dir / "public", setup["public_prepared"])
    else:
        fetch_hf(setup["public"], task_dir / "public")
    for artifact in setup["artifacts"]:
        fetch_hf(artifact, artifact_target(artifacts_dir, artifact["to"]), optional=artifact.get("optional", False))
    # Setup runs as root; the coding agent must be able to train these weights.
    if setup["artifacts"] and os.geteuid() == 0:
        import pwd
        actor = pwd.getpwnam("actor")
        for entry in [artifacts_dir, *artifacts_dir.rglob("*")]:
            os.chown(entry, actor.pw_uid, actor.pw_gid)
    # The agent never needs the data source; remove it so the sandbox does not point at the dataset.
    (task_dir / PREPARED).write_text("prepared\n")
    (task_dir / "setup.json").unlink()


if __name__ == "__main__":
    try:
        prepare()
    except Exception as exc:
        # Wallet credentials must not enter setup logs.
        print(f"Task preparation failed: {type(exc).__name__}: {str(exc)[:300]}")
        raise SystemExit(1)
