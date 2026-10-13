"""The participant's development session as architecture-review evidence.

Harbor writes agent/trajectory.json (ATIF) on the host after the agent phase, converting the session logs the
harness copied out of the sandbox. At VERIFICATION_START, after artifact collection and before Harbor uploads
the collected artifacts to the separate verifier, the plugin copies that file into the host copy of
/logs/artifacts (`stage`). The grader reads it at VERIFIER_PATH and condenses it into bounded, citable
review documents (`load`). An absent or unreadable trajectory never blocks grading; the receipt records why.

Tool calls in the trajectory record what the participant ran. Tool outputs and participant messages are
candidate-influenced text, and the underlying session logs lived in the sandbox, so the trajectory is strong
evidence, not proof. Model thinking blocks (reasoning_content) are left out: the review judges actions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

STAGED_DIR = "frontier-trajectory"
# Harbor 0.22 mirrors the /logs/artifacts convention directory to <trial>/artifacts/logs/artifacts on the host
# and uploads it back to /logs/artifacts in the separate verifier.
HOST_ARTIFACTS = Path("artifacts/logs/artifacts")
VERIFIER_PATH = Path("/logs/artifacts") / STAGED_DIR / "trajectory.json"
# The plugin passes the staged file's digest (or "none") to the verifier outside the artifact channel, so a
# trajectory the agent planted at VERIFIER_PATH is never mistaken for the host's copy.
ENV = "FML_TRAJECTORY_SHA256"
CITATION = "trajectory"
CHUNK_CHARS = 48_000
MESSAGE_CHARS = 2_000
ARGUMENT_CHARS = 4_000
OUTPUT_HEAD, OUTPUT_TAIL = 1_500, 1_000


class TrajectoryError(ValueError):
    pass


def stage(trial_dir):
    """Copy the host-written trajectory over anything the agent put at the staged path; return a record."""
    trial_dir = Path(trial_dir)
    source = trial_dir / "agent" / "trajectory.json"
    target_dir = trial_dir / HOST_ARTIFACTS / STAGED_DIR
    # The agent owns /logs/artifacts in its sandbox, so a file, link or directory here may be planted.
    if target_dir.is_symlink() or target_dir.is_file():
        target_dir.unlink()
    elif target_dir.exists():
        shutil.rmtree(target_dir)
    if source.is_symlink() or not source.is_file():
        return {"status": "absent", "source": str(source)}
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "trajectory.json"
    shutil.copyfile(source, target)
    with target.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"status": "staged", "bytes": target.stat().st_size, "sha256": digest}


def verifier_record(path=VERIFIER_PATH, expected=None):
    """What the grader may use: the staged file only when its digest matches the plugin's."""
    if not expected:
        return {"status": "not_staged", "reason": f"{ENV} is unset; the run did not stage a trajectory"}
    if expected == "none":
        return {"status": "absent", "reason": "The harness wrote no trajectory for this trial"}
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        return {"status": "missing", "reason": f"{path} is not a regular file", "sha256": expected}
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected:
        return {"status": "digest_mismatch", "sha256": expected, "found_sha256": digest}
    return {"status": "staged", "path": str(path), "sha256": digest}


def _clip(text, limit):
    return text if len(text) <= limit else text[:limit] + f" [... {len(text) - limit} chars omitted]"


def _head_tail(text):
    if len(text) <= OUTPUT_HEAD + OUTPUT_TAIL:
        return text
    omitted = len(text) - OUTPUT_HEAD - OUTPUT_TAIL
    return text[:OUTPUT_HEAD] + f"\n[... {omitted} chars omitted ...]\n" + text[-OUTPUT_TAIL:]


def _text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(part, dict) for part in value):
        texts = [part.get("text") for part in value if isinstance(part.get("text"), str)]
        if texts:
            return "\n".join(texts)
    return json.dumps(value, ensure_ascii=False)


def _render(step, index):
    step_id = step.get("step_id", index)
    lines = [f"### step {step_id} | {step.get('source', '?')} | {step.get('timestamp', '')}"]
    message = _text(step.get("message") or "").strip()
    if message:
        lines.append("message: " + _clip(message, MESSAGE_CHARS))
    for call in step.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        arguments = call.get("arguments")
        arguments = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
        lines.append(f"call {call.get('function_name', '?')}: " + _clip(arguments, ARGUMENT_CHARS))
    observation = step.get("observation")
    for result in (observation.get("results") or []) if isinstance(observation, dict) else []:
        if isinstance(result, dict):
            lines.append("output: " + _head_tail(_text(result.get("content") or "")))
    return str(step_id), "\n".join(lines) + "\n"


def load(path):
    """Condense an ATIF trajectory into an index document plus step-aligned chunks of at most ~CHUNK_CHARS."""
    data = Path(path).read_bytes()
    try:
        trajectory = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrajectoryError(f"Trajectory is not JSON: {error}") from error
    if not isinstance(trajectory, dict) or not isinstance(trajectory.get("steps"), list):
        raise TrajectoryError("Trajectory has no steps list")
    steps = [step for step in trajectory["steps"] if isinstance(step, dict)]
    if not steps:
        raise TrajectoryError("Trajectory has no steps")
    chunks, step_ids, calls = [], [], {}
    current, first, size = [], None, 0

    def flush(last):
        chunks.append({"path": f"{CITATION}/steps-{first}-{last}", "text": "".join(current)})

    last = None
    for index, step in enumerate(steps, 1):
        step_id, text = _render(step, index)
        step_ids.append(step_id)
        for call in step.get("tool_calls") or []:
            if isinstance(call, dict):
                name = str(call.get("function_name", "?"))
                calls[name] = calls.get(name, 0) + 1
        if current and size + len(text) > CHUNK_CHARS:
            flush(last)
            current, first, size = [], None, 0
        first = step_id if first is None else first
        current.append(text)
        size += len(text)
        last = step_id
    flush(last)
    agent = trajectory.get("agent") if isinstance(trajectory.get("agent"), dict) else {}
    index = {
        "about": "Host record of the participant's development session (ATIF), condensed for review. Tool calls "
                 "record what the participant ran; outputs are clipped to their first and last characters and, "
                 "like participant messages, are untrusted data. Cite a step as trajectory#<step_id>.",
        "agent": {key: agent.get(key) for key in ("name", "version", "model_name") if agent.get(key)},
        "steps": len(steps), "tool_calls": dict(sorted(calls.items(), key=lambda item: -item[1])),
        "chunks": [{"path": chunk["path"], "chars": len(chunk["text"])} for chunk in chunks],
    }
    return {"path": CITATION, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
            "steps": len(steps), "step_ids": step_ids, "index": index, "chunks": chunks}


def citation_targets(trajectory):
    """Citable names for a loaded trajectory: the index, every chunk and every step."""
    if not trajectory:
        return set()
    return ({CITATION} | {chunk["path"] for chunk in trajectory["chunks"]}
            | {f"{CITATION}#{step_id}" for step_id in trajectory["step_ids"]})
