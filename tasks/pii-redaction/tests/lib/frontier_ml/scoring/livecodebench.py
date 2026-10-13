"""Pinned LiveCodeBench release_v6, pass@1 with remote-only code execution.

Prompt, code extraction, safe testcase decoding, output comparison and the generation loop are
shared with the public development runner in frontier_ml.validation.livecodebench; this module
adds the source lock, checksums, preflight and the remote Harbor executor.
"""

import json
from pathlib import Path
import tempfile
import time
import urllib.request
import hashlib

from ..validation import livecodebench as public
from ..validation.livecodebench import (  # noqa: F401  (re-exported; one implementation)
    PrimitiveUnpickler, compare_cases, compare_stdio, decode_private, decode_value, extract_code,
    prompt, test_cases,
)


LOCK_PATH = Path(__file__).with_name("metadata") / "livecodebench-v6.json"


def source_lock():
    return json.loads(LOCK_PATH.read_text())


def iter_problems(root):
    return public.iter_problems(root, list(source_lock()["files"]))


def _hash_file(path):
    with open(path, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def validate_data(root):
    root, lock = Path(root), source_lock()
    for name, expected in lock["files"].items():
        path = root / name
        if path.is_symlink():
            raise ValueError("LCB source files must be regular immutable copies")
        info = path.stat()
        if info.st_size != expected["bytes"] or _hash_file(path) != expected["sha256"]:
            raise ValueError(f"Pinned LiveCodeBench checksum mismatch: {name}")
    ids, dates = set(), []
    for row in iter_problems(root):
        identity = (row["platform"], row["question_id"])
        if identity in ids:
            raise ValueError("Duplicate LiveCodeBench problem")
        ids.add(identity)
        dates.append(row["contest_date"])
    if len(ids) != lock["problems"]:
        raise ValueError("LiveCodeBench requires all 1055 release_v6 problems")
    return {"release": lock["release"], "revision": lock["revision"], "problems": len(ids),
            "first_contest": min(dates), "last_contest": max(dates),
            "upstream_revision": lock["upstream_revision"]}


def prepare(destination):
    root, lock = Path(destination), source_lock()
    root.mkdir(parents=True, exist_ok=True)
    for name, info in lock["files"].items():
        path = root / name
        if not path.exists():
            url = f"https://huggingface.co/datasets/{lock['repo']}/resolve/{lock['revision']}/{name}?download=true"
            temporary = path.with_suffix(".partial")
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                while chunk := response.read(8 * 1024**2):
                    output.write(chunk)
            temporary.rename(path)
    identity = validate_data(root)
    (root / "manifest.json").write_text(json.dumps({**lock, **identity}, indent=2) + "\n")
    return identity


def sandbox_settings(config):
    from ..runtime.resources import resource_settings
    return resource_settings(config, "lcb")


def preflight(data_path, config=None):
    config = config or {}
    sandbox_settings(config)
    if config.get("test_timeout_seconds", 6) != 6:
        raise ValueError("Pinned LiveCodeBench testcase timeout is 6 seconds")
    if config.get("release", "release_v6") != "release_v6":
        raise ValueError("Only complete release_v6 is accepted")
    concurrency = config.get("generation_concurrency", 4)
    if type(concurrency) is not int or not 1 <= concurrency <= 256:  # pooled over up to 8 server copies
        raise ValueError("LCB generation_concurrency must be between 1 and 256")
    return validate_data(data_path)


SANDBOX_ATTEMPTS = 3
SANDBOX_BACKOFF_SECONDS = (10, 30)  # between attempts; tests replace SLEEP
SLEEP = time.sleep


class HarborExecutor:
    """Fresh offline Harbor environment per problem; expected outputs stay here."""
    def __init__(self, config, *, resource_factory=None):
        from ..runtime.resources import grading_resource
        self.config = config
        self.resource_factory = resource_factory or grading_resource
        sandbox_settings(config)

    def execute(self, code, inputs, fn_name):
        lifetime = 60 + 6 * (len(inputs) + 1)
        if not 60 < lifetime <= 86400:
            raise ValueError("LCB problem exceeds configured resource lifetime")
        for attempt in range(SANDBOX_ATTEMPTS):
            try:
                return self._execute_once(code, inputs, fn_name, lifetime)
            except Exception:  # noqa: BLE001 - the sandbox layer, never the candidate (its errors come back as records)
                # The sandbox layer failed: provider start refused or rate-limited (on 2026-09-30 Modal answered 32
                # concurrent creations with "Could not claim the sandbox name" and RESOURCE_EXHAUSTED, errors that are
                # neither TimeoutError nor RuntimeError), hung exec, lost transfer. A fresh sandbox after a pause;
                # only then does the caller record the problem as an execution failure.
                if attempt + 1 == SANDBOX_ATTEMPTS:
                    raise
                SLEEP(SANDBOX_BACKOFF_SECONDS[attempt])

    def _execute_once(self, code, inputs, fn_name, lifetime):
        with self.resource_factory(self.config, "lcb", lifetime) as sandbox:
            with tempfile.TemporaryDirectory(prefix="lcb-inputs-") as directory:
                root = Path(directory)
                (root / "candidate.py").write_text(code)
                (root / "inputs.json").write_text(json.dumps({"inputs": inputs, "fn_name": fn_name, "timeout_seconds": 6}))
                sandbox.upload(root / "candidate.py", "/candidate.py")
                sandbox.upload(root / "inputs.json", "/inputs.json")
                sandbox.upload(Path(public.runner.__file__), "/runner.py")
            sandbox.exec("mkdir -p /tmp/candidate; chmod 700 /tmp/candidate; "
                         "chown 0:0 /candidate.py /inputs.json /runner.py; chmod 0444 /candidate.py /inputs.json /runner.py")
            # Keep large outputs off provider stdout channels, which may truncate.
            process = sandbox.exec("python3 -I /runner.py > /results.jsonl 2> /runner-error.log", timeout=lifetime-15,
                                   env={"FRONTIER_LCB_REMOTE": "1"}, check=False)
            if process.return_code != 0:
                error = sandbox.exec("tail -c 2000 /runner-error.log", check=False)
                raise RuntimeError(f"Remote LCB supervisor failed ({process.return_code}): " + (error.stdout or process.stderr or "")[-2000:])
            with tempfile.TemporaryDirectory(prefix="lcb-results-") as directory:
                result = Path(directory) / "results.jsonl"
                sandbox.download("/results.jsonl", result)
                records = [json.loads(line) for line in result.read_text().splitlines() if line.strip()]
            if any("error" in record for record in records):
                return []
            return records


def _evaluate_rows(transport, rows, executor, config):
    return public.evaluate_rows(transport, rows, executor, config, execution="offline_harbor_cpu")


def evaluate(transport, data_path, config=None):
    config = config or {}
    identity = preflight(data_path, config)
    result = _evaluate_rows(transport, iter_problems(data_path), HarborExecutor(config), config)
    if result["details"]["problems"] != 1055:
        raise ValueError("Incomplete LiveCodeBench release_v6 evaluation")
    result["details"]["source"] = identity
    return result


evaluate_livecodebench = evaluate


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download and verify full pinned LiveCodeBench release_v6")
    parser.add_argument("destination")
    args = parser.parse_args()
    print(json.dumps(prepare(args.destination), indent=2))
