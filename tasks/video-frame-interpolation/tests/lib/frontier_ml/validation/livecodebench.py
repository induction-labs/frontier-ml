"""LiveCodeBench v6 prompt, code extraction, testcase decoding, comparison and a local development runner.

The trusted verifier imports the pure functions here (prompt, extract_code, decode_private,
test_cases, decode_value, compare_stdio, compare_cases, evaluate_rows), so development and final
scoring share one implementation. Only the executor differs: the verifier runs every problem in a
fresh offline remote sandbox, while `LocalExecutor` runs it on this machine.

WARNING: `LocalExecutor` EXECUTES MODEL-GENERATED CODE ON YOUR OWN MACHINE. Each problem runs in a
separate Python subprocess with the same six-second compile and per-testcase watchdogs as the
final evaluation, plus best-effort resource limits (address space, file size, open files,
no-new-privileges, its own process group, a scratch working directory and a minimal
environment). It does not change user, drop network access or restrict the filesystem, which
needs root. Run it only in a disposable development sandbox that holds nothing you care about.

AST wrapping, import prelude and input mocking live in livecodebench_runner.py and adapt
LiveCodeBench testing_util.py, MIT Copyright (c) 2024 LiveCodeBench (LICENSE-livecodebench.txt).
"""
from __future__ import annotations

import base64
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import tempfile
import zlib

from . import livecodebench_runner as runner
from .errors import EvaluationError
from .text_benchmarks import completion_content

# Official release_v6 = test.jsonl through test6.jsonl, in this order.
FILES = ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl")
TEST_TIMEOUT_SECONDS = 6
SYSTEM_PROMPT = ("You are an expert Python programmer. You will be given a question (problem specification) "
                 "and will generate a correct Python program that matches the specification and passes all tests.")


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError("Dataset pickle globals are forbidden")
    def persistent_load(self, pid):
        raise pickle.UnpicklingError("Dataset pickle persistent references are forbidden")


def decode_private(value):
    try:
        tests = json.loads(value)
    except json.JSONDecodeError:
        decoded = zlib.decompress(base64.b64decode(value, validate=True))
        primitive = PrimitiveUnpickler(io.BytesIO(decoded)).load()
        if not isinstance(primitive, (str, bytes)):
            raise ValueError("Official private testcase pickle must contain only a JSON string")
        tests = json.loads(primitive)
    if not isinstance(tests, list):
        raise ValueError("Private tests must be a list")
    return tests


def test_cases(problem):
    tests = json.loads(problem["public_test_cases"]) + decode_private(problem["private_test_cases"])
    if not tests or any(not isinstance(t, dict) or not {"input", "output", "testtype"} <= t.keys() for t in tests):
        raise ValueError("Missing LiveCodeBench testcase fields")
    fn = json.loads(problem["metadata"]).get("func_name")
    expected_type = "functional" if fn else "stdin"
    if any(t["testtype"] != expected_type for t in tests):
        raise ValueError("Test type disagrees with official function metadata")
    return fn, tests


test_cases.__test__ = False  # not a pytest test


def iter_problems(root, files=FILES):
    root = Path(root)
    for name in files:
        with (root / name).open() as file:
            for line in file:
                if line.strip():
                    yield json.loads(line)


def extract_code(content):
    lines = content.split("\n")
    fences = [i for i, line in enumerate(lines) if "```" in line]
    return "\n".join(lines[fences[-2]+1:fences[-1]]) if len(fences) >= 2 else ""


def prompt(problem):
    text = "### Question:\n" + problem["question_content"] + "\n\n"
    if problem["starter_code"]:
        text += "### Format: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
        text += "```python\n" + problem["starter_code"] + "\n```\n\n"
    else:
        text += "### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
        text += "```python\n# YOUR CODE HERE\n```\n\n"
    text += "### Answer: (use the provided format with backticks)\n\n"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]


def decode_value(tagged):
    if not isinstance(tagged, list) or len(tagged) != 2:
        raise ValueError("Malformed functional output")
    kind, value = tagged
    if kind == "scalar" and (value is None or type(value) in (str, int, float, bool)):
        return value
    if kind in ("list", "tuple") and isinstance(value, list):
        values = [decode_value(v) for v in value]
        return tuple(values) if kind == "tuple" else values
    if kind == "dict" and isinstance(value, list):
        return {decode_value(k): decode_value(v) for k, v in value}
    raise ValueError("Unknown functional output type")


def compare_stdio(actual, expected):
    # Pinned upstream: trim lines, require equal line counts, then exact text or
    # exact Decimal token equality. There is no fuzzy float tolerance.
    if not isinstance(actual, str) or not isinstance(expected, str):
        return False
    left = [line.strip() for line in actual.strip().split("\n")]
    right = [line.strip() for line in expected.strip().split("\n")]
    if len(left) != len(right):
        return False
    for a, b in zip(left, right, strict=True):
        if a == b:
            continue
        try:
            if [Decimal(v) for v in a.split()] != [Decimal(v) for v in b.split()]:
                return False
        except (InvalidOperation, ValueError):
            return False
    return True


def compare_cases(tests, responses, fn_name):
    if len(tests) != len(responses):
        return False
    for index, (case, record) in enumerate(zip(tests, responses, strict=True)):
        if record.get("index") != index or not isinstance(record.get("result"), dict):
            return False
        response = record["result"]
        if response.get("ok") is not True:
            return False
        try:
            if fn_name:
                if decode_value(response["value"]) != json.loads(case["output"]):
                    return False
            elif not compare_stdio(response.get("stdout"), case["output"]):
                return False
        except (ValueError, TypeError, KeyError, RecursionError):
            return False
    return True


class LocalExecutor:
    """Run candidate code in a local subprocess per problem.

    WARNING: this executes model-generated code on the current machine (see the module docstring).
    Returns the same `[{"index", "result"}]` records as the trusted remote executor.
    """
    def __init__(self, timeout_seconds=TEST_TIMEOUT_SECONDS):
        self.timeout_seconds = timeout_seconds

    def execute(self, code, inputs, fn_name):
        with tempfile.TemporaryDirectory(prefix="lcb-local-") as directory:
            root = Path(directory)
            (root / "candidate.py").write_text(code)
            work = root / "work"
            work.mkdir(mode=0o700)
            process = subprocess.Popen(
                [sys.executable, "-I", runner.__file__, "local-worker", json.dumps(fn_name), str(root / "candidate.py")],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                cwd=work, start_new_session=True,  # own process group, killed as a whole below
                env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(work), "TMPDIR": str(work),
                     "PYTHONDONTWRITEBYTECODE": "1"},
            )
            os.set_blocking(process.stdin.fileno(), False)
            os.set_blocking(process.stdout.fileno(), False)
            records = []
            try:
                for line in runner.run_inputs(process, inputs, self.timeout_seconds):
                    records.append(json.loads(line))
            except Exception:
                # Same as the remote supervisor's error record: the problem fails.
                records = []
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                process.wait(timeout=10)
                for stream in (process.stdin, process.stdout):
                    try:
                        stream.close()
                    except OSError:
                        pass
            return records


# Execution infrastructure failures (a sandbox that never starts or hangs, a lost subprocess) score their problem
# as failed; above this share of the problems the evaluation is an infrastructure error, not a score.
EXECUTION_FAILURE_PERCENT = 5


def evaluate_rows(transport, rows, executor, config, *, execution):
    """Generate one solution per problem, execute it, and compare outputs here; same for both executors."""
    concurrency = config.get("generation_concurrency", 4)
    if type(concurrency) is not int or not 1 <= concurrency <= 256:  # pooled over up to 8 server copies
        raise ValueError("LCB generation_concurrency must be between 1 and 256")

    def solve(problem):
        response = transport.request("POST", "/v1/chat/completions",
            {"model": "submission", "messages": prompt(problem), "temperature": config.get("temperature", 1.0),
             "max_tokens": config.get("max_tokens", 32768), "stream": False},
            timeout=config.get("request_timeout_seconds", 3600))
        content = completion_content(response)
        code = extract_code(content)
        fn, tests = test_cases(problem)
        # Never send expected outputs, metadata labels, or full problem objects.
        execution_error = None
        try:
            outputs = executor.execute(code, [test["input"] for test in tests], fn) if code else []
        except Exception as error:  # noqa: BLE001 - the executor's own infrastructure (sandbox, subprocess), not the candidate
            # A candidate error inside the sandbox comes back as an error record; an exception here means the
            # execution infrastructure failed for this problem. Score the problem as failed and record it: one
            # hung sandbox out of a thousand must not zero a whole run. Too many of them fail the run closed below.
            outputs, execution_error = [], f"{type(error).__name__}: {str(error)[:300]}"
        passed = compare_cases(tests, outputs, fn)
        record = {"platform": problem["platform"], "question_id": problem["question_id"],
                  "passed": passed, "tests": len(tests), "code_sha256": hashlib.sha256(code.encode()).hexdigest()}
        if execution_error:
            record["execution_error"] = execution_error
        return record

    # Bound queued problem data as well as live GPU requests and CPU sandboxes.
    iterator = enumerate(rows)
    completed = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending = {}
        for _ in range(concurrency):
            item = next(iterator, None)
            if item is not None:
                index, row = item
                pending[pool.submit(solve, row)] = index
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                completed[pending.pop(future)] = future.result()
                item = next(iterator, None)
                if item is not None:
                    index, row = item
                    pending[pool.submit(solve, row)] = index
    records = [completed[i] for i in sorted(completed)]
    if not records:
        raise ValueError("LCB evaluation must not be empty")
    failed = [record for record in records if record.get("execution_error")]
    if len(failed) * 100 > EXECUTION_FAILURE_PERCENT * len(records):
        raise EvaluationError(f"Code execution infrastructure failed for {len(failed)} of {len(records)} problems, more than "
                              f"{EXECUTION_FAILURE_PERCENT}%; the submission was not scored. First errors: "
                              + " | ".join(f"{r['platform']}:{r['question_id']}: {r['execution_error']}" for r in failed[:3]))
    return {"score": sum(record["passed"] for record in records)/len(records), "n": len(records),
            "details": {"metric": "pass@1", "samples_per_problem": 1, "problems": len(records),
                        "records": records, "execution": execution, "test_timeout_seconds": TEST_TIMEOUT_SECONDS,
                        "generation_concurrency": concurrency,
                        "execution_failures": {"total": len(failed), "limit_percent": EXECUTION_FAILURE_PERCENT,
                                               "ids": [f"{r['platform']}:{r['question_id']}" for r in failed]}}}


def evaluate_local(transport, data_path, config, *, limit=None, executor=None):
    """Development pass@1 over the provided release_v6 files, executing code locally (see WARNING above)."""
    root = Path(data_path)
    missing = [name for name in FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"LiveCodeBench v6 files missing under {root}: {', '.join(missing)}")
    rows = iter_problems(root)
    if limit is not None:
        rows = (row for index, row in zip(range(limit), rows))
    return evaluate_rows(transport, rows, executor or LocalExecutor(), config, execution="local_subprocess")
