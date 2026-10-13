"""Runs only inside the isolated checker image. One request, one fresh Lean process."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, "/checker")
from simplification_metric import MAX_PROOF_BYTES, MAX_STATEMENT_BYTES, proof_policy


def source_for(request):
    if not isinstance(request, dict) or set(request) != {"statement", "proof"}:
        raise ValueError("Invalid checker request")
    statement, proof = request["statement"], request["proof"]
    if not isinstance(statement, str) or len(statement.encode()) > MAX_STATEMENT_BYTES:
        raise ValueError("Invalid statement")
    if not isinstance(proof, str) or len(proof.encode()) > MAX_PROOF_BYTES:
        raise ValueError("Invalid proof")
    if problem := proof_policy(proof):
        return None, {"valid": False, "status": "invalid", "reason": problem}
    # Both fields are Lean string literals, never pasted as candidate commands.
    quoted = lambda s: json.dumps(s, ensure_ascii=False)
    source = ("import ProofCheck\nset_option maxHeartbeats 200000\n"
              "set_option maxRecDepth 1024\nopen BigOperators Real Nat Topology Rat\n"
              f"frontier_check {quoted(statement)} {quoted(proof)}\n")
    return source, None


def verdict(returncode, raw):
    if len(raw) > 1_048_576:
        return {"valid": False, "status": "invalid", "reason": "checker output limit"}
    messages = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        try:
            message = json.loads(line)
        except ValueError:
            return {"valid": False, "status": "invalid", "reason": "non-JSON compiler output"}
        if not isinstance(message, dict):
            return {"valid": False, "status": "invalid", "reason": "bad compiler message"}
        messages.append(message)
    valid = (returncode == 0
             and not any(m.get("severity") == "error" for m in messages)
             and sum(m.get("severity") == "information" and m.get("data") == "FRONTIER_PROOF_VALID"
                     for m in messages) == 1)
    errors = [m.get("data", "")[:1000] for m in messages if m.get("severity") == "error"][:2]
    return {"valid": valid, "status": "ok" if valid else "invalid", "errors": errors}


def check(request):
    source, early = source_for(request)
    if early:
        return early
    libraries = [Path("/checker"), Path("/opt/mathlib/.lake/build/lib")]
    libraries.extend(sorted(Path("/opt/mathlib/.lake/packages").glob("*/.lake/build/lib")))
    env = {"PATH": "/opt/lean/bin:/usr/bin:/bin", "LEAN_PATH": ":".join(map(str, libraries)),
           "HOME": "/tmp", "LANG": "C.UTF-8"}
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        path = Path(directory) / "Attempt.lean"
        path.write_text(source)
        with (Path(directory) / "messages").open("w+b") as log:
            try:
                proc = subprocess.run(["/opt/lean/bin/lean", "--json", str(path)], env=env,
                                      cwd=directory, stdout=log, stderr=subprocess.STDOUT, timeout=60)
            except subprocess.TimeoutExpired:
                return {"valid": False, "status": "timeout"}
            log.seek(0)
            raw = log.read(1_048_577)
        return verdict(proc.returncode, raw)


if __name__ == "__main__":
    try:
        raw = sys.stdin.buffer.read(600_001)
        if len(raw) > 600_000:
            raise ValueError("checker request byte limit")
        print(json.dumps(check(json.loads(raw))))
    except Exception as exc:
        print(json.dumps({"status": "infrastructure_error", "error": str(exc)}))
        sys.exit(2)
