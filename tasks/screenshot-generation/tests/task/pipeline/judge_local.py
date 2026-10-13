"""Operator-side judge for data building and baselines: gpt-6-luna through OpenRouter, with a spend ledger.

The key is read at runtime from $OPENROUTER_API_KEY or from the operator env file (never printed or logged).
Every call's token usage is appended to LEDGER; calls stop once the recorded spend reaches the cap.
The callable has the same signature as frontier_ml.runtime.judge.judge_from_config: judge(**payload) -> body.
"""
import json
import os
import threading
import urllib.request
from pathlib import Path

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-6-luna"
ENV_FILE = Path("/home/user/frontier-ml/conav/.env")
LEDGER = Path("/scratch/frontier-ml-workspace/work/screenshot-generation/judge_ledger.jsonl")
CAP_USD = float(os.environ.get("JUDGE_CAP_USD", "25"))
PRICE = {"prompt": 0.10 / 1e6, "completion": 0.50 / 1e6}  # OpenRouter list price for openai/gpt-6-luna
_lock = threading.Lock()


def _key():
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    for line in ENV_FILE.read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("OPENROUTER_API_KEY not found")


def spent():
    if not LEDGER.exists():
        return 0.0
    total = 0.0
    for line in LEDGER.read_text().splitlines():
        if line.strip():
            total += json.loads(line)["usd"]
    return total


def make_judge(tag="run"):
    key = _key()
    state = {"spent": spent()}

    def judge(**payload):
        payload = dict(payload)
        payload["model"] = MODEL
        payload["usage"] = {"include": True}
        with _lock:
            if state["spent"] >= CAP_USD:
                raise RuntimeError(f"judge spend cap ${CAP_USD} reached")
        request = urllib.request.Request(ENDPOINT, json.dumps(payload).encode(),
                                         {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.load(response)
        usage = body.get("usage") or {}
        usd = usage.get("cost")
        if not isinstance(usd, (int, float)):
            usd = PRICE["prompt"] * usage.get("prompt_tokens", 0) + PRICE["completion"] * usage.get("completion_tokens", 0)
        with _lock:
            state["spent"] += usd
            with LEDGER.open("a") as out:
                out.write(json.dumps({"tag": tag, "usd": usd, "prompt_tokens": usage.get("prompt_tokens"),
                                      "completion_tokens": usage.get("completion_tokens")}) + "\n")
        return body
    return judge
