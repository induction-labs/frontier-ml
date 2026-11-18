"""The cutoff proxy (frontier_ml/cutoff) as a load-balanced Modal deployment.

Two web endpoints in one app, named after the cutoff date:

- front: answers Hugging Face and OpenRouter itself and scales out with load (Modal balances requests across its
  containers). This is the URL runs are given.
- git: one container that holds the GitHub mirrors on its own disk; the front forwards the GitHub mounts to it, so
  a clone and its retries always meet the same mirror.

Every container sends its access-log lines to a Modal Queue; `sync` drains that into a local access.jsonl.

    export FML_CUTOFF_DATE=2025-11-17 FML_CUTOFF_PROXY_SECRET=...   # plus HF_TOKEN, FML_CUTOFF_GITHUB_TOKEN
    export FML_CUTOFF_HF_ALLOW=/path/allow.json                      # optional, from `frontier-cutoff-proxy allowlist`
    modal deploy frontier_ml/cutoff/modal_app.py                       # prints the front URL (…-front.modal.run)
    python frontier_ml/cutoff/modal_app.py sync --out DIR [--every 30] # append drained log lines to DIR/access.jsonl
                                                                     # (same FML_CUTOFF_DATE: it names the queue)
    modal app stop cutoff-proxy-20251117

A cloudflared quick tunnel in front of one local process caps at 200 in-flight requests, which a batch of runs
labelling data through OpenRouter exceeds; that is what this replaces. Mirrors and pin caches are per container and
are rebuilt on demand when Modal replaces one.
"""
import json
import os
from pathlib import Path
import threading
import time

import modal

PACKAGE = Path(__file__).resolve().parent  # frontier_ml/cutoff
# Read where `modal deploy` runs; inside the containers the same names come from the secret built below.
DATE = os.environ.get("FML_CUTOFF_DATE", "")
# FML_CUTOFF_APP names the deployment when two proxies for one date must coexist.
NAME = os.environ.get("FML_CUTOFF_APP") or ("cutoff-proxy-" + DATE.replace("-", "")[:8] if DATE else "cutoff-proxy")
FORWARDED = ("FML_CUTOFF_DATE", "FML_CUTOFF_PROXY_SECRET", "FML_CUTOFF_GITHUB_TOKEN", "FML_CUTOFF_OPENROUTER_UPSTREAM")

app = modal.App(NAME)
queue = modal.Queue.from_name(NAME + "-log", create_if_missing=True)

if modal.is_local():
    values = {key: os.environ[key] for key in FORWARDED if os.environ.get(key)}
    values["FML_CUTOFF_APP"] = NAME
    if os.environ.get("HF_TOKEN"):
        values["FML_CUTOFF_HF_TOKEN"] = os.environ["HF_TOKEN"]
    if os.environ.get("FML_CUTOFF_HF_ALLOW"):
        values["FML_CUTOFF_HF_ALLOW_JSON"] = Path(os.environ["FML_CUTOFF_HF_ALLOW"]).read_text()
    secrets = [modal.Secret.from_dict(values)]
else:
    secrets = [modal.Secret.from_dict({})]  # the same shape as at deploy time; the container already has the values

# Only the proxy package goes in: frontier_ml/__init__.py is a docstring and a version, the rest stays out.
image = (modal.Image.debian_slim(python_version="3.12").apt_install("git", "git-lfs")
         .pip_install("fastapi>=0.115,<1", "uvicorn>=0.30,<1", "httpx>=0.28,<1")
         .add_local_file(PACKAGE.parent / "__init__.py", "/root/frontier_ml/__init__.py", copy=True)
         .add_local_dir(PACKAGE, "/root/frontier_ml/cutoff", copy=True,
                        ignore=["__pycache__", "*.pyc"]))


class QueueLog:
    """Batch log lines and put one queue item every couple of seconds, so logging never waits on the network."""

    def __init__(self):
        self.lines, self.lock = [], threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def __call__(self, line):
        with self.lock:
            self.lines.append(line)

    def _run(self):
        while True:
            time.sleep(2)
            with self.lock:
                batch, self.lines = self.lines, []
            for start in range(0, len(batch), 2000):
                try:
                    queue.put("\n".join(batch[start:start + 2000]))
                except Exception as exc:  # a full or unreachable queue must not take the proxy down
                    print(f"access log batch dropped: {type(exc).__name__}: {exc}", flush=True)


def build(*, github_forward=None):
    from frontier_ml.cutoff.core import Settings
    from frontier_ml.cutoff.server import create_app
    env = os.environ
    allow = env.get("FML_CUTOFF_HF_ALLOW_JSON")
    settings = Settings(
        cutoff=env["FML_CUTOFF_DATE"], state_dir="/tmp/cutoff-proxy", secret=env["FML_CUTOFF_PROXY_SECRET"],
        hf_token=env.get("FML_CUTOFF_HF_TOKEN") or None, github_token=env.get("FML_CUTOFF_GITHUB_TOKEN") or None,
        github_max_kb=3 * 1024 * 1024, hf_allow=json.loads(allow) if allow else None,
        public_url=front.get_web_url(), github_forward=github_forward, log_sink=QueueLog(),
        **({"openrouter_upstream": env["FML_CUTOFF_OPENROUTER_UPSTREAM"]} if env.get("FML_CUTOFF_OPENROUTER_UPSTREAM") else {}))
    return create_app(settings)


@app.function(image=image, secrets=secrets, min_containers=1, max_containers=1, cpu=8, memory=32768, timeout=3600,
              scaledown_window=1200)
@modal.concurrent(max_inputs=400)
@modal.asgi_app(label=NAME + "-git")
def git():
    return build()


@app.function(image=image, secrets=secrets, min_containers=2, max_containers=16, cpu=2, memory=4096, timeout=3600,
              scaledown_window=600)
@modal.concurrent(max_inputs=300, target_inputs=120)
@modal.asgi_app(label=NAME + "-front")
def front():
    return build(github_forward=git.get_web_url())


def sync(out, every):
    """Drain the log queue into out/access.jsonl (run one of these per deployment; items are removed as read)."""
    path = Path(out) / "access.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        while batches := queue.get_many(100, block=False):
            with path.open("a") as stream:
                stream.write("".join(batch + "\n" for batch in batches))
        if not every:
            return
        time.sleep(every)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("sync")
    p.add_argument("--out", required=True)
    p.add_argument("--every", type=float, default=0, help="keep draining every N seconds")
    args = parser.parse_args()
    sync(args.out, args.every)
