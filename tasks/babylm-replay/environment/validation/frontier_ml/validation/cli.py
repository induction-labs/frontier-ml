"""Command line and diagnostic reports shared by task validation scripts."""
import argparse
import json
import os
from pathlib import Path
import sys
import urllib.parse
import urllib.request
import uuid
from .http import HTTPTransport


def broker_judge(base_url, key, model, *, timeout=120, opener=None):
    """Use an agent-scoped proxy key; one paid request per unique idempotency key."""
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost", "::1")):
        raise ValueError("Judge proxy must use HTTPS (HTTP is allowed only for loopback)")
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError("Judge proxy URL cannot contain credentials, query or fragment")
    if not key or not model:
        raise ValueError("Scoped judge key and explicit judge deployment ID are required")
    endpoint = base_url.rstrip("/") + "/chat/completions"
    opener = opener or urllib.request.urlopen

    def judge(**payload):
        if payload.get("model") != model:
            raise ValueError("Judge model differs from explicitly selected deployment")
        request = urllib.request.Request(endpoint, json.dumps(payload).encode(),
                                         {"Authorization": "Bearer " + key,
                                          "Content-Type": "application/json",
                                          "Idempotency-Key": "validation-" + uuid.uuid4().hex})
        # No automatic retries: an uncertain paid response is surfaced to the agent.
        with opener(request, timeout=timeout) as response:
            return json.load(response)
    return judge


def envelope(task, result):
    """Diagnostic report: the score is informational and certifies no gate."""
    return {"task": task, "mode": "public_validation", "official_score": False,
            "architecture_verified": False, "offline_isolation_verified": False,
            "diagnostic_score": result.pop("score"), **result}


def emit(result, output=None):
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
    sys.stdout.write(rendered)


def public_main(task, run, argv=None, *, default_data, description, judge=False,
                default_judge_model=None, extra_arguments=None, extra_config=None):
    """Command line shared by every seed's validate.py."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server-url", required=True, help="HTTP(S) origin of your running server, without /v1")
    parser.add_argument("--data", default=default_data, help=f"public evaluation input (default {default_data})")
    parser.add_argument("--concurrency", type=int, default=32, help="concurrent requests")
    parser.add_argument("--timeout-seconds", type=float, default=3600, help="whole-batch and per-request ceiling")
    parser.add_argument("--output", help="also write the JSON report here")
    if judge:
        parser.add_argument("--judge-base-url", help="your budgeted OpenRouter-compatible base_url, including /v1; omit to get raw predictions")
        parser.add_argument("--judge-model", help=(f"judge deployment ID (default {default_judge_model})"
                                                if default_judge_model else "explicit judge deployment ID; no substitution"))
        parser.add_argument("--judge-key-env", default="MLE_JUDGE_API_KEY", help="environment variable holding the judge key")
    if extra_arguments:
        extra_arguments(parser)
    args = parser.parse_args(argv)
    try:
        config = {"concurrency": args.concurrency, "inference_timeout_seconds": args.timeout_seconds,
                  "request_timeout_seconds": args.timeout_seconds, "validation_mode": True, "task_slug": task}
        if judge and args.judge_base_url:
            model = args.judge_model or default_judge_model
            config["judge"] = broker_judge(args.judge_base_url, os.environ.get(args.judge_key_env), model)
            config["judge_model"] = model
        elif judge and args.judge_model:
            raise ValueError("--judge-model requires --judge-base-url")
        if extra_config:
            config.update(extra_config(args))
        result = envelope(task, run(HTTPTransport(args.server_url), args.data, config))
        code = 0
    except Exception as exc:
        result = {"task": task, "mode": "public_validation", "official_score": False,
                  "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        code = 1
    emit(result, args.output)
    return code
