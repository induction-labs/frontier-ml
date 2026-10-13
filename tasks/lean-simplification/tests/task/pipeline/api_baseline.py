"""Organizer-only GPT-6 Astra baseline; paid requests require authorization.

One response per public proof, with no tools, compiler feedback, or retries.
Credentials are read in-process and never written to reports. Checkpoints permit
resuming completed requests; an interrupted in-flight request needs inspection
before rerunning, because it may already have been billed.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import time
import urllib.error
import urllib.request

from baseline import PROMPT, extract
from simplification_client import CheckerClient
from validate import score as official_score

MODEL = "gpt-6-astra"
PRICING = {"input_per_million": 10.0, "cached_input_per_million": 1.0,
           "cache_write_per_million": 12.5,
           "output_per_million": 50.0,
           "source": "https://developers.openai.com/api/docs/models/gpt-6-astra",
           "checked_date": "2026-09-24"}


def read_key(path):
    for line in path.read_text().splitlines():
        fields = shlex.split(line, comments=True)
        if fields and fields[0] == "export":
            fields = fields[1:]
        assignment = " ".join(fields)
        if "=" in assignment:
            name, value = assignment.split("=", 1)
            if name.strip() == "OPENAI_API_KEY" and value.strip():
                return value.strip()
    raise ValueError("OPENAI_API_KEY is absent from the supplied credential file")


def cost(usage):
    inputs = usage["input_tokens"]
    cached = usage.get("input_tokens_details", {}).get("cached_tokens", 0)
    written = usage.get("input_tokens_details", {}).get("cache_write_tokens", 0)
    return ((inputs - cached - written) * PRICING["input_per_million"]
            + cached * PRICING["cached_input_per_million"]
            + written * PRICING["cache_write_per_million"]
            + usage["output_tokens"] * PRICING["output_per_million"]) / 1e6


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def request_one(row, key, settings):
    body = {"model": MODEL, "input": PROMPT.format(**row), "store": False,
            "reasoning": {"effort": settings["reasoning_effort"]},
            "max_output_tokens": settings["max_output_tokens"], "service_tier": "default"}
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses", json.dumps(body).encode(),
        {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        # Do not echo response messages: authentication errors may quote secrets.
        return {"id": row["id"], "api_error": {"http_status": exc.code},
                "outputs": [{"proof": "", "raw": "", "finish_reason": "api_error"}],
                "elapsed_seconds": time.monotonic() - started}
    except Exception as exc:
        return {"id": row["id"], "api_error": {"type": type(exc).__name__},
                "outputs": [{"proof": "", "raw": "", "finish_reason": "api_error"}],
                "elapsed_seconds": time.monotonic() - started}
    text = "".join(part.get("text", "") for item in result.get("output", [])
                   if item.get("type") == "message" for part in item.get("content", [])
                   if part.get("type") == "output_text")
    usage = result.get("usage")
    return {"id": row["id"], "response_id": result.get("id"),
            "response_model": result.get("model"), "service_tier": result.get("service_tier"),
            "status": result.get("status"), "incomplete_details": result.get("incomplete_details"),
            "usage": usage, "estimated_cost_usd": cost(usage) if usage else None,
            "elapsed_seconds": time.monotonic() - started,
            "outputs": [{"proof": extract(text, row), "raw": text,
                         "finish_reason": result.get("status")}]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["generate", "score"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--checker-url")
    parser.add_argument("--checker-fingerprint")
    parser.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--budget-usd", type=float, default=50)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, help="Issue at most this many new requests, then checkpoint")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.data.read_text().splitlines()]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Expected nonempty data with unique IDs")
    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if args.mode == "score":
        generated = json.loads(args.predictions.read_text())
        if generated["data_sha256"] != digest:
            raise ValueError("Predictions are for a different dataset")
        if [r["id"] for r in generated["records"]] != [r["id"] for r in rows]:
            raise ValueError("Expected exactly one ordered response per input")
        if any(r.get("api_error") or not r.get("usage") for r in generated["records"]):
            raise ValueError("API failures must be resolved explicitly before reporting a score")
        predictions = [{"id": r["id"], "proof": r["outputs"][0]["proof"]}
                       for r in generated["records"]]
        checker = CheckerClient(args.checker_url, args.checker_fingerprint)
        result = official_score(rows, predictions, {"proof_checker": checker})
        result.update(model=MODEL, settings=generated["settings"], checker=checker.health,
                      data_sha256=digest, estimated_cost_usd=generated["estimated_cost_usd"],
                      generation_seconds=generated["generation_seconds"], pricing=PRICING)
        save(args.output, result)
        print(json.dumps({k: v for k, v in result.items() if k != "details"}))
        print(json.dumps({k: v for k, v in result["details"].items() if k != "records"}))
        return
    if not args.env_file or args.workers < 1 or args.max_output_tokens < 1 or args.budget_usd <= 0:
        raise ValueError("Credential file and positive request/budget parameters are required")
    # UTF-8 byte count plus generous framing overhead is a conservative input-token
    # bound. No tools, retries, fast tier, or >272K-token contexts are used.
    bound = sum((len(PROMPT.format(**row).encode()) + 4096) * 12.5 / 1e6
                + args.max_output_tokens * PRICING["output_per_million"] / 1e6 for row in rows)
    if bound > args.budget_usd:
        raise ValueError(f"Conservative full-run cost bound ${bound:.2f} exceeds budget")
    settings = {"reasoning_effort": args.effort, "max_output_tokens": args.max_output_tokens,
                "samples_per_proof": 1, "tools": [], "compiler_feedback": False,
                "retries": 0, "service_tier": "default"}
    if args.resume:
        report = json.loads(args.output.read_text())
        if (report["data_sha256"] != digest or report["settings"] != settings
                or report["prompt"] != PROMPT or report["model"] != MODEL):
            raise ValueError("Cannot resume a different dataset, model, prompt, or configuration")
        done = {r["id"] for r in report["records"]}
        if set(report["started_ids"]) != done:
            raise ValueError("Interrupted in-flight requests need inspection before resuming")
    else:
        if args.output.exists():
            raise ValueError("Output exists; use --resume instead of issuing duplicate requests")
        report = {"model": MODEL, "data_sha256": digest, "prompt": PROMPT,
                  "created_at": datetime.now(timezone.utc).isoformat(), "settings": settings,
                  "pricing": PRICING, "conservative_cost_bound_usd": bound,
                  "generation_seconds": 0.0, "estimated_cost_usd": 0.0,
                  "started_ids": [], "records": []}
        done = set()
    key = read_key(args.env_file)
    pending = [row for row in rows if row["id"] not in done]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        pending = pending[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, report)
    print(json.dumps({"model": MODEL, "new_requests": len(pending),
                      "conservative_full_run_bound_usd": bound, "settings": settings}), flush=True)
    started = time.monotonic()
    prior_seconds = report["generation_seconds"]
    order = {row["id"]: i for i, row in enumerate(rows)}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        iterator = iter(pending)
        while True:
            while len(futures) < args.workers:
                row = next(iterator, None)
                if row is None:
                    break
                report["started_ids"].append(row["id"])
                save(args.output, report)
                futures[pool.submit(request_one, row, key, settings)] = row["id"]
            if not futures:
                break
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                record = future.result()
                del futures[future]
                report["records"].append(record)
                report["records"].sort(key=lambda r: order[r["id"]])
                report["estimated_cost_usd"] = sum(r.get("estimated_cost_usd") or 0 for r in report["records"])
                report["generation_seconds"] = prior_seconds + time.monotonic() - started
                save(args.output, report)
                print(json.dumps({"completed": len(report["records"]), "total": len(rows),
                                  "status": record.get("status", "api_error"),
                                  "estimated_cost_usd": report["estimated_cost_usd"]}), flush=True)


if __name__ == "__main__":
    main()
