"""Send each split sample to a running server exactly as the task client does (build_request, one at a time),
validate each response with decode_response, and save <id>.png plus latency.json (resumable).

python http_client.py SPLIT_JSONL OUT_DIR --url http://127.0.0.1:18765 [--limit N]
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import screenshot_generation as sg  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split")
    parser.add_argument("out")
    parser.add_argument("--url", default="http://127.0.0.1:18765")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    rows = [json.loads(l) for l in open(args.split) if l.strip()][:args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "latencies.jsonl"
    done = {json.loads(l)["id"]: json.loads(l)["s"] for l in open(log)} if log.exists() else {}
    for i, row in enumerate(rows):
        if row["id"] in done and (out / f"{row['id']}.png").exists():
            continue
        path, payload = sg.build_request(row)
        request = urllib.request.Request(args.url + path, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=600) as response:
            body = json.loads(response.read())
        seconds = time.monotonic() - started
        (out / f"{row['id']}.png").write_bytes(sg.decode_response(body))
        done[row["id"]] = seconds
        with open(log, "a") as f:
            f.write(json.dumps({"id": row["id"], "s": seconds}) + "\n")
        print(f"{i + 1}/{len(rows)} {row['id']} {seconds:.2f}s", flush=True)
    latencies = sorted(done[r["id"]] for r in rows)
    stats = {"n": len(latencies), "mean_s": sum(latencies) / len(latencies),
             "p95_s": sg.percentile(latencies, 95), "max_s": latencies[-1]}
    (out / "latency.json").write_text(json.dumps(stats, indent=1))
    print(stats)


if __name__ == "__main__":
    main()
