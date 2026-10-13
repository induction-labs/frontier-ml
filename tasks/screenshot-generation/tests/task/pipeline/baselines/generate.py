"""Generate baseline images for a split: python generate.py METHOD SPLIT_JSONL OUT_DIR [--limit N]"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import methods  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method")
    parser.add_argument("split")
    parser.add_argument("out")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    rows = [json.loads(l) for l in open(args.split) if l.strip()][:args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.split).parent
    if args.method == "real":
        generator = methods.Real(root)
    else:
        generator = methods.make(args.method)
    if args.method in ("zimage", "flux2klein"):  # warm up kernels once
        generator(rows[0]["description"], rows[0]["texts"])
    latencies = []
    for row in rows:
        started = time.perf_counter()
        if args.method == "real":
            image = generator(row["description"], row["texts"], sample=row)
        else:
            image = generator(row["description"], row["texts"])
        latencies.append(time.perf_counter() - started)
        image.save(out / f"{row['id']}.png")
    latencies.sort()
    stats = {"method": args.method, "n": len(rows), "mean_s": sum(latencies) / len(latencies),
             "p95_s": latencies[max(0, int(0.95 * len(latencies) + 0.999) - 1)]}
    (out / "latency.json").write_text(json.dumps(stats, indent=1))
    print(stats)


if __name__ == "__main__":
    main()
