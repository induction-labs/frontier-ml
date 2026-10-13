"""Prepare/restore trusted metric weights. Stdlib only; never executed by candidates."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request

PROTOCOL = "vfi-lpips-alex01-flolpips-sequence-v1"
ASSETS = {"alexnet", "lpips_alex", "pwcnet"}


def download(url, path, expected=None):
    if not url.startswith("https://"):
        raise ValueError("Metric weights require HTTPS")
    digest, size = hashlib.sha256(), 0
    with urllib.request.urlopen(url, timeout=60) as source, path.open("xb") as output:
        if not source.geturl().startswith("https://"):
            raise ValueError("Insecure metric download redirect")
        while block := source.read(1024 * 1024):
            size += len(block)
            if size > 512 * 1024**2:
                raise ValueError("Metric asset exceeds 512 MiB")
            digest.update(block)
            output.write(block)
    actual = digest.hexdigest()
    if not size or (expected is not None and actual != expected):
        raise ValueError("Metric download is empty or fails its frozen checksum")
    return {"path": path.name, "url": url, "sha256": actual, "bytes": size}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="first acquisition: inspect and commit the resulting lock")
    fetch.add_argument("--lpips-revision", required=True, help="full PerceptualSimilarity Git commit")
    restore = commands.add_parser("restore", help="restore only bytes matching an existing lock")
    restore.add_argument("--lock", required=True)
    for subparser in (fetch, restore):
        subparser.add_argument("--output", required=True, help="new directory")
    args = parser.parse_args(argv)
    if args.command == "fetch":
        if not re.fullmatch(r"[0-9a-f]{40}", args.lpips_revision):
            parser.error("--lpips-revision must be a full lowercase Git SHA")
        urls = {"alexnet": "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
                "lpips_alex": f"https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/{args.lpips_revision}/lpips/weights/v0.1/alex.pth",
                "pwcnet": "https://content.sniklaus.com/github/pytorch-pwc/network-default.pytorch"}
        lock = {"protocol": PROTOCOL, "files": {key: {"url": url, "path": key + ".pth"} for key, url in urls.items()}}
    else:
        lock = json.loads(Path(args.lock).read_text())
        if lock.get("protocol") != PROTOCOL or set(lock.get("files", {})) != ASSETS:
            parser.error("Metric lock is not configured; freeze all three official weight files first")
        for record in lock["files"].values():
            if not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", "")):
                parser.error("Every metric asset requires a full SHA-256")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    for name, record in lock["files"].items():
        if Path(record["path"]).name != record["path"]:
            raise ValueError("Metric paths must be simple filenames")
        result = download(record["url"], output / record["path"], record.get("sha256"))
        if args.command == "restore" and result["bytes"] != record["bytes"]:
            raise ValueError("Metric asset size mismatch")
        lock["files"][name] = result
    if args.command == "restore":
        (output / "lock.json").write_bytes(Path(args.lock).read_bytes())
    else:
        (output / "lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"bundle": str(output), "metric_bundle_sha256": hashlib.sha256((output / "lock.json").read_bytes()).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
