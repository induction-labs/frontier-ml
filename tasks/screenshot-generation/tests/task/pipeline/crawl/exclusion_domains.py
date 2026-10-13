"""Collect registrable domains that appear in public GUI datasets (ScreenParse v2 urls, Mind2Web websites).

Reads only the `url` column of each ScreenParse parquet file through HTTP range requests.
Output: exclusion_domains.txt (one registrable domain per line).
"""
from concurrent.futures import ThreadPoolExecutor
import sys
import urllib.parse

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

fs = HfFileSystem()


def registrable(host):
    host = (host or "").lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    parts = host.split(".")
    # Coarse eTLD+1: keep three labels for common two-level public suffixes.
    if len(parts) >= 3 and (parts[-2] in {"co", "com", "org", "net", "ac", "gov", "edu", "ne", "or"} and len(parts[-1]) == 2):
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def screenparse_file(name):
    for attempt in range(4):
        try:
            with fs.open(name, block_size=1 << 20) as f:
                urls = pq.ParquetFile(f).read(columns=["url"]).column("url").to_pylist()
            return {registrable(urllib.parse.urlsplit(u).hostname) for u in urls if u}
        except Exception as exc:  # retry transient HTTP errors
            error = exc
    print("failed", name, error, file=sys.stderr)
    return set()


def main():
    files = sorted(p for p in fs.glob("datasets/docling-project/screenparse/train/*.parquet"))
    domains = set()
    with ThreadPoolExecutor(24) as pool:
        for i, found in enumerate(pool.map(screenparse_file, files)):
            domains |= found
            if i % 100 == 0:
                print(i, len(files), len(domains), flush=True)
    screenparse = len(domains)
    for split in ("train",):
        try:
            from datasets import load_dataset
            m2w = load_dataset("osunlp/Mind2Web", split=split, columns=["website"])
            for site in set(m2w["website"]):
                domains.add(registrable(site if "." in site else site + ".com"))
        except Exception as exc:
            print("mind2web failed", exc, file=sys.stderr)
    with open("exclusion_domains.txt", "w") as out:
        out.write("\n".join(sorted(d for d in domains if d)) + "\n")
    print("screenparse", screenparse, "total", len(domains))


if __name__ == "__main__":
    main()
