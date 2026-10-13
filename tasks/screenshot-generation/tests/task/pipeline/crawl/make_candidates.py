"""Candidate domains: Tranco (list 8P4GV, fetched 2026-09-22) ranks > 10,000, English-leaning TLDs,
minus domains found in public GUI datasets. Shuffled with a fixed private seed."""
import random
import re
import sys
from pathlib import Path

TLD = re.compile(r"\.(com|org|net|io|co|us|uk|ca|au|nz|ie|info|app|dev|blog|shop|store|online|site|tech|ai|me)$")
excluded = set(Path("exclusion_domains.txt").read_text().split()) if Path("exclusion_domains.txt").exists() else set()
domains = []
for line in open("top-1m.csv"):
    rank, domain = line.strip().split(",", 1)
    if int(rank) > 10000 and TLD.search(domain):
        domains.append(domain)
random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 20260922).shuffle(domains)
kept = [d for d in domains if d not in excluded and (d[4:] if d.startswith("www.") else d) not in excluded]
Path("candidates.txt").write_text("\n".join(kept) + "\n")
print(len(domains), "candidates;", len(excluded), "excluded domains;", len(kept), "kept")
