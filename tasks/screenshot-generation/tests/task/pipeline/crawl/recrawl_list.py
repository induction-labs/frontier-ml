"""Domains whose pages were kept before imagery regions were recorded; recrawl them."""
import json
domains = []
for log in ("crawl_log.jsonl", "crawl_log_b.jsonl"):
    for line in open(log):
        r = json.loads(line)
        if any(v.startswith("ok:") for v in r["status"].values()):
            domains.append(r["domain"])
open("recrawl.txt", "w").write("\n".join(domains) + "\n")
print(len(domains))
