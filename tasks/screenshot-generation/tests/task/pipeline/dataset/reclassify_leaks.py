"""Recompute the description-leak status of validated.jsonl with the current leaks() rule (no judge calls).

Usage: python reclassify_leaks.py CANDIDATES_JSONL VALIDATED_JSONL
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from validate_questions import MIN_PAIRS, leaks  # noqa: E402

processed = {json.loads(l)["key"]: json.loads(l) for l in open(sys.argv[1]) if l.strip()}
rows = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
for r in rows:
    if "description" not in r:
        continue
    p = processed[r["key"]]
    r["leaks"] = leaks(r["description"], p["texts"], p.get("domain"))
    r["status"] = "description_leak" if r["leaks"] else ("too_few_pairs" if len(r["questions"]) < 2 * MIN_PAIRS else "ok")
Path(sys.argv[2]).write_text("".join(json.dumps(r) + "\n" for r in rows))
print({s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})})
