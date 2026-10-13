"""verified.jsonl -> candidates.jsonl in the shape draft_descriptions.py / validate_questions.py expect.

Keeps pages that passed cross-OCR verification and whose real screenshot the pinned detector reads
well (real-page layout F1 >= MIN_REAL_L), so every kept page is a fair evaluation item.
"""
import json
import sys

MIN_REAL_L = 0.85
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
kept = 0
with open(sys.argv[2], "w") as out:
    for r in rows:
        if r.get("status") != "ok" or r.get("real_L", 0) < MIN_REAL_L:
            continue
        out.write(json.dumps({"key": r["key"], "status": "ok", "domain": None, "video_id": r["video_id"],
                              "heuristic_type": r["heuristic_type"], "texts": r["requested"],
                              "ignore_regions": r["ignore_regions"], "real_T": r["real_T"], "real_L": r["real_L"],
                              "n_boxes": r["n_requested"], "n_chars": r["requested_chars"],
                              "requested_char_fraction": r["requested_char_fraction"],
                              "ignore_area": r["ignore_area"]}) + "\n")
        kept += 1
print(kept, "candidates of", len(rows))
