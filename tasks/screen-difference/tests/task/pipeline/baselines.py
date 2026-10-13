"""Measure screen-difference normalization anchors under the live grader (reference_observation_coverage).

Scores fixed descriptions with the task's own score() and the verifier's gpt-6-luna judge on the 14 held-out
pairs x 5 rounds, exactly the graded shape. Floor = best input-blind template; ceiling = the human reference
observations passed through the judge as the candidate.

    uv run frontier fetch screen-difference --out /tmp/sd
    uv run --env-file .env python tasks/screen-difference/pipeline/baselines.py /tmp/sd/screen-difference/private/test.jsonl
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from frontier_ml.runtime.judge import judge_from_config
from frontier_ml.taskspec import load_task, sibling

task = load_task("screen-difference")
screen = sibling(task.path / "verifier.py", "validate.py")
config = dict(task.spec["verifier"]["config"])
config["judge"] = judge_from_config(config)
rows = [dict(r, id=r.get("id") or f"heldout-{i:04d}") for i, r in enumerate(map(json.loads, Path(sys.argv[1]).read_text().splitlines()))]

TEMPLATES = {
    "cursor": "The cursor moved.",
    "generic": ("The cursor moved to a different position. Some text was typed, edited or deleted. A menu, dropdown, "
                "dialog or panel opened or closed. The page or view scrolled. An element was highlighted, selected or "
                "hovered. The zoom level changed."),
    "exhaustive": ("The mouse cursor moved to a new position and now hovers over a different element, which is highlighted. "
                   "Text in a field or editor was typed, edited, selected or deleted, and the text cursor moved. "
                   "A menu, dropdown, context menu, tooltip, popup, dialog, sidebar or panel appeared or disappeared. "
                   "The page, list or document scrolled, so different content is visible. A button, tab, checkbox or item "
                   "was clicked, selected or toggled and its color or outline changed. A new tab, window or page opened, "
                   "or the view switched to another page. The zoom level, window size or layout changed. "
                   "Numbers, labels, a status indicator, a timestamp or a progress indicator changed. "
                   "An image, icon or video frame changed. Everything else stays the same."),
}

def measure(name, describe):
    predictions = [{"id": r["id"], "round": k, "description": describe(r)} for k in range(5) for r in rows]
    result = screen.score(rows, predictions, config)
    per_round = [sum(x["quality"] for x in result["details"]["records"] if x["round"] == k) / len(rows) for k in range(5)]
    print(json.dumps({"condition": name, "score": result["score"], "per_round": per_round}), flush=True)
    return {"condition": name, "score": result["score"], "per_round": per_round,
            "description": None if name == "reference" else describe(rows[0])}

out = [measure(n, lambda r, t=t: t) for n, t in TEMPLATES.items()]
out.append(measure("reference", lambda r: "\n".join(r["observations"])))
Path(__file__).with_name("baseline_results.json").write_text(json.dumps(
    {"judge_model": config["judge_model"], "pairs": len(rows), "rounds": 5, "data": sys.argv[1], "conditions": out}, indent=2) + "\n")
