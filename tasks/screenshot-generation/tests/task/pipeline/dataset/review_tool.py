"""Human-fix pass (PENDING): build a static review page and apply reviewers' corrections.

    python review_tool.py build DATA_DIR/public/validation.jsonl review-val.html
    (open the HTML, edit descriptions / questions, mark rejects, click "Export" -> corrections.json)
    python review_tool.py apply DATA_DIR/public/validation.jsonl corrections.json

Rules for reviewers (also shown on the page): the description must describe only visual design and
layout, contain no literal page text, brand names, numbers or URLs, and every statement must be true
of the screenshot. Each question pair must be answerable from the description and from the image,
with the "yes" question true and the "no" question false. Remove pairs you are unsure about; a
sample needs >= 4 pairs. Applying corrections records reviewer and time; the manifest must then be
regenerated.
"""
import base64
import html
import json
import sys
import time
from pathlib import Path

PAGE = """<!doctype html><meta charset=utf-8><title>screenshot-generation review</title>
<style>body{font:14px sans-serif;margin:20px} .s{border:1px solid #ccc;margin:0 0 30px;padding:10px}
img{width:960px;border:1px solid #999} textarea{width:960px;height:90px} input.q{width:700px}</style>
<h1>Human review (descriptions + question pairs)</h1><p>%s</p>
<p>Reviewer: <input id=reviewer> <button onclick="exportAll()">Export corrections.json</button></p>%s
<script>
function exportAll(){const out={reviewer:document.getElementById('reviewer').value,samples:{}};
for(const s of document.querySelectorAll('.s')){const id=s.dataset.id;const pairs=[];
 for(const p of s.querySelectorAll('.pair')){if(p.querySelector('.drop').checked)continue;
  pairs.push({yes:p.querySelector('.y').value,no:p.querySelector('.n').value});}
 out.samples[id]={reject:s.querySelector('.reject').checked,description:s.querySelector('textarea').value,pairs};}
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)]));
a.download='corrections.json';a.click();}
</script>"""


def build(split, out):
    root = Path(split).parent
    blocks = []
    for line in open(split):
        row = json.loads(line)
        data = base64.b64encode((root / row["image"]).read_bytes()).decode()
        pairs = {}
        for q in row["questions"]:
            pairs.setdefault(q["pair"], {})[q["answer"]] = q["question"]
        pair_html = "".join(
            f"<div class=pair>yes: <input class='q y' value=\"{html.escape(p['yes'])}\"><br>no:&nbsp; "
            f"<input class='q n' value=\"{html.escape(p['no'])}\"> <label><input type=checkbox class=drop> drop</label></div>"
            for _, p in sorted(pairs.items()))
        blocks.append(f"<div class=s data-id=\"{row['id']}\"><h3>{row['id']} <label><input type=checkbox class=reject> reject sample</label></h3>"
                      f"<img src=\"data:image/png;base64,{data}\"><br><textarea>{html.escape(row['description'])}</textarea>{pair_html}</div>")
    Path(out).write_text(PAGE % (html.escape(__doc__.split("Rules for reviewers")[1]), "".join(blocks)))
    print("wrote", out, len(blocks), "samples")


def apply(split, corrections):
    fixes = json.loads(Path(corrections).read_text())
    rows, kept = [json.loads(l) for l in open(split)], []
    for row in rows:
        fix = fixes["samples"].get(row["id"])
        if fix is None:
            raise SystemExit(f"{row['id']} was not reviewed")
        if fix["reject"] or len(fix["pairs"]) < 4:
            continue
        row["description"] = fix["description"].strip()
        row["questions"] = [{"question": q, "answer": a, "pair": i} for i, p in enumerate(fix["pairs"][:6])
                            for q, a in ((p["yes"], "yes"), (p["no"], "no"))]
        kept.append(row)
    Path(split).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    log = Path(split).with_name("human_review_log.jsonl")
    with open(log, "a") as out:
        out.write(json.dumps({"split": Path(split).name, "reviewer": fixes.get("reviewer"), "time": time.time(),
                              "kept": len(kept), "rejected": len(rows) - len(kept)}) + "\n")
    print(f"kept {len(kept)} / {len(rows)}; re-run the manifest")


if __name__ == "__main__":
    {"build": build, "apply": apply}[sys.argv[1]](*sys.argv[2:])
