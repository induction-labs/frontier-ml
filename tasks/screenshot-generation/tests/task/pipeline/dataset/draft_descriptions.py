"""VLM-draft a text-free description and contrastive yes/no question pairs for each candidate page.

Drafting model: Qwen/Qwen3.6-27B-FP8 (local, vLLM). It sees the real screenshot and the list of visible
texts (so that it can avoid quoting them). Output per page: description + 7 question pairs.
The human-fix pass is pending (see review_tool.py).

Usage: python draft_descriptions.py processed.jsonl drafts.jsonl [--limit N]
"""
import argparse
import base64
import json
import re
from pathlib import Path

DRAFTER = "/scratch/hf_home/hub/models--Qwen--Qwen3.6-27B-FP8/snapshots/e89b16ebf1988b3d6befa7de50abc2d76f26eb09"
RAW = Path("/scratch/frontier-ml-workspace/work/screenshot-generation/crawl/raw")

PROMPT = """You are writing the specification for an image generator that must recreate this web page screenshot (1280x720 desktop viewport).
The generator separately receives every visible text string with its exact bounding box, so your job is ONLY the visual design and layout.

Visible text strings on the page (for reference only; NEVER quote, paraphrase, translate or spell out any of them):
{texts}

Write JSON with these keys:
- "unsafe": true if the page is sexual, violent, hateful, a gambling site, a login-wall/captcha/error/parked-domain page, or mostly blank; else false.
- "page_type": a few words (e.g. "restaurant homepage", "SaaS product landing page", "news article").
- "description": 4 to 7 sentences describing what the page LOOKS like: overall kind of page; background and color scheme; header/navigation bar layout and colors; the main sections and where they sit (left/right/top/bottom, columns, cards, hero banner); imagery (photos, illustrations, icons, logos) and what they depict; buttons (shape, color, fill); typography (serif/sans-serif, weight, relative sizes, text colors). Refer to text only generically ("a large bold headline", "a row of five navigation links", "a paragraph of body text"). No quoted words, no brand, product, person or place names, no numbers, prices, dates or URLs taken from the page.
- "pairs": exactly 7 objects {{"yes": ..., "no": ...}}. "yes" is a yes/no question whose correct answer for this screenshot is YES and which follows from your description; "no" is a minimally edited version of the same question whose correct answer is NO because it contradicts the description (change a color, position, presence, shape or style). Ask about clearly visible, unambiguous visual facts: colors of large regions, header/background style, position of elements, presence of photos/illustrations/buttons/cards/columns, dark vs light theme, image subjects. Do not ask about text content, exact small counts above 3, fonts sizes in pixels, or things a viewer could argue about.

Return only the JSON object."""

SCREEN_PROMPT = """You are writing the specification for an image generator that must recreate this screenshot of a computer screen (1280x720). It may show a web page in a browser, a desktop application, a code editor or terminal, a mobile screen, or a whole desktop.
The generator separately receives the exact text strings with their bounding boxes, so your job is ONLY the visual design and layout.

Visible text strings on the screen (for reference only; NEVER quote, paraphrase, translate or spell out any of them):
{texts}

Write JSON with these keys:
- "unsafe": true if the screen is sexual, violent or hateful, shows a person's face or webcam overlay, is mostly a video/photo/game scene rather than a user interface, is heavily blurred, or is mostly blank; else false.
- "screen_type": exactly one of "web_page" (a web site or web app, with or without browser chrome), "desktop_app" (a native application window, file manager, office software, settings, or the desktop itself), "ide_terminal" (a code editor, IDE, terminal or console), "mobile" (a phone or tablet screen or emulator), "other".
- "page_type": a few words (e.g. "spreadsheet application", "cloud console dashboard", "Python IDE with terminal").
- "description": 4 to 7 sentences describing what the screen LOOKS like: kind of screen; operating-system or browser chrome (title bars, tabs, address bar, taskbar/dock) if visible; background and color scheme (light/dark theme); layout of panels, sidebars, toolbars, menus, tables, editors, dialogs and where they sit (left/right/top/bottom, columns); icons and imagery and what they depict; buttons and controls (shape, color); typography (monospace/sans-serif, weight, relative sizes, text colors). Refer to text only generically ("a column of file names", "a toolbar of small labels", "a block of code"). No quoted words, no brand, product, application, person or place names, no numbers, dates or URLs.
- "pairs": exactly 7 objects {{"yes": ..., "no": ...}}. "yes" is a yes/no question whose correct answer for this screenshot is YES and which follows from your description; "no" is a minimally edited version of the same question whose correct answer is NO because it contradicts the description (change a color, position, presence, shape or style). Ask about clearly visible, unambiguous visual facts: light vs dark theme, colors of large regions, presence and position of sidebars, toolbars, tabs, taskbars, dialogs, tables, code editors, images, buttons. Do not ask about text content, application or brand names, exact small counts above 3, font sizes in pixels, or things a viewer could argue about.

Return only the JSON object."""


def data_url(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def parse(text):
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no json")
    obj = json.loads(m.group(0))
    if not isinstance(obj.get("description"), str) or not isinstance(obj.get("pairs"), list):
        raise ValueError("bad schema")
    pairs = [p for p in obj["pairs"] if isinstance(p, dict) and isinstance(p.get("yes"), str) and isinstance(p.get("no"), str)
             and p["yes"].strip() and p["no"].strip() and p["yes"].strip() != p["no"].strip()]
    obj["pairs"] = pairs
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("processed")
    parser.add_argument("out")
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--status", default="ok")
    parser.add_argument("--raw", default=str(RAW), help="directory of <key>.png screenshots")
    parser.add_argument("--kind", choices=["web", "screen"], default="web", help="prompt for web pages or any screen")
    args = parser.parse_args()
    done = set()
    if Path(args.out).exists():
        done = {json.loads(l)["key"] for l in open(args.out) if l.strip()}
    rows = [json.loads(l) for l in open(args.processed) if l.strip()]
    rows = [r for r in rows if r["status"] == args.status and r["key"] not in done][:args.limit]
    print(len(rows), "pages to draft", flush=True)
    if not rows:
        return
    from vllm import LLM, SamplingParams
    llm = LLM(model=DRAFTER, max_model_len=16384, gpu_memory_utilization=float(__import__("os").environ.get("DRAFT_GPU_UTIL", "0.85")), limit_mm_per_prompt={"image": 1},
              mm_processor_kwargs={"max_pixels": 1280 * 720}, seed=0,
              max_num_seqs=int(__import__("os").environ.get("DRAFT_MAX_SEQS", "256")))
    params = SamplingParams(max_tokens=2500, temperature=0.3, top_p=0.9, seed=0)
    batch = 64
    with open(args.out, "a") as out:
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]
            convs = []
            for r in chunk:
                texts = "\n".join(f"- {t['text']}" for t in r["texts"][:150])
                convs.append([{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url(Path(args.raw) / f"{r['key']}.png")}},
                    {"type": "text", "text": (SCREEN_PROMPT if args.kind == "screen" else PROMPT).format(texts=texts)}]}])
            outputs = llm.chat(convs, params, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
            for r, o in zip(chunk, outputs):
                text = o.outputs[0].text
                try:
                    obj = parse(text)
                    status = "ok"
                except Exception as error:
                    obj, status = {"raw": text[-2000:]}, f"parse_error:{error}"
                out.write(json.dumps({"key": r["key"], "status": status, "drafter": "Qwen/Qwen3.6-27B-FP8@e89b16eb", **obj}) + "\n")
            out.flush()
            print(start + len(chunk), "drafted", flush=True)


if __name__ == "__main__":
    main()
