"""Crawl fresh web pages at 1280x720 DPR1 and record visible DOM text boxes.

One homepage per domain (plus, for a fraction of domains, one same-site link found on it).
robots.txt is honoured for both the generic agent and our bot token. Output per page:
raw/<key>.png (viewport screenshot) and raw/<key>.json (url, boxes, stats, capture time).

Usage: python crawl.py --domains candidates.txt --out raw --workers 16 --limit 4000
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
import urllib.robotparser
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

BOT = "FrontierMLEResearchBot"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/140.0.0.0 Safari/537.36")
JS = (Path(__file__).with_name("extract_boxes.js")).read_text()
BAD_TITLE = re.compile(r"(just a moment|attention required|access denied|forbidden|403|404|not found|error|"
                       r"domain (is )?for sale|buy this domain|parked|under construction|coming soon|"
                       r"verify you are human|captcha|suspended|index of /|default page|welcome to nginx|"
                       r"it works|apache2|site not found|page not found|maintenance)", re.I)
BAD_TEXT = re.compile(r"(this domain (name )?(is|may be) for sale|domain has expired|checking your browser|"
                      r"enable javascript and cookies|verify you are human|access to this page has been denied|"
                      r"porn|xxx|escort|nude|sex cam|onlyfans|casino|betting|gambl)", re.I)
FREEZE_CSS = "*,*::before,*::after{animation-play-state:paused!important;transition:none!important;caret-color:transparent!important}"


def registrable(host):
    host = (host or "").lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    parts = host.split(".")
    if len(parts) >= 3 and parts[-2] in {"co", "com", "org", "net", "ac", "gov", "edu", "ne", "or"} and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def robots_allows(url):
    parts = urllib.parse.urlsplit(url)
    robots = f"{parts.scheme}://{parts.netloc}/robots.txt"
    parser = urllib.robotparser.RobotFileParser()
    try:
        request = urllib.request.Request(robots, headers={"User-Agent": BOT})
        with urllib.request.urlopen(request, timeout=10) as response:
            parser.parse(response.read(200_000).decode("utf-8", "replace").splitlines())
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            return False
        return True  # no robots.txt
    except Exception:
        return False  # unreachable host; do not crawl
    return parser.can_fetch(BOT, url) and parser.can_fetch("*", url)


def changed_fraction(a, b):
    x = np.asarray(Image.open(BytesIO(a)).convert("RGB"), dtype=np.int16)
    y = np.asarray(Image.open(BytesIO(b)).convert("RGB"), dtype=np.int16)
    if x.shape != y.shape:
        return 1.0
    return float((np.abs(x - y).max(axis=2) > 24).mean())


def ascii_ratio(text):
    letters = [c for c in text if c.isalpha()]
    return sum(c.isascii() for c in letters) / max(1, len(letters))


DISMISS_JS = r"""() => {
  const re = /^(accept( all)?( cookies)?|allow( all)?( cookies)?|agree|i agree|i accept|got it|ok|okay|reject( all)?|decline( all)?|deny|only (necessary|essential)( cookies)?|continue|close|dismiss|no,? thanks)[.! ]*$/i;
  const nodes = Array.from(document.querySelectorAll('button, a, [role=button], input[type=button], input[type=submit]'));
  const inBanner = (el) => { for (let e = el; e && e !== document.body; e = e.parentElement) {
      const s = ((e.id || '') + ' ' + (typeof e.className === 'string' ? e.className : '') + ' ' + (e.getAttribute('aria-label') || '')).toLowerCase();
      if (/cookie|consent|gdpr|privacy|cmp|onetrust|banner|modal|popup|dialog|newsletter|overlay/.test(s)) return true;
      const p = getComputedStyle(e).position; if (p === 'fixed' || p === 'sticky') return true; }
    return false; };
  let clicked = 0;
  for (const el of nodes) {
    const t = (el.innerText || el.value || el.getAttribute('aria-label') || '').trim();
    const r = el.getBoundingClientRect();
    if (!t || r.width === 0 || r.height === 0 || !re.test(t) || !inBanner(el)) continue;
    try { el.click(); clicked++; } catch (e) {}
    if (clicked >= 2) break;
  }
  return clicked;
}"""


def capture(page, url, excluded):
    response = page.goto(url, wait_until="load", timeout=25000)
    if response is None or response.status >= 400:
        return None, "http_status"
    final = page.url
    if registrable(urllib.parse.urlsplit(final).hostname) in excluded:
        return None, "redirect_excluded"
    try:
        page.wait_for_load_state("networkidle", timeout=6000)
    except Exception:
        pass
    page.evaluate("() => document.fonts ? document.fonts.ready.then(() => true) : true")
    page.wait_for_timeout(1000)
    try:
        if page.evaluate(DISMISS_JS):
            page.wait_for_timeout(1200)
            if page.url != final:  # a dismiss link navigated away
                return None, "navigated"
    except Exception:
        pass
    page.wait_for_timeout(500)
    page.add_style_tag(content=FREEZE_CSS)
    page.evaluate("() => window.scrollTo(0, 0)")
    page.wait_for_timeout(700)
    shot1 = page.screenshot(type="png")
    info = page.evaluate(JS)
    page.wait_for_timeout(1200)
    shot2 = page.screenshot(type="png")
    change = changed_fraction(shot1, shot2)
    if change > 0.003:
        return None, "unstable"
    info["change_fraction"] = change
    info["final_url"] = final
    info["status"] = response.status
    return (shot1, info), None


def page_ok(info):
    boxes = info["boxes"]
    text = " ".join(b["text"] for b in boxes)
    if BAD_TITLE.search(info.get("title") or ""):
        return "bad_title"
    if BAD_TEXT.search(text) or BAD_TEXT.search(info.get("title") or ""):
        return "bad_text"
    lang = (info.get("lang") or "").lower()
    if lang and not lang.startswith("en"):
        return "lang"
    if ascii_ratio(text) < 0.97:
        return "non_ascii"
    if not 8 <= len(boxes) <= 250:
        return "box_count"
    if sum(len(b["text"]) for b in boxes) < 80:
        return "too_little_text"
    return None


def internal_link(page, base):
    links = page.evaluate("""() => Array.from(document.querySelectorAll('a[href]')).filter(a => {
        const r = a.getBoundingClientRect(); return r.width > 0 && r.height > 0; }).map(a => a.href).slice(0, 300)""")
    host = registrable(urllib.parse.urlsplit(base).hostname)
    candidates = []
    for link in links:
        parts = urllib.parse.urlsplit(link)
        if parts.scheme not in ("http", "https") or registrable(parts.hostname) != host:
            continue
        if parts.path in ("", "/") or re.search(r"\.(pdf|jpg|png|zip|mp4|mp3)$|login|signin|signup|cart|account|privacy|terms|cookie", link, re.I):
            continue
        candidates.append(urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, "")))
    return sorted(set(candidates))


def worker(queue, results, out, excluded, seed):
    os.environ.setdefault("FONTCONFIG_FILE", "/scratch/frontier-ml-workspace/work/screenshot-generation/fonts/fonts.conf")
    from playwright.sync_api import sync_playwright
    rng = random.Random(seed)
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--disable-blink-features=AutomationControlled", "--hide-scrollbars", "--mute-audio"])
        while True:
            domain = queue.get()
            if domain is None:
                break
            urls = [f"https://{domain}/"]
            status = {}
            context = None
            try:
                context = browser.new_context(viewport={"width": 1280, "height": 720}, device_scale_factor=1, locale="en-US",
                                              timezone_id="UTC", color_scheme="light", reduced_motion="reduce", user_agent=UA,
                                              java_script_enabled=True, ignore_https_errors=False)
                context.set_default_timeout(25000)
                page = context.new_page()
                visited = 0
                while urls and visited < 2:
                    url = urls.pop(0)
                    visited += 1
                    if not robots_allows(url):
                        status[url] = "robots"
                        continue
                    try:
                        captured, reason = capture(page, url, excluded)
                    except Exception as error:
                        status[url] = "error:" + type(error).__name__
                        continue
                    if reason:
                        status[url] = reason
                        continue
                    shot, info = captured
                    reason = page_ok(info)
                    if visited == 1 and rng.random() < 0.35:
                        try:
                            links = internal_link(page, info["final_url"])
                            if links:
                                urls.append(rng.choice(links))
                        except Exception:
                            pass
                    if reason:
                        status[url] = reason
                        continue
                    key = hashlib.sha256(info["final_url"].encode()).hexdigest()[:20]
                    info.update(url=url, domain=domain, captured_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                viewport=[1280, 720], device_scale_factor=1, user_agent=UA)
                    (out / f"{key}.png").write_bytes(shot)
                    (out / f"{key}.json").write_text(json.dumps(info))
                    status[url] = "ok:" + key
            except Exception as error:
                status[domain] = "context_error:" + type(error).__name__
            finally:
                if context is not None:
                    try:
                        context.close()
                    except Exception:
                        pass
            results.put((domain, status))
        browser.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", required=True)
    parser.add_argument("--exclude", default="exclusion_domains.txt")
    parser.add_argument("--out", default="raw")
    parser.add_argument("--log", default="crawl_log.jsonl")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    excluded = set(Path(args.exclude).read_text().split()) if Path(args.exclude).exists() else set()
    done = set()
    if Path(args.log).exists():
        done = {json.loads(line)["domain"] for line in open(args.log) if line.strip()}
    domains = [d for d in Path(args.domains).read_text().split() if d not in done and d not in excluded][:args.limit]
    print(f"{len(domains)} domains to crawl ({len(done)} already done)", flush=True)
    queue, results = mp.Queue(), mp.Queue()
    for d in domains:
        queue.put(d)
    procs = [mp.Process(target=worker, args=(queue, results, out, excluded, i), daemon=True) for i in range(args.workers)]
    for _ in procs:
        queue.put(None)
    for p in procs:
        p.start()
    ok = 0
    with open(args.log, "a") as log:
        for i in range(len(domains)):
            try:
                domain, status = results.get(timeout=900)
            except Exception:
                print('result timeout; stopping', flush=True)
                break
            ok += sum(v.startswith("ok:") for v in status.values())
            log.write(json.dumps({"domain": domain, "status": status}) + "\n")
            log.flush()
            if i % 50 == 0:
                print(f"{i}/{len(domains)} domains, {ok} pages kept", flush=True)
    for p in procs:
        p.join(timeout=60)
    print("done", ok)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
