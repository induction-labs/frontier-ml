"""Render private ocr-latent-compression images with DOM ground truth (operator-only).

Every image is an HTML page rendered by headless Chromium (Playwright). The ground
truth is ``document.body.innerText`` of the rendered page, so it is produced by the
renderer, never by an annotator or an OCR model. Pages are generated from a private
seed and a fresh text corpus (``ocr_latent_sources``); half are letter-size document
pages (1275x1650, 150 dpi) and half are 1920x1080 application screenshots.

Every page passes mechanical checks before it is accepted:

* no scrolling or overflowing container; every text rectangle lies inside the viewport
  and inside every clipping ancestor;
* no text below the minimum font size, no hidden or transparent text;
* the normalized reference length falls inside the requested density bucket.

    python tasks/ocr-latent-compression/ocr_latent_render.py build --corpus corpus.jsonl \
        --fonts FONTS_DIR --out data/ocr-latent-compression --validation 100 --heldout 400

Requires ``playwright`` with Chromium; never imported by the verifier.
"""
from __future__ import annotations

import argparse
import hashlib
from html import escape
import io
import json
from pathlib import Path
import random
import re
import secrets
import shutil
import tempfile

from ocr_text import VERSION as NORMALIZATION, normalize

BUCKETS = (("sparse", 200, 900), ("medium", 900, 2000), ("dense", 2000, 3500), ("very_dense", 3500, 6500))
DOC_SIZE, SCREEN_SIZE = (1275, 1650), (1920, 1080)
GENERATOR_VERSION = "ocr-latent-render-v1"


def bucket_of(length):
    for name, low, high in BUCKETS:
        if low <= length < high:
            return name
    return None


# --------------------------------------------------------------------------- corpus

SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
ASCII_NAME = re.compile(r"^[A-Z][a-z]+(?:-[A-Z][a-z]+)?$")
PROVENANCE_WORDS = re.compile(r"arxiv|oai-?pmh|preprint server", re.IGNORECASE)


class Corpus:
    """Fresh text for one split. Papers are assigned to a split by a keyed hash (grouped by paper)."""

    def __init__(self, records, seed, pool):
        def assigned(record):
            if pool == "all":  # organizer training renders from a separate, disjoint corpus
                return True
            digest = hashlib.sha256(f"{seed}:pool:{record['arxiv_id']}".encode()).digest()
            return ("validation" if digest[0] < 52 else "heldout") == pool  # about 20% validation
        records = [r for r in records if assigned(r)]

        def neutral(text):  # page text must not name where it came from
            return not PROVENANCE_WORDS.search(text)
        self.papers = [{**r, "full_text": [b for b in r["full_text"] if neutral(b["text"])]} for r in records
                       if r["title_plain"] and r["abstract_plain"] and len(r["abstract"]) >= 300
                       and neutral(r["title"]) and neutral(r["abstract"])]
        self.full = [r for r in self.papers if sum(b["kind"] == "paragraph" for b in r["full_text"]) >= 4]
        self.sentences = [s for r in self.papers for s in SENTENCE.split(r["abstract"]) if 40 <= len(s) <= 260]
        self.titles = [r["title"] for r in self.papers if len(r["title"]) <= 140]
        first, last = set(), set()
        for record in records:
            for author in record["authors"]:
                parts = author.split()
                if len(parts) >= 2 and ASCII_NAME.match(parts[0]) and ASCII_NAME.match(parts[-1]):
                    first.add(parts[0])
                    last.add(parts[-1])
        self.first, self.last = sorted(first), sorted(last)
        words = set()
        for title in self.titles:
            words.update(w for w in re.findall(r"\b[A-Za-z][a-z]{3,13}\b", title))
        self.words = sorted(words)
        if not (self.papers and self.full and len(self.first) > 50 and len(self.last) > 50):
            raise ValueError(f"Corpus pool {pool} is too small")


class Text:
    """Random text helpers bound to one item's RNG."""

    def __init__(self, rng, corpus):
        self.rng, self.c = rng, corpus

    def pick(self, items):
        return self.rng.choice(items)

    def name(self):
        return f"{self.pick(self.c.first)} {self.pick(self.c.last)}"

    def names(self, low, high):
        return ", ".join(self.name() for _ in range(self.rng.randint(low, high)))

    def initials(self, name):
        return "".join(part[0] for part in name.split()[:2])

    def sentence(self):
        return self.pick(self.c.sentences)

    def sentences(self, low, high):
        return " ".join(self.sentence() for _ in range(self.rng.randint(low, high)))

    def title(self, max_len=140):
        for _ in range(50):
            title = self.pick(self.c.titles)
            if len(title) <= max_len:
                return title
        return self.pick(self.c.titles)[:max_len].rsplit(" ", 1)[0]

    def phrase(self, low=1, high=3):
        return " ".join(self.pick(self.c.words).lower() for _ in range(self.rng.randint(low, high))).capitalize()

    def word(self):
        return self.pick(self.c.words)

    def date(self):
        month = self.rng.randint(1, 12)
        day = self.rng.randint(1, 28)
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        style = self.rng.randrange(3)
        if style == 0:
            return f"{months[month - 1]} {day}, 2026"
        if style == 1:
            return f"2026-{month:02d}-{day:02d}"
        return f"{day} {months[month - 1]} 2026"

    def time(self):
        if self.rng.random() < 0.5:
            return f"{self.rng.randint(0, 23):02d}:{self.rng.randint(0, 59):02d}"
        return f"{self.rng.randint(1, 12)}:{self.rng.randint(0, 59):02d} {self.pick(['AM', 'PM'])}"

    def money(self):
        value = self.rng.choice([self.rng.uniform(1, 99), self.rng.uniform(100, 9999), self.rng.uniform(10000, 250000)])
        return self.pick(["$", "€", "£"]) + f"{value:,.2f}"

    def count(self):
        return f"{self.rng.choice([self.rng.randint(2, 99), self.rng.randint(100, 9999), self.rng.randint(10000, 999999)]):,}"

    def percent(self, signed=False):
        value = self.rng.uniform(0.1, 48.0)
        sign = self.pick(["+", "-"]) if signed else ""
        return f"{sign}{value:.1f}%"

    def code(self):
        letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
        return "".join(self.rng.choice(letters) for _ in range(self.rng.randint(2, 4))) + "-" + str(self.rng.randint(100, 99999))

    def email(self, name=None):
        name = name or self.name()
        first, last = name.lower().split()[0], name.lower().split()[-1]
        domain = self.word().lower() + self.pick([".io", ".com", ".org", ".net", ".dev", ".co"])
        return self.pick([f"{first}.{last}", f"{first[0]}{last}", f"{first}"]) + "@" + domain

    def domain(self):
        return self.word().lower() + self.pick([".io", ".com", ".org", ".net", ".dev"])

    def filename(self):
        return "-".join(self.word().lower() for _ in range(self.rng.randint(1, 3))) + self.pick(
            [".pdf", ".docx", ".xlsx", ".png", ".csv", ".zip", ".pptx", ".md"])


# --------------------------------------------------------------------------- HTML helpers


def e(tag, text="", cls="", attrs="", raw=False):
    body = text if raw else escape(text)
    cls_attr = f' class="{cls}"' if cls else ""
    return f"<{tag}{cls_attr}{(' ' + attrs) if attrs else ''}>{body}</{tag}>"


def item(tag, text, cls="", raw=False):
    return e(tag, text, cls, "data-item", raw)


def page(fonts_css, css, body):
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{fonts_css}
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{width:100%;height:100%;overflow:hidden}}
body{{-webkit-font-smoothing:antialiased;overflow-wrap:break-word}}
ul{{list-style:none}}
{css}</style></head><body>{body}</body></html>"""


# --------------------------------------------------------------------------- document pages

DOC_PARAMS = {  # body px range, columns choices, max chars
    "sparse": ((24, 34), (1,)),
    "medium": ((19, 25), (1,)),
    "dense": ((16, 20), (1, 2)),
    "very_dense": ((14, 17), (2,)),
}


def document(rng, corpus, fonts, bucket):
    t = Text(rng, corpus)
    size = rng.randint(*DOC_PARAMS[bucket][0])
    columns = rng.choice(DOC_PARAMS[bucket][1])
    body_font = fonts.pick(rng, "serif" if rng.random() < 0.7 else "sans")
    head_font = rng.choice([body_font, fonts.pick(rng, "sans")])
    margin_x = rng.randint(90, 150) if bucket != "very_dense" else rng.randint(70, 100)
    margin_y = rng.randint(80, 130)
    align = rng.choice(["justify", "left"])
    layout = rng.choice({"sparse": ["title_page", "title_page", "memo"],
                         "medium": ["article_first", "article_body", "memo", "digest"],
                         "dense": ["article_first", "article_body", "digest"],
                         "very_dense": ["article_first", "article_body", "digest"]}[bucket])
    paper = t.pick(corpus.full)
    css = f"""
body{{background:#fff;color:#111;font-family:{body_font};font-size:{size}px;line-height:{rng.uniform(1.3, 1.55):.2f}}}
.page{{position:absolute;inset:{margin_y}px {margin_x}px;display:flex;flex-direction:column}}
.run{{font-size:{max(13, int(size * 0.8))}px;color:#444;display:flex;justify-content:space-between;border-bottom:1px solid #999;padding-bottom:6px;margin-bottom:{int(size * 1.2)}px}}
.foot{{font-size:{max(13, int(size * 0.8))}px;color:#444;text-align:center;margin-top:{int(size * 0.8)}px}}
h1{{font-family:{head_font};font-size:{int(size * rng.uniform(1.5, 2.0))}px;line-height:1.25;margin-bottom:{int(size * 0.6)}px;text-align:{rng.choice(['left', 'center'])}}}
h2{{font-family:{head_font};font-size:{int(size * 1.15)}px;margin:{int(size * 0.7)}px 0 {int(size * 0.35)}px}}
h3{{font-family:{head_font};font-size:{int(size * 1.0)}px;font-weight:700;margin:{int(size * 0.5)}px 0 {int(size * 0.25)}px}}
.authors{{font-size:{int(size * 1.0)}px;margin-bottom:{int(size * 0.3)}px;text-align:inherit}}
.meta{{color:#333;font-size:{max(13, int(size * 0.85))}px;margin-bottom:{int(size * 0.9)}px}}
.abstract{{margin:0 {rng.randint(0, 60)}px {int(size * 0.8)}px;font-size:{max(13, int(size * 0.92))}px}}
.abstract b{{font-family:{head_font}}}
p{{text-align:{align};margin-bottom:{int(size * rng.uniform(0.3, 0.7))}px;{('text-indent:' + str(int(size * 1.5)) + 'px') if rng.random() < 0.5 else ''}}}
.flow{{flex:1;min-height:0;overflow:hidden;column-count:{columns};column-gap:{rng.randint(28, 48)}px;column-fill:auto}}
.entry{{margin-bottom:{int(size * 0.9)}px;break-inside:avoid}}
.entry .t{{font-weight:700;font-family:{head_font}}}
.entry .a{{font-style:italic;color:#333}}
ul.bul li{{margin:0 0 {int(size * 0.3)}px {int(size * 1.2)}px;list-style:disc outside}}
.center{{text-align:center}}
"""
    head_left = rng.choice(["Preprint", "Working paper", "Technical report", "Draft manuscript", "Under review"])
    head_right = rng.choice([t.date(), f"Page {rng.randint(1, 30)}", f"Vol. {rng.randint(1, 40)}, No. {rng.randint(1, 12)}"])
    running = e("div", e("span", head_left) + e("span", head_right), "run", raw=True) if rng.random() < 0.8 or layout == "article_body" else ""
    footer = e("div", str(rng.randint(1, 40)), "foot") if rng.random() < 0.6 else ""
    flow = []
    top = ""
    if layout == "title_page":
        top = (e("h1", paper["title"]) + e("div", t.names(1, 6), "authors center")
               + e("div", t.date(), "meta center"))
        sentences = SENTENCE.split(paper["abstract"])
        flow.append(e("h2", "Abstract", "center"))
        flow.extend(item("p", s) for s in sentences)
        running = footer = "" if rng.random() < 0.5 else running
    elif layout == "memo":
        sender, receiver = t.name(), t.name()
        top = (e("h1", rng.choice(["Memorandum", "Research note", "Project update", "Briefing note"]) )
               + e("div", f"To: {receiver}", "meta") + e("div", f"From: {sender}", "meta") + e("div", f"Date: {t.date()}", "meta")
               + e("div", f"Subject: {paper['title']}", "meta"))
        blocks = [b for b in paper["full_text"] if b["kind"] == "paragraph"]
        flow.extend(item("p", b["text"]) for b in blocks)
    elif layout == "article_first":
        top = e("h1", paper["title"]) + e("div", t.names(1, 8), "authors")
        top += e("div", e("b", "Abstract. ") + escape(paper["abstract"]), "abstract", raw=True)
        for block in paper["full_text"]:
            flow.append(item("h2" if block["kind"] == "heading" else "p", block["text"]))
    elif layout == "article_body":
        start = rng.randint(0, max(0, len(paper["full_text"]) - 6))
        for block in paper["full_text"][start:]:
            flow.append(item("h2" if block["kind"] == "heading" else "p", block["text"]))
        if rng.random() < 0.3:
            flow.insert(rng.randint(1, max(1, len(flow) - 1)),
                        e("ul", "".join(e("li", t.sentence()) for _ in range(rng.randint(2, 4))), "bul", "data-item", raw=True))
    else:  # digest
        top = e("h1", rng.choice(["Abstracts", "Session abstracts", "New submissions", "Selected abstracts"]) + f" {rng.randint(1, 12)}")
        others = rng.sample(corpus.papers, min(12, len(corpus.papers)))
        for other in others:
            flow.append(e("div", e("div", other["title"], "t") + e("div", t.names(1, 4), "a")
                          + e("p", other["abstract"]), "entry", "data-item", raw=True))
    body = e("div", running + top + e("div", "".join(flow), "flow", "data-box", raw=True) + footer, "page", raw=True)
    return layout, page(fonts.css, css, body), DOC_SIZE, 13


# --------------------------------------------------------------------------- screenshots

THEMES = {
    "light": {"bg": "#ffffff", "panel": "#f6f8fa", "fg": "#1f2328", "muted": "#59636e", "line": "#d1d9e0", "card": "#ffffff"},
    "light2": {"bg": "#f4f5f7", "panel": "#ffffff", "fg": "#172b4d", "muted": "#505f79", "line": "#dfe1e6", "card": "#ffffff"},
    "dark": {"bg": "#0d1117", "panel": "#161b22", "fg": "#e6edf3", "muted": "#9aa5b1", "line": "#30363d", "card": "#161b22"},
    "dark2": {"bg": "#1e1f22", "panel": "#2b2d31", "fg": "#f2f3f5", "muted": "#b5bac1", "line": "#3f4147", "card": "#313338"},
}
ACCENTS = ["#0969da", "#1a7f37", "#8250df", "#bf3989", "#d1242f", "#bc4c00", "#0e7490", "#4f46e5"]
SCREEN_FONT = {"sparse": (20, 26), "medium": (15, 19), "dense": (13, 16), "very_dense": (12, 14)}


def screen_css(rng, fonts, bucket):
    theme = THEMES[rng.choice(list(THEMES))]
    accent = rng.choice(ACCENTS)
    size = rng.randint(*SCREEN_FONT[bucket])
    font = fonts.pick(rng, "sans")
    gap = {"sparse": 22, "medium": 14, "dense": 10, "very_dense": 7}[bucket]
    return size, gap, f"""
body{{background:{theme['bg']};color:{theme['fg']};font-family:{font};font-size:{size}px;line-height:1.4}}
.app{{position:absolute;inset:0;display:flex;flex-direction:column}}
.top{{display:flex;align-items:center;gap:{gap * 2}px;padding:{gap}px {gap * 2}px;border-bottom:1px solid {theme['line']};background:{theme['panel']}}}
.brand{{font-weight:700;font-size:{int(size * 1.25)}px;color:{accent}}}
.search{{flex:0 1 480px;border:1px solid {theme['line']};border-radius:6px;padding:{gap // 2 + 2}px {gap}px;color:{theme['muted']};background:{theme['bg']}}}
.main{{flex:1;display:flex;min-height:0}}
.side{{width:{rng.randint(220, 300)}px;flex:none;padding:{gap}px;border-right:1px solid {theme['line']};background:{theme['panel']};overflow:hidden}}
.side li{{padding:{gap // 3 + 2}px {gap // 2}px;border-radius:5px;display:flex;justify-content:space-between;gap:8px}}
.side li.on{{background:{accent};color:#fff}}
.side h4{{color:{theme['muted']};font-size:{int(size * 0.85)}px;margin:{gap}px 0 {gap // 2}px;text-transform:uppercase;letter-spacing:.04em}}
.content{{flex:1;min-width:0;padding:{gap}px {gap * 2}px;overflow:hidden;display:flex;flex-direction:column;gap:{gap}px}}
.muted{{color:{theme['muted']}}}
.box{{flex:1;min-height:0;overflow:hidden}}
.card{{background:{theme['card']};border:1px solid {theme['line']};border-radius:8px;padding:{gap}px}}
.row{{display:flex;gap:{gap}px;align-items:baseline}}
.grow{{flex:1;min-width:0}}
.btn{{background:{accent};color:#fff;border-radius:6px;padding:{gap // 2 + 2}px {gap + 4}px;font-weight:600;display:inline-block}}
.btn2{{border:1px solid {theme['line']};border-radius:6px;padding:{gap // 2 + 2}px {gap + 4}px;display:inline-block}}
.pill{{border:1px solid {theme['line']};border-radius:999px;padding:1px {gap // 2 + 4}px;font-size:{int(size * 0.85)}px;color:{theme['muted']}}}
.avatar{{width:{int(size * 2.2)}px;height:{int(size * 2.2)}px;border-radius:50%;background:{accent};color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:{int(size * 0.8)}px;flex:none}}
table{{border-collapse:collapse;width:100%}}
th,td{{text-align:left;padding:{gap // 2 + 1}px {gap}px;border-bottom:1px solid {theme['line']};vertical-align:top}}
th{{color:{theme['muted']};font-weight:600}}
h1{{font-size:{int(size * 1.9)}px;line-height:1.2}}
h2{{font-size:{int(size * 1.35)}px;line-height:1.25}}
h3{{font-size:{int(size * 1.1)}px}}
.accent{{color:{accent}}}
.bar{{height:{int(size * 0.6)}px;background:{accent};border-radius:3px;opacity:.8}}
""", theme, accent


def s_email(rng, t, bucket, size, gap, theme, accent):
    folders = [("Inbox", t.count()), ("Starred", ""), ("Snoozed", ""), ("Sent", ""), ("Drafts", str(rng.randint(1, 9))), ("Archive", ""), ("Spam", "")]
    side = "".join(e("li", e("span", f) + e("span", n, "muted"), "on" if i == 0 else "", raw=True) for i, (f, n) in enumerate(folders))
    side += e("h4", "Labels") + "".join(item("li", t.phrase(1, 2)) for _ in range(8))
    rows = []
    for _ in range(40):
        sender = t.name()
        rows.append(e("div", e("div", e("b", sender, "grow") + e("span", t.time(), "muted"), "row", raw=True)
                      + e("div", t.title(90)) + e("div", t.sentence(), "muted"),
                      "card", f'data-item style="margin-bottom:{gap}px"', raw=True))
    sender = t.name()
    reading = (e("h2", t.title(110)) + e("div", f"From: {sender} · {t.email(sender)}", "muted")
               + e("div", f"To: {t.name()}, {t.name()} · {t.date()} {t.time()}", "muted")
               + e("div", "".join(item("p", t.sentences(1, 4), raw=False) for _ in range(12)), "box", 'data-box style="margin-top:8px"', raw=True))
    body = (e("div", e("span", rng.choice(["Mail", "Inbox", "Postbox", "Relay"]), "brand") + e("div", "Search mail", "search") + e("span", t.name(), "muted"), "top", raw=True)
            + e("div", e("ul", side, "side", "data-box", raw=True)
                + e("div", "".join(rows), "content", f'data-box style="flex:0 0 {rng.randint(520, 640)}px;display:block"', raw=True)
                + e("div", reading, "content", raw=True), "main", raw=True))
    return body


def s_chat(rng, t, bucket, size, gap, theme, accent):
    channels = [t.word().lower() + rng.choice(["", "-team", "-review", "-ops", "-notes"]) for _ in range(14)]
    side = e("h4", "Channels") + "".join(item("li", c, "on" if i == 2 else "") for i, c in enumerate(channels))
    side += e("h4", "Direct messages") + "".join(item("li", t.name()) for _ in range(8))
    messages = []
    for _ in range(40):
        who = t.name()
        messages.append(e("div", e("div", t.initials(who), "avatar")
                          + e("div", e("div", e("b", who) + " " + e("span", t.time(), "muted"), raw=True) + e("div", t.sentences(1, 2)), "grow", raw=True),
                          "row", f'data-item style="margin-bottom:{gap}px;align-items:flex-start"', raw=True))
    content = (e("div", e("h2", channels[2]) + e("span", f"{rng.randint(3, 400)} members", "pill"), "row", raw=True)
               + e("div", "".join(messages), "box", "data-box", raw=True)
               + e("div", f"Message {channels[2]}", "search", 'style="flex:none;max-width:none"'))
    return (e("div", e("span", rng.choice(["Huddle", "Campfire", "Threadline", "Chatter"]), "brand") + e("div", "Search workspace", "search"), "top", raw=True)
            + e("div", e("ul", side, "side", "data-box", raw=True) + e("div", content, "content", raw=True), "main", raw=True))


def s_dashboard(rng, t, bucket, size, gap, theme, accent):
    cards = "".join(e("div", e("div", t.phrase(1, 2), "muted") + e("h2", rng.choice([t.count(), t.money(), t.percent()]))
                          + e("div", t.percent(True) + " vs last period", "muted"), "card grow", "data-item", raw=True) for _ in range(rng.randint(3, 5)))
    bars = "".join(e("div", "", "bar", f'style="width:{rng.randint(15, 100)}%;margin:6px 0"', raw=True) for _ in range(rng.randint(4, 8)))
    headers = ["Name", "Owner", rng.choice(["Region", "Team", "Category"]), "Updated", rng.choice(["Revenue", "Cost", "Value"]), "Status"]
    rows = "".join(e("tr", "".join(e("td", v) for v in [t.phrase(1, 3), t.name(), t.word(), t.date(), t.money(),
                                                          rng.choice(["Active", "Paused", "At risk", "On track", "Closed"])]), attrs="data-item", raw=True) for _ in range(40))
    nav = e("ul", "".join(item("li", w, "on" if i == 0 else "") for i, w in enumerate(["Overview", "Reports", "Customers", "Pipeline", "Billing", "Integrations", "Settings"])), "side", raw=True)
    content = (e("div", e("h1", rng.choice(["Overview", "Weekly report", "Performance", "Revenue"]), "grow") + e("span", f"{t.date()} to {t.date()}", "pill") + e("span", "Export", "btn"), "row", raw=True)
               + e("div", cards, "row", raw=True)
               + e("div", e("div", e("h3", t.title(60)) + bars, "card", 'style="flex:0 0 38%"', raw=True)
                   + e("div", e("table", e("tr", "".join(e("th", h) for h in headers), raw=True) + rows, raw=True), "card grow box", "data-box", raw=True),
                   "row box", 'style="align-items:stretch"', raw=True))
    return (e("div", e("span", rng.choice(["Metricly", "Northstar", "Ledger", "Pulse"]), "brand") + e("div", "Search reports", "search"), "top", raw=True)
            + e("div", nav + e("div", content, "content", raw=True), "main", raw=True))


def s_article(rng, t, bucket, size, gap, theme, accent):
    nav = "".join(e("span", w) for w in rng.sample(["World", "Science", "Technology", "Business", "Health", "Climate", "Culture", "Opinion", "Sports"], 6))
    paragraphs = "".join(item("p", t.sentences(2, 4)) for _ in range(14))
    related = "".join(item("li", t.title(90)) for _ in range(12))
    width = rng.randint(760, 980)
    return (e("div", e("span", rng.choice(["The Ledger", "Daily Signal", "Meridian", "The Courier"]), "brand") + e("div", nav, "row", raw=True), "top", raw=True)
            + e("div", e("div", e("div", rng.choice(["Science", "Technology", "Research", "Analysis"]), "accent")
                          + e("h1", t.title(130)) + e("h3", t.sentence(), "muted")
                          + e("div", f"By {t.name()} · {t.date()} · {rng.randint(3, 15)} min read", "muted")
                          + e("div", paragraphs, "box", f'data-box style="column-count:1"', raw=True),
                          "content", f'style="flex:0 0 {width}px;margin-left:{rng.randint(60, 200)}px"', raw=True)
                + e("div", e("h3", rng.choice(["Most read", "Related", "Trending"])) + e("ul", related, "box", 'data-box style="display:flex;flex-direction:column;gap:10px"', raw=True),
                    "content", raw=True), "main", raw=True))


def s_docs(rng, t, bucket, size, gap, theme, accent):
    tree = "".join(e("h4", t.phrase(1, 2)) + "".join(item("li", t.phrase(1, 3)) for _ in range(5)) for _ in range(4))
    headings = [t.phrase(2, 4) for _ in range(5)]
    main = e("div", f"Docs / {t.phrase(1, 1)} / {t.phrase(1, 2)}", "muted") + e("h1", t.title(90)) + e("p", t.sentences(1, 2), "muted")
    parts = []
    for heading in headings:
        parts.append(item("h2", heading))
        parts.extend(item("p", t.sentences(2, 3)) for _ in range(2))
        if rng.random() < 0.3:
            parts.append(e("div", e("b", "Note: ") + escape(t.sentence()), "card", "data-item", raw=True))
    main += e("div", "".join(parts), "box", 'data-box style="display:flex;flex-direction:column;gap:10px"', raw=True)
    toc = e("h4", "On this page") + "".join(e("li", h) for h in headings[:4])
    return (e("div", e("span", rng.choice(["DevHub", "Handbook", "Knowledge base", "Guides"]), "brand") + e("div", "Search documentation", "search"), "top", raw=True)
            + e("div", e("ul", tree, "side", "data-box", raw=True) + e("div", main, "content", raw=True)
                + e("ul", toc, "side", 'style="border-right:none;border-left:1px solid;width:240px"', raw=True), "main", raw=True))


def s_sheet(rng, t, bucket, size, gap, theme, accent):
    columns = rng.randint(5, 9)
    kinds = [rng.choice(["name", "word", "date", "money", "count", "percent", "code"]) for _ in range(columns)]
    kinds[0] = rng.choice(["name", "code", "word"])
    labels = {"name": "Contact", "word": "Category", "date": "Date", "money": "Amount", "count": "Units", "percent": "Change", "code": "Reference"}
    make = {"name": t.name, "word": t.word, "date": t.date, "money": t.money, "count": t.count, "percent": lambda: t.percent(True), "code": t.code}
    letters = "ABCDEFGHIJ"
    header = e("tr", e("th", "") + "".join(e("th", letters[i]) for i in range(columns)), raw=True)
    first = e("tr", e("th", "1") + "".join(e("td", e("b", labels[k])) for k in kinds), raw=True)
    rows = "".join(e("tr", e("th", str(r + 2)) + "".join(e("td", make[k]()) for k in kinds), attrs="data-item", raw=True) for r in range(60))
    toolbar = e("div", "".join(e("span", w) for w in ["File", "Edit", "View", "Insert", "Format", "Data", "Tools", "Help"]), "row", raw=True)
    return (e("div", e("span", t.phrase(1, 3) + rng.choice([" budget", " tracker", " export", " 2026"]), "brand") + toolbar, "top", raw=True)
            + e("div", e("div", e("div", e("b", "fx") + e("span", make[kinds[1]]()), "row card", raw=True)
                         + e("div", e("table", header + first + rows, raw=True), "box", "data-box", raw=True), "content", raw=True), "main", raw=True))


def s_settings(rng, t, bucket, size, gap, theme, accent):
    sections = ["Profile", "Account", "Notifications", "Security", "Billing", "Integrations", "Appearance", "Advanced"]
    side = "".join(e("li", s, "on" if i == 1 else "") for i, s in enumerate(sections))
    fields = []
    for _ in range(16):
        label = t.phrase(1, 3)
        value = rng.choice([t.name(), t.email(), t.domain(), t.date(), t.count(), t.phrase(1, 2)])
        fields.append(e("div", e("b", label) + e("div", value, "search", 'style="max-width:560px;margin:4px 0"') + e("div", t.sentence(), "muted"),
                        "", f'data-item style="margin-bottom:{gap}px"', raw=True))
    content = (e("h1", "Account settings") + e("p", t.sentence(), "muted")
               + e("div", "".join(fields), "box", 'data-box style="column-count:2;column-gap:40px"', raw=True)
               + e("div", e("span", "Save changes", "btn") + e("span", "Cancel", "btn2"), "row", raw=True))
    return (e("div", e("span", rng.choice(["Console", "Workspace", "Admin"]), "brand") + e("div", "Search settings", "search"), "top", raw=True)
            + e("div", e("ul", side, "side", raw=True) + e("div", content, "content", raw=True), "main", raw=True))


def s_search(rng, t, bucket, size, gap, theme, accent):
    query = " ".join(t.title(60).split()[:rng.randint(2, 5)])
    results = []
    for _ in range(14):
        domain = t.domain()
        results.append(e("div", e("div", f"{domain} › {t.word().lower()} › {rng.randint(2019, 2026)}", "muted")
                         + e("h3", t.title(100), "accent") + e("div", f"{t.date()} — {t.sentences(1, 2)}"), "", f'data-item style="margin-bottom:{gap + 6}px;max-width:900px"', raw=True))
    content = (e("div", f"About {t.count()} results ({rng.uniform(0.1, 0.9):.2f} seconds)", "muted")
               + e("div", "".join(results), "box", "data-box", raw=True))
    tabs = "".join(e("span", w) for w in ["All", "News", "Images", "Videos", "Maps", "More"])
    return (e("div", e("span", rng.choice(["Seek", "Findr", "Lookup", "Scout"]), "brand") + e("div", query, "search", 'style="flex:0 1 720px;color:inherit"') + e("div", tabs, "row", raw=True), "top", raw=True)
            + e("div", e("div", content, "content", f'style="padding-left:{rng.randint(120, 220)}px"', raw=True), "main", raw=True))


def s_shop(rng, t, bucket, size, gap, theme, accent):
    filters = e("h4", "Category") + "".join(item("li", t.phrase(1, 2)) for _ in range(6)) + e("h4", "Brand") + "".join(item("li", t.word()) for _ in range(6))
    cards = []
    for _ in range(24):
        name = f"{t.word()} {t.word().lower()} {rng.choice(['Pro', 'Lite', 'Max', 'Mini', 'Plus', 'S', 'X'])} {rng.randint(2, 900)}"
        cards.append(e("div", e("div", "", "", f'style="height:{int(size * 5)}px;background:{accent};opacity:.15;border-radius:6px;margin-bottom:8px"', raw=True)
                       + e("b", name) + e("div", t.money(), "accent") + e("div", f"{rng.uniform(3.0, 5.0):.1f} stars ({t.count()})", "muted")
                       + e("div", rng.choice(["Free delivery", "Ships in 2 days", "In stock", "Only 3 left"]), "muted"),
                       "card", "data-item", raw=True))
    grid = e("div", "".join(cards), "box", f'data-box style="display:grid;grid-template-columns:repeat({rng.randint(3, 5)},1fr);gap:{gap}px;align-content:start"', raw=True)
    content = e("div", e("h2", f"{t.phrase(1, 2)}", "grow") + e("span", f"{t.count()} results", "muted") + e("span", "Sort: Featured", "btn2"), "row", raw=True) + grid
    return (e("div", e("span", rng.choice(["Cartwheel", "Shelf", "Marketly", "Bazaar"]), "brand") + e("div", "Search products", "search") + e("span", "Cart (2)"), "top", raw=True)
            + e("div", e("ul", filters, "side", "data-box", raw=True) + e("div", content, "content", raw=True), "main", raw=True))


def s_board(rng, t, bucket, size, gap, theme, accent):
    columns = rng.sample(["Backlog", "To do", "In progress", "Review", "Blocked", "Done"], rng.randint(3, 5))
    cols = []
    for column in columns:
        cards = "".join(e("div", e("div", t.title(70)) + e("div", e("span", t.word(), "pill") + e("span", t.date(), "muted grow") + e("span", t.initials(t.name()), "avatar"), "row", 'style="margin-top:6px;align-items:center"', raw=True),
                          "card", f'data-item style="margin-bottom:{gap}px"', raw=True) for _ in range(12))
        cols.append(e("div", e("h3", f"{column} {rng.randint(1, 30)}") + e("div", cards, "box", "data-box", raw=True), "grow", 'style="display:flex;flex-direction:column;min-height:0;gap:8px"', raw=True))
    content = e("div", e("h1", t.phrase(1, 3) + " sprint " + str(rng.randint(1, 40)), "grow") + e("span", "New issue", "btn"), "row", raw=True) + e("div", "".join(cols), "row box", 'style="align-items:stretch"', raw=True)
    return (e("div", e("span", rng.choice(["Trackline", "Kanbo", "Sprintly"]), "brand") + e("div", "Search issues", "search"), "top", raw=True)
            + e("div", e("div", content, "content", raw=True), "main", raw=True))


def s_files(rng, t, bucket, size, gap, theme, accent):
    side = "".join(e("li", s, "on" if i == 0 else "") for i, s in enumerate(["My files", "Shared with me", "Recent", "Starred", "Trash"]))
    side += e("h4", "Storage") + e("div", f"{rng.uniform(1, 99):.1f} GB of 100 GB used", "muted")
    rows = "".join(e("tr", "".join(e("td", v) for v in [t.filename(), t.name(), f"{t.date()} {t.time()}", f"{rng.uniform(0.1, 900):.1f} {rng.choice(['KB', 'MB'])}"]), attrs="data-item", raw=True) for _ in range(50))
    content = (e("div", e("h2", f"My files / {t.phrase(1, 2)}", "grow") + e("span", "Upload", "btn"), "row", raw=True)
               + e("div", e("table", e("tr", "".join(e("th", h) for h in ["Name", "Owner", "Modified", "Size"]), raw=True) + rows, raw=True), "box", "data-box", raw=True))
    return (e("div", e("span", rng.choice(["Drive", "Vault", "Cabinet"]), "brand") + e("div", "Search files", "search"), "top", raw=True)
            + e("div", e("ul", side, "side", raw=True) + e("div", content, "content", raw=True), "main", raw=True))


def s_landing(rng, t, bucket, size, gap, theme, accent):
    nav = "".join(e("span", w) for w in rng.sample(["Product", "Pricing", "Customers", "Docs", "Blog", "Company"], 4))
    features = "".join(e("div", e("h3", t.phrase(1, 3)) + e("p", t.sentence(), "muted"), "card grow", "data-item", raw=True) for _ in range(3))
    hero = (e("h1", t.title(80), "", f'style="font-size:{int(size * 2.4)}px;max-width:1300px"') + e("p", t.sentence(), "muted", f'style="font-size:{int(size * 1.2)}px;max-width:1000px"')
            + e("div", e("span", rng.choice(["Get started", "Start free trial", "Request a demo"]), "btn") + e("span", rng.choice(["Learn more", "Watch video", "Contact sales"]), "btn2"), "row", raw=True))
    return (e("div", e("span", t.word(), "brand") + e("div", nav, "row grow", raw=True) + e("span", "Sign in", "btn2"), "top", raw=True)
            + e("div", e("div", hero + e("div", features, "row box", 'data-box style="align-items:stretch;margin-top:40px"', raw=True),
                         "content", 'style="padding:80px 160px;gap:28px"', raw=True), "main", raw=True))


def s_slide(rng, t, bucket, size, gap, theme, accent):
    bullets = "".join(item("li", t.sentence(), "", raw=False) for _ in range(6))
    return e("div", e("div", e("h1", t.title(90), "", f'style="font-size:{int(size * 2.2)}px"')
                      + e("ul", bullets, "box", f'data-box style="list-style:disc;padding-left:40px;display:flex;flex-direction:column;gap:{gap}px;font-size:{int(size * 1.25)}px"', raw=True)
                      + e("div", f"{t.name()} · {t.date()}", "muted"), "content", 'style="padding:90px 140px;gap:40px"', raw=True), "app", raw=True)


SCREENS = {"email": s_email, "chat": s_chat, "dashboard": s_dashboard, "article": s_article, "docs": s_docs,
           "spreadsheet": s_sheet, "settings": s_settings, "search": s_search, "shop": s_shop, "board": s_board,
           "files": s_files, "landing": s_landing, "slide": s_slide}
SCREEN_CHOICES = {
    "sparse": ["landing", "slide", "search", "chat", "settings", "article", "board", "email", "shop"],
    "medium": ["email", "chat", "dashboard", "article", "docs", "settings", "search", "shop", "board", "files", "landing", "slide"],
    "dense": ["email", "chat", "dashboard", "article", "docs", "spreadsheet", "settings", "search", "shop", "board", "files"],
    "very_dense": ["email", "dashboard", "docs", "spreadsheet", "settings", "search", "files", "chat", "article", "board"],
}


def screenshot(rng, corpus, fonts, bucket):
    t = Text(rng, corpus)
    layout = rng.choice(SCREEN_CHOICES[bucket])
    size, gap, css, theme, accent = screen_css(rng, fonts, bucket)
    body = SCREENS[layout](rng, t, bucket, size, gap, theme, accent)
    if not body.startswith('<div class="app"'):
        body = e("div", body, "app", raw=True)
    return layout, page(fonts.css, css, body), SCREEN_SIZE, 11


# --------------------------------------------------------------------------- browser checks

FIT_JS = r"""
(maxChars) => {
  const vw = innerWidth, vh = innerHeight;
  const len = () => document.body.innerText.replace(/\s+/g, ' ').trim().length;
  const overflowing = () => {
    const out = [];
    for (const box of document.querySelectorAll('[data-box]')) {
      if (box.scrollHeight > box.clientHeight + 1 || box.scrollWidth > box.clientWidth + 1) out.push(box);
    }
    return out;
  };
  const items = (box) => [...box.querySelectorAll('[data-item]')].filter(n => n.closest('[data-box]') === box);
  let guard = 0;
  while (guard++ < 4000) {
    const over = overflowing();
    if (over.length) {
      let removed = false;
      for (const box of over) { const list = items(box); if (list.length) { list[list.length - 1].remove(); removed = true; } }
      if (!removed) return {ok: false, reason: 'unfixable overflow'};
      continue;
    }
    if (len() <= maxChars) break;
    const boxes = [...document.querySelectorAll('[data-box]')].map(b => [b, items(b)]).filter(x => x[1].length);
    if (!boxes.length) return {ok: false, reason: 'cannot reach density'};
    boxes.sort((a, b) => b[1].length - a[1].length);
    boxes[0][1][boxes[0][1].length - 1].remove();
  }
  if (document.documentElement.scrollHeight > vh + 1 || document.documentElement.scrollWidth > vw + 1)
    return {ok: false, reason: 'page scrolls'};
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let node, minFont = 1e9;
  while ((node = walker.nextNode())) {
    if (!node.textContent.trim()) continue;
    const el = node.parentElement, cs = getComputedStyle(el);
    if (cs.visibility !== 'visible' || parseFloat(cs.opacity) < 0.99) return {ok: false, reason: 'hidden text'};
    minFont = Math.min(minFont, parseFloat(cs.fontSize));
    const range = document.createRange(); range.selectNodeContents(node);
    for (const r of range.getClientRects()) {
      if (r.width < 0.5 || r.height < 0.5) continue;
      if (r.left < -0.5 || r.top < -0.5 || r.right > vw + 0.5 || r.bottom > vh + 0.5) return {ok: false, reason: 'text outside viewport'};
      for (let a = el; a && a !== document.documentElement; a = a.parentElement) {
        const acs = getComputedStyle(a);
        if (acs.overflowX !== 'visible' || acs.overflowY !== 'visible') {
          if (a === document.body) continue;
          const b = a.getBoundingClientRect();
          if (r.left < b.left - 0.5 || r.top < b.top - 0.5 || r.right > b.right + 0.5 || r.bottom > b.bottom + 0.5)
            return {ok: false, reason: 'clipped text'};
        }
        if (acs.opacity !== '1') return {ok: false, reason: 'translucent ancestor'};
      }
    }
  }
  return {ok: true, text: document.body.innerText, minFont};
}
"""


class Fonts:
    def __init__(self, directory):
        self.directory = Path(directory)
        lock = json.loads((self.directory / "fonts.lock.json").read_text())
        faces, self.families = [], {"serif": [], "sans": [], "mono": []}
        by_family = {}
        for name, info in sorted(lock["files"].items()):
            family = info["family_dir"].split("/")[1]
            by_family.setdefault(family, (info["kind"], []))[1].append(name)
        for family, (kind, files) in sorted(by_family.items()):
            alias = "F" + family
            for name in files:
                weight = "700" if name.endswith("-Bold.ttf") else ("400" if name.endswith("-Regular.ttf") else "100 900")
                faces.append(f"@font-face{{font-family:'{alias}';src:url('fonts/{name}');font-weight:{weight}}}")
            generic = {"serif": "serif", "sans": "sans-serif", "mono": "monospace"}[kind]
            self.families[kind].append(f"'{alias}',{generic}")
        self.css = "\n".join(faces)
        self.lock = lock

    def pick(self, rng, kind):
        return rng.choice(self.families[kind])


class Renderer:
    def __init__(self, fonts):
        from playwright.sync_api import sync_playwright
        self.fonts = fonts
        self.directory = tempfile.TemporaryDirectory(prefix="ocr-render-")
        shutil.copytree(fonts.directory, Path(self.directory.name) / "fonts")
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(args=["--font-render-hinting=none", "--disable-lcd-text"])

    def render(self, html, size, max_chars, min_font):
        from PIL import Image
        path = Path(self.directory.name) / "page.html"
        path.write_text(html, encoding="utf-8")
        context = self.browser.new_context(viewport={"width": size[0], "height": size[1]}, device_scale_factor=1,
                                           java_script_enabled=True, offline=True)
        try:
            page = context.new_page()
            page.goto(path.as_uri(), wait_until="load")
            page.evaluate("document.fonts.ready.then(() => true)")
            result = page.evaluate(FIT_JS, max_chars)
            if not result["ok"]:
                return None, result["reason"]
            if result["minFont"] < min_font:
                return None, "font too small"
            png = page.screenshot(type="png", full_page=False)
        finally:
            context.close()
        # Re-encode without ancillary chunks (no text metadata survives).
        image = Image.open(io.BytesIO(png)).convert("RGB")
        if image.size != tuple(size):
            return None, "wrong size"
        out = io.BytesIO()
        image.save(out, format="PNG", optimize=True)
        return (out.getvalue(), result["text"]), "ok"

    def close(self):
        self.browser.close()
        self.playwright.stop()
        self.directory.cleanup()


def generate(renderer, corpus, seed, split, index, kind, bucket, max_attempts=40):
    low, high = next((lo, hi) for name, lo, hi in BUCKETS if name == bucket)
    reasons = {}
    for attempt in range(max_attempts):
        rng = random.Random(f"{seed}:{split}:{index}:{attempt}")
        maker = document if kind == "document" else screenshot
        layout, html, size, min_font = maker(rng, corpus, renderer.fonts, bucket)
        budget = rng.randint(low + (high - low) // 3, high - 1)
        output, reason = renderer.render(html, size, budget, min_font)
        if output is None:
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        png, text = output
        length = len(normalize(text))
        if bucket_of(length) != bucket:
            reasons["bucket"] = reasons.get("bucket", 0) + 1
            continue
        return {"png": png, "text": text, "layout": layout, "chars": length, "attempt": attempt}
    raise RuntimeError(f"Could not generate {split}/{index} {kind}/{bucket}: {reasons}")


def plan(count):
    """Balanced (kind, bucket) assignments: half documents, half screenshots, buckets round-robin."""
    names = [b[0] for b in BUCKETS]
    kinds = ["document", "screenshot"]
    return [(kinds[i % 2], names[(i // 2) % len(names)]) for i in range(count)]


SCHEMA = Path(__file__).with_name("SCHEMA.md")


def build(corpus_path, fonts_dir, out, validation, heldout, seed=None, progress=print):
    records = [json.loads(line) for line in Path(corpus_path).read_text().splitlines() if line.strip()]
    out = Path(out)
    # Operator-only provenance (never listed as a public file): frozen corpus and font lock.
    provenance = out / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(corpus_path, provenance / "corpus.jsonl")
    shutil.copyfile(Path(fonts_dir) / "fonts.lock.json", provenance / "fonts.lock.json")
    (out / "public").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SCHEMA, out / "public" / "SCHEMA.md")
    seed_file = out / "provenance" / "generator_seed.json"  # operator-only; never packaged
    if seed is None:
        seed = json.loads(seed_file.read_text())["seed"] if seed_file.exists() else secrets.token_hex(32)
    fonts = Fonts(fonts_dir)
    renderer = Renderer(fonts)
    try:
        for split, count, directory, name in (("validation", validation, out / "public", "validation.jsonl"),
                                              ("heldout", heldout, out / "private", "test.jsonl")):
            corpus = Corpus(records, seed, split)
            images = directory / "images"
            if images.exists():
                shutil.rmtree(images)
            images.mkdir(parents=True)
            rows = []
            for index, (kind, bucket) in enumerate(plan(count)):
                sample = generate(renderer, corpus, seed, split, index, kind, bucket)
                prefix = "v" if split == "validation" else "t"
                image_name = f"images/{prefix}{index:04d}.png"
                (directory / image_name).write_bytes(sample["png"])
                rows.append({"id": f"{split}-{index:04d}", "image": image_name, "text": sample["text"], "kind": kind,
                             "bucket": bucket, "layout": sample["layout"], "chars": sample["chars"],
                             "sha256": hashlib.sha256(sample["png"]).hexdigest()})
                if (index + 1) % 20 == 0:
                    progress(f"{split}: {index + 1}/{count}")
            (directory / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        seed_file.parent.mkdir(parents=True, exist_ok=True)
        seed_file.write_text(json.dumps({"seed": seed, "generator": GENERATOR_VERSION, "normalization": NORMALIZATION}) + "\n")
    finally:
        renderer.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--corpus", type=Path, required=True)
    b.add_argument("--fonts", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--validation", type=int, default=100)
    b.add_argument("--heldout", type=int, default=400)
    b.add_argument("--seed", help="private seed; default: reuse OUT/provenance/generator_seed.json or draw a new one")
    s = sub.add_parser("sample", help="render a few pages to a scratch directory for inspection")
    s.add_argument("--corpus", type=Path, required=True)
    s.add_argument("--fonts", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--count", type=int, default=16)
    s.add_argument("--seed", default="inspection")
    s.add_argument("--pool", choices=["validation", "heldout"], default="validation",
                   help="corpus pool; organizer training renders must never use the held-out pool")
    s.add_argument("--pool-seed", help="seed that assigns papers to pools (the build's private seed)")
    args = parser.parse_args(argv)
    if args.command == "build":
        build(args.corpus, args.fonts, args.out, args.validation, args.heldout, args.seed)
    else:
        records = [json.loads(line) for line in args.corpus.read_text().splitlines() if line.strip()]
        corpus = Corpus(records, args.pool_seed or args.seed, args.pool)
        renderer = Renderer(Fonts(args.fonts))
        args.out.mkdir(parents=True, exist_ok=True)
        try:
            for index, (kind, bucket) in enumerate(plan(args.count)):
                sample = generate(renderer, corpus, args.seed, "sample", index, kind, bucket)
                stem = f"{index:04d}-{kind}-{bucket}-{sample['layout']}"
                (args.out / f"{stem}.png").write_bytes(sample["png"])
                (args.out / f"{stem}.txt").write_text(sample["text"])
                print(stem, sample["chars"], "attempt", sample["attempt"], flush=True)
        finally:
            renderer.close()


if __name__ == "__main__":
    main()
