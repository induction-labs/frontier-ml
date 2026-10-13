"""Fetch the fresh text corpus behind the ocr-latent-compression images (operator-only).

Two kinds of text, both created after every relevant model cutoff:

* arXiv metadata (titles, author names, abstracts) from OAI-PMH. arXiv distributes
  metadata under CC0 1.0, so abstracts of every license are usable.
* Full-text prose paragraphs and section titles from arXiv's HTML rendering, only for
  papers whose own license is CC BY 4.0, CC BY-SA 4.0 or CC0 1.0. Paragraphs with
  math markup are dropped so that ground truth stays plain text.

    python tasks/ocr-latent-compression/ocr_latent_sources.py OUT.jsonl --from 2026-09-01 --until 2026-09-21

Nothing here runs in the verifier. The frozen corpus is recorded with its SHA-256 in
the dataset manifest.
"""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import time
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET

OAI = "https://oaipmh.arxiv.org/oai"
NS = {"o": "http://www.openarchives.org/OAI/2.0/", "a": "http://arxiv.org/OAI/arXiv/"}
OPEN_LICENSES = {
    "http://creativecommons.org/licenses/by/4.0/": "CC-BY-4.0",
    "http://creativecommons.org/licenses/by-sa/4.0/": "CC-BY-SA-4.0",
    "http://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
}
SETS = ("cs", "math", "physics", "q-bio", "q-fin", "stat", "eess", "econ")
USER_AGENT = "frontier-mle-dataset-builder/1.0 (benchmark preparation; contact operator)"
# Characters that make ground truth ambiguous once rendered (markup, math, exotic symbols).
BAD = re.compile(r"[$\\{}^<>|~`#*_]|https?://|\bwww\.")


def fetch(url, *, retries=20, delay=5.0):
    """GET with curl: arXiv's CDN answers Python's urllib with HTTP 406 after a few requests."""
    import subprocess
    import tempfile
    for attempt in range(retries):
        with tempfile.NamedTemporaryFile() as body:
            try:
                result = subprocess.run(["curl", "-sS", "-L", "-m", "120", "-A", USER_AGENT, "-o", body.name,
                                         "-w", "%{http_code}", url], capture_output=True, text=True, timeout=150)
                status = int(result.stdout.strip() or 0)
            except (subprocess.TimeoutExpired, ValueError):
                status = 0
            if status == 200:
                return Path(body.name).read_bytes()
        if status == 404:
            return None
        wait = 60 if status in (406, 429, 503) else delay * (attempt + 1)
        print(f"HTTP {status} for {url}; retrying in {wait}s", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url}")


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def plain(text: str) -> bool:
    return bool(text) and not BAD.search(text) and all(ord(c) < 0x2500 for c in text)


def oai_records(arxiv_set, start, until, max_pages):
    params = {"verb": "ListRecords", "metadataPrefix": "arXiv", "from": start, "until": until, "set": arxiv_set}
    url = OAI + "?" + urllib.parse.urlencode(params)
    for _ in range(max_pages):
        root = ET.fromstring(fetch(url))
        for record in root.iterfind(".//o:record", NS):
            meta = record.find(".//a:arXiv", NS)
            if meta is None:
                continue
            get = lambda tag: (meta.findtext(f"a:{tag}", default="", namespaces=NS) or "")
            authors = []
            for author in meta.iterfind("a:authors/a:author", NS):
                names = [author.findtext(f"a:{t}", default="", namespaces=NS) for t in ("forenames", "keyname")]
                authors.append(clean(" ".join(n for n in names if n)))
            yield {"arxiv_id": get("id"), "created": get("created"), "categories": get("categories"),
                   "license": get("license"), "title": clean(get("title")), "abstract": clean(get("abstract")),
                   "authors": authors}
        token = root.findtext(".//o:resumptionToken", default="", namespaces=NS)
        if not token:
            return
        # The server returns the token already percent-encoded; do not encode it twice.
        url = OAI + "?verb=ListRecords&resumptionToken=" + token.strip()
        time.sleep(10)


class ArxivHTML(HTMLParser):
    """Collect section titles and math-free paragraphs from arXiv's LaTeXML HTML."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks, self.stack, self.buffer, self.kind, self.depth, self.tainted = [], [], [], None, 0, False
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        classes = dict(attrs).get("class", "") or ""
        if self.kind is not None:
            self.depth += 1
            if tag in ("math", "img", "svg", "table") or "ltx_note" in classes or "ltx_equation" in classes:
                self.tainted = True
            if tag in ("sup", "sub"):
                self.tainted = True
            return
        if tag == "p" and "ltx_p" in classes:
            self.kind, self.depth, self.buffer, self.tainted = "paragraph", 1, [], False
        elif tag in ("h2", "h3") and "ltx_title" in classes:
            self.kind, self.depth, self.buffer, self.tainted = "heading", 1, [], False

    def handle_endtag(self, tag):
        if self.kind is None:
            return
        self.depth -= 1
        if self.depth == 0:
            text = clean("".join(self.buffer))
            if not self.tainted and plain(text):
                if self.kind == "heading":
                    text = re.sub(r"^(\d+(\.\d+)*|[A-Z]\.?)\s+", "", text)
                self.blocks.append({"kind": self.kind, "text": text})
            self.kind = None

    def handle_data(self, data):
        if self.kind is not None:
            self.buffer.append(data)


def full_text(arxiv_id):
    raw = fetch(f"https://arxiv.org/html/{arxiv_id}")
    if not raw:
        return []
    parser = ArxivHTML()
    parser.feed(raw.decode("utf-8", "replace"))
    blocks = []
    for block in parser.blocks:
        if block["kind"] == "heading" and (len(block["text"]) > 80 or block["text"].lower() in {"references", "acknowledgments", "acknowledgements"}):
            continue
        if block["kind"] == "paragraph" and not 120 <= len(block["text"]) <= 1500:
            continue
        blocks.append(block)
    return blocks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path)
    parser.add_argument("--from", dest="start", required=True)
    parser.add_argument("--until", required=True)
    parser.add_argument("--created-after", default="2026-07-01", help="drop older papers that were merely updated")
    parser.add_argument("--pages-per-set", type=int, default=3)
    parser.add_argument("--full-text", type=int, default=300, help="maximum open-license papers to fetch full text for")
    args = parser.parse_args(argv)
    records, seen = [], set()
    for arxiv_set in SETS:
        for record in oai_records(arxiv_set, args.start, args.until, args.pages_per_set):
            if record["arxiv_id"] in seen or record["created"] < args.created_after:
                continue
            seen.add(record["arxiv_id"])
            record["abstract_plain"] = plain(record["abstract"])
            record["title_plain"] = plain(record["title"])
            records.append(record)
        print(arxiv_set, len(records), flush=True)
        time.sleep(10)
    fetched = 0
    for record in records:
        record["open_license"] = OPEN_LICENSES.get(record["license"])
        record["full_text"] = []
        if record["open_license"] and fetched < args.full_text:
            record["full_text"] = full_text(record["arxiv_id"])
            fetched += 1
            time.sleep(1)
            if fetched % 25 == 0:
                print("full text", fetched, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"records": len(records), "full_text_papers": sum(bool(r["full_text"]) for r in records)}))


if __name__ == "__main__":
    main()
