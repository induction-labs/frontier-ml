"""Prepare data/babylm-replay: pinned corpus, public dev-split validation sample, private held-out set.

    python tasks/babylm-replay/babylm_data.py corpus   --raw WORK/raw --destination data/babylm-replay
    python tasks/babylm-replay/babylm_data.py heldout  --sources WORK/heldout --destination data/babylm-replay
    python tasks/babylm-replay/babylm_data.py verify   data/babylm-replay

`corpus` downloads (or reuses) the BabyLM-2026 strict-small train files and the BabyLM dev split at
pinned revisions, checks the train hashes against the verifier pin, writes public/corpus/ and a
hardlinked private/corpus/, and samples public/validation.jsonl from the dev split. `heldout` merges the
per-genre JSONL files produced by the sourcing scripts (see provenance.json in that directory),
re-checks 13-gram decontamination against train+dev+test, and writes private/heldout.jsonl.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import unicodedata
import urllib.request

TRAIN_REPO, TRAIN_REV = "BabyLM-community/BabyLM-2026-Strict-Small", "c92ab16b4f08858304b0815706065b3354d8fc0a"
DEV_REPO, DEV_REV = "BabyLM-community/BabyLM-dev", "169f42e32d0aaf65ec6b91d55bafad27a3afc729"
TEST_REPO, TEST_REV = "BabyLM-community/BabyLM-Test", "2c47b98e2dc3707465aed81da69dc36cdca5d13b"
GENRES = ("bnc_spoken", "childes", "gutenberg", "open_subtitles", "simple_wiki", "switchboard")
VALIDATION_BYTES = 1_500_000
DOC_MIN, DOC_MAX = 2048, 16384


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch(repo, rev, name, target):
    target = Path(target)
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{name}"
        with urllib.request.urlopen(url, timeout=300) as response, open(str(target) + ".part", "wb") as out:
            shutil.copyfileobj(response, out)
        os.replace(str(target) + ".part", target)
    return target


def chunk_lines(text, rng, lo=DOC_MIN, hi=DOC_MAX):
    """Contiguous chunks of whole lines with random target sizes in [lo, hi] bytes."""
    lines = text.splitlines(keepends=True)
    chunks, cur, size, target = [], [], 0, rng.randint(lo, hi)
    for line in lines:
        n = len(line.encode("utf-8"))
        if cur and size + n > target:
            chunks.append("".join(cur))
            cur, size, target = [], 0, rng.randint(lo, hi)
        if n > hi:
            continue
        cur.append(line)
        size += n
    if cur and size >= lo:
        chunks.append("".join(cur))
    return chunks


def prepare_corpus(raw, destination, verifier_pins):
    raw, destination = Path(raw), Path(destination)
    public, private = destination / "public", destination / "private"
    (public / "corpus").mkdir(parents=True, exist_ok=True)
    (private / "corpus").mkdir(parents=True, exist_ok=True)
    manifest = {"corpus": {"repo": TRAIN_REPO, "revision": TRAIN_REV, "license": "MIT", "files": {}},
                "validation": {"repo": DEV_REPO, "revision": DEV_REV}}
    for genre in GENRES:
        name = f"{genre}.train.txt"
        source = fetch(TRAIN_REPO, TRAIN_REV, name, raw / "train" / name)
        digest = sha256(source)
        if digest != verifier_pins[name]:
            raise SystemExit(f"{name}: hash {digest} differs from the verifier pin")
        for target in (public / "corpus" / name, private / "corpus" / name):
            if target.exists():
                target.unlink()
            shutil.copyfile(source, target)
        manifest["corpus"]["files"][name] = {"sha256": digest, "bytes": source.stat().st_size}
    # Validation: stratified contiguous chunks of the dev split, proportional to corpus bytes.
    rng = random.Random(20260922)
    total_train = sum(v["bytes"] for v in manifest["corpus"]["files"].values())
    rows = []
    for genre in GENRES:
        dev = fetch(DEV_REPO, DEV_REV, f"{genre}.dev", raw / "dev" / f"{genre}.dev")
        text = unicodedata.normalize("NFC", dev.read_text(encoding="utf-8"))
        chunks = chunk_lines(text, rng)
        rng.shuffle(chunks)
        budget = max(DOC_MAX, VALIDATION_BYTES * manifest["corpus"]["files"][f"{genre}.train.txt"]["bytes"] // total_train)
        used = 0
        for chunk in chunks:
            if used >= budget:
                break
            rows.append({"genre": genre, "text": chunk})
            used += len(chunk.encode("utf-8"))
    rng.shuffle(rows)
    for i, row in enumerate(rows):
        row["id"] = f"dev-{i:05d}"
    with open(public / "validation.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({"id": row["id"], "genre": row["genre"], "text": row["text"]}, ensure_ascii=False) + "\n")
    manifest["validation"].update(documents=len(rows), bytes=sum(len(r["text"].encode()) for r in rows),
                                  sha256=sha256(public / "validation.jsonl"))
    write_manifest(destination, manifest)
    return manifest


def write_manifest(destination, update):
    path = Path(destination) / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest.update(update)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def ngram_hashes(text, n=13):
    words = text.lower().split()
    return {hash(" ".join(words[i:i + n])) for i in range(len(words) - n + 1)}


def prepare_heldout(sources, destination, raw):
    sources, destination, raw = Path(sources), Path(destination), Path(raw)
    rows = []
    for path in sorted(sources.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    reference = set()
    for folder, pattern in (("train", "*.train.txt"), ("dev", "*.dev"), ("test", "*.test")):
        for path in sorted((raw / folder).glob(pattern)):
            for line in path.read_text(encoding="utf-8").splitlines():
                reference |= ngram_hashes(line)
    kept, dropped, seen = [], 0, set()
    for row in rows:
        text = row["text"]
        if not isinstance(text, str) or unicodedata.normalize("NFC", text) != text or not text.endswith("\n"):
            raise SystemExit(f"{row.get('id')}: text must be NFC and end with a newline")
        if not DOC_MIN // 2 <= len(text.encode()) <= DOC_MAX + 1024 or re.search(r"[\x00-\x08\x0b-\x1f\x7f]", text):
            dropped += 1
            continue
        key = hashlib.sha256(text.encode()).hexdigest()
        if key in seen or ngram_hashes(text) & reference:
            dropped += 1
            continue
        seen.add(key)
        kept.append({"id": row["id"], "genre": row["genre"], "text": text, "source": row.get("source")})
    random.Random(7).shuffle(kept)
    private = destination / "private"
    private.mkdir(parents=True, exist_ok=True)
    with open(private / "heldout.jsonl", "w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps({"id": row["id"], "genre": row["genre"], "text": row["text"]}, ensure_ascii=False) + "\n")
    with open(private / "heldout_sources.jsonl", "w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps({"id": row["id"], "source": row["source"]}, ensure_ascii=False) + "\n")
    genres = {}
    for row in kept:
        g = genres.setdefault(row["genre"], {"documents": 0, "bytes": 0})
        g["documents"] += 1
        g["bytes"] += len(row["text"].encode())
    provenance = sources / "provenance.json"
    write_manifest(destination, {"heldout": {"documents": len(kept), "dropped_in_merge": dropped, "genres": genres,
                                             "bytes": sum(g["bytes"] for g in genres.values()),
                                             "sha256": sha256(private / "heldout.jsonl"),
                                             "provenance": json.loads(provenance.read_text()) if provenance.exists() else None}})
    return len(kept)


def verify(destination):
    destination = Path(destination)
    manifest = json.loads((destination / "manifest.json").read_text())
    for name, info in manifest["corpus"]["files"].items():
        for split in ("public", "private"):
            if sha256(destination / split / "corpus" / name) != info["sha256"]:
                raise SystemExit(f"{split}/corpus/{name} hash mismatch")
    if sha256(destination / "public" / "validation.jsonl") != manifest["validation"]["sha256"]:
        raise SystemExit("validation.jsonl hash mismatch")
    if "heldout" in manifest and sha256(destination / "private" / "heldout.jsonl") != manifest["heldout"]["sha256"]:
        raise SystemExit("heldout.jsonl hash mismatch")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("documents", "bytes")} for k, v in manifest.items()
                      if isinstance(v, dict) and "documents" in v}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("corpus")
    c.add_argument("--raw", required=True)
    c.add_argument("--destination", required=True)
    h = sub.add_parser("heldout")
    h.add_argument("--sources", required=True)
    h.add_argument("--raw", required=True)
    h.add_argument("--destination", required=True)
    v = sub.add_parser("verify")
    v.add_argument("destination")
    args = parser.parse_args(argv)
    if args.command == "corpus":
        from frontier_ml.taskspec import load_task
        pins = load_task("babylm-replay").module.CORPUS_SHA256
        print(json.dumps(prepare_corpus(args.raw, args.destination, pins)["validation"]))
    elif args.command == "heldout":
        print(prepare_heldout(args.sources, args.destination, args.raw))
    else:
        verify(args.destination)


if __name__ == "__main__":
    main()
