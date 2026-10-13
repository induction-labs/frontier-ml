"""Trusted DetectRL-X importer; never shipped to the candidate.

python tasks/ai-text-detection/ai_text_detection.py prepare --cache CACHE --destination NEW_DIRECTORY
python tasks/ai-text-detection/ai_text_detection.py develop --pool POOL_DIRECTORY --destination NEW_DIRECTORY
python tasks/ai-text-detection/ai_text_detection.py verify --destination data/ai-text-detection

`prepare` streams the pinned, ungated Binary/*_open.json release into a stratified pool of source groups
(a validation partition and a held-out partition) and publishes the payload: the held-out partition is
shipped whole as private/test.jsonl, and public/validation.jsonl is a small balanced development sample
drawn from the validation partition with one language withheld entirely (`--pool-only` ships the whole
pool instead). `develop` repeats only the publishing step from a pool payload. No model/API calls.
The source lock beside this module records upstream LFS SHA-256 digests.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import random
import urllib.request

SOURCE_LOCK = Path(__file__).with_name("ai_text_detection_sources.json")
LANGUAGES = {"arabic": "ar", "chinese": "zh", "english": "en", "french": "fr", "german": "de",
             "portuguese": "pt", "russian": "ru", "spanish": "es"}
MODELS = {name: f"generator-{i + 1}" for i, name in enumerate(
    ("deepseek-v3", "gemini-2.5-flash", "gpt-4o", "qwen-max"))}
DOMAINS = ("Academic", "News", "Novel", "SEO", "Wiki", "Webtext")
CONDITIONS = ("general", "general_64", "general_128", "general_256", "general_512", "backtranslation",
              "encoder_paraphrasing", "decoder_paraphrasing", "seq2seq_paraphrasing", "character_deletion",
              "character_insertion", "character_substitution", "zero_width_insertion", "polishing", "expanding", "condensing")
SEED = "multilingual-binary-v1"
SCHEMA = "multilingual-binary-v1"
DEVELOPMENT_ROWS = 200  # public development sample: half human, half AI, one row per source group


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_meta(path):
    with Path(path).open("rb") as stream:
        return {"bytes": Path(path).stat().st_size, "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}


def iter_array(path, chunk_size=1024 * 1024):
    """Stream a top-level JSON array without loading a gigabyte into memory; reject truncation."""
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8") as stream:
        buffer, offset, eof, state = "", 0, False, "start"
        while True:
            while offset < len(buffer) and buffer[offset].isspace():
                offset += 1
            if offset == len(buffer) and not eof:
                buffer, offset = stream.read(chunk_size), 0
                eof = not buffer
                continue
            char = buffer[offset:offset + 1]
            if state == "start":
                if char != "[":
                    raise ValueError(f"Expected JSON array: {path}")
                offset, state = offset + 1, "first"
            elif state == "done":
                if char:
                    raise ValueError(f"Trailing content: {path}")
                if eof:
                    return
            elif char == "]" and state in ("first", "separator"):
                offset, state = offset + 1, "done"
            elif state == "separator":
                if char != ",":
                    raise ValueError(f"Expected array separator: {path}")
                offset, state = offset + 1, "item"
            else:
                try:
                    row, end = decoder.raw_decode(buffer, offset)
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError(f"Incomplete/invalid JSON: {path}") from None
                    chunk = stream.read(chunk_size)
                    buffer, offset = buffer[offset:] + chunk, 0
                    eof = not chunk
                    continue
                if not isinstance(row, dict):
                    raise ValueError(f"Expected object row: {path}")
                yield row
                offset, state = end, "separator"
            if eof and offset == len(buffer) and state != "done":
                raise ValueError(f"Incomplete JSON array: {path}")


def fetch_sources(cache, workers=4):
    """Verify cached files, otherwise download to a temporary path then verify and rename."""
    lock = json.loads(SOURCE_LOCK.read_text())
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)

    def fetch(item):
        relative, expected = item
        path = cache / Path(relative).name
        if path.exists():
            if file_meta(path) != expected:
                raise ValueError(f"Cached source checksum mismatch: {path}")
            return
        part = path.with_suffix(".partial")
        url = f"https://huggingface.co/datasets/{lock['repo']}/resolve/{lock['revision']}/{relative}?download=true"
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=120) as source, part.open("wb") as target:
                    while chunk := source.read(4 * 1024 * 1024):
                        target.write(chunk)
                if file_meta(part) != expected:
                    raise ValueError(f"Downloaded source checksum mismatch: {relative}")
                part.replace(path)
                return
            except (OSError, ValueError):
                if attempt == 2:
                    raise
        raise AssertionError("unreachable")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, lock["files"].items()))
    return lock


def source_group(row):
    # Whitespace-only differences must not move a source into the other split.
    return digest(" ".join(row["human_written_text"].split()))


def check_row(row):
    if row.get("lang") not in LANGUAGES or row.get("model") not in MODELS or row.get("domain") not in DOMAINS:
        raise ValueError("Unrecognized source language, generator or domain")
    if row.get("split") not in ("train", "test"):
        raise ValueError("Unrecognized upstream split")
    if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ("human_written_text", "llm_generated_text")):
        raise ValueError("Missing or empty source text")


def aligned(base, variant, condition):
    """No source IDs are provided: check metadata and human text before using positional correspondence."""
    if any(base[k] != variant[k] for k in ("lang", "model", "domain", "split")):
        return False
    original, current = base["human_written_text"], variant["human_written_text"]
    if condition.startswith("general_"):
        # The release's tokenizer can introduce spaces next to punctuation when decoding an excerpt.
        # This is only an alignment key; emitted text is never normalized.
        return "".join(original.split()).startswith("".join(current.split()))
    return original == current


def select_groups(base_rows, validation_per_cell, heldout_per_cell, seed=SEED):
    """Stratify by language/generator/domain; omit ambiguous duplicate and train-overlapping anchors."""
    training = {source_group(row) for row in base_rows if row["split"] == "train"}
    groups = defaultdict(list)
    for index, row in enumerate(base_rows):
        if row["split"] == "test":
            groups[source_group(row)].append((index, row))
    cells = defaultdict(list)
    excluded = Counter()
    for group, members in groups.items():
        if group in training:
            excluded["overlaps_upstream_train"] += 1
        elif len(members) != 1:
            excluded["duplicate_source_text"] += 1
        else:
            row = members[0][1]
            cells[row["lang"], row["model"], row["domain"]].append(group)
    expected = {(lang, model, domain) for lang in LANGUAGES for model in MODELS for domain in DOMAINS}
    if set(cells) != expected:
        raise ValueError(f"Incomplete source coverage: {expected - set(cells)}")
    selection = {}
    for cell, candidates in sorted(cells.items()):
        ordered = sorted(candidates, key=lambda g: digest(f"{seed}:{g}"))
        needed = validation_per_cell + heldout_per_cell
        if len(ordered) < needed:
            raise ValueError(f"Insufficient complete groups for {cell}: {len(ordered)} < {needed}")
        for i, group in enumerate(ordered[:needed]):
            split = "validation" if i < validation_per_cell else "heldout"
            for index, _ in groups[group]:
                selection[index] = (split, group)
    return selection, dict(excluded)


def split_audit(rows):
    groups = [{r["group"] for r in rows[split]} for split in ("validation", "heldout")]
    texts = [{digest(r["text"]) for r in rows[split]} for split in ("validation", "heldout")]
    normalized = [{digest(" ".join(r["text"].split())) for r in rows[split]} for split in ("validation", "heldout")]
    labels = defaultdict(set)
    for samples in rows.values():
        for row in samples:
            labels[digest(row["text"])].add(row["label"])
    audit = {"group_overlap": len(groups[0] & groups[1]), "input_fingerprint_overlap": len(texts[0] & texts[1]),
             "whitespace_normalized_input_overlap": len(normalized[0] & normalized[1]),
             "conflicting_label_inputs": sum(len(values) > 1 for values in labels.values())}
    audit["passed"] = not any(audit.values())
    audit["limitations"] = ("Public open-release test labels, not the organizers' hidden test. Source grouping uses "
                            "human text because document/author/site IDs are absent; near-duplicate sources may remain.")
    return audit


def heldout_language(rows):
    """The language withheld from the public sample: the lower-median pool row count (languages sorted by
    their rows in the whole pool, then by code, position (n - 1) // 2), so it is neither the sparsest nor the densest."""
    counts = Counter(row["language"] for row in rows)
    ordered = sorted(counts, key=lambda language: (counts[language], language))
    return ordered[(len(ordered) - 1) // 2]


def select_development(rows, withheld, count=DEVELOPMENT_ROWS, seed=SEED):
    """count/2 human and count/2 AI rows from distinct source groups of the remaining languages. Each language's
    quota is spread over conditions, then generator families, then domains; ties use a seeded SHA-256 of the row ID."""
    if type(count) is not int or count < 2 or count % 2:
        raise ValueError("The development sample needs a positive even row count")
    languages = sorted({row["language"] for row in rows} - {withheld}, key=lambda l: digest(f"{seed}:development:{l}"))
    if withheld not in {row["language"] for row in rows} or not languages:
        raise ValueError("The withheld language must be one of several pool languages")
    half, used, selected = count // 2, set(), []
    for label in (1, 0):
        extra = half % len(languages)
        for index, language in enumerate(languages):
            # The two labels put their odd rows on different languages, so language totals stay even-handed.
            quota = half // len(languages) + ((index - (0 if label else extra)) % len(languages) < extra)
            candidates = [r for r in rows if r["label"] == label and r["language"] == language]
            seen = Counter()
            for _ in range(quota):
                eligible = [r for r in candidates if r["group"] not in used]
                if not eligible:
                    raise ValueError(f"Insufficient source groups for the development sample: {language}")
                pick = min(eligible, key=lambda r: (seen["condition", r["condition"]], seen["family", r["family"]],
                                                     seen["domain", r["domain"]], digest(f"{seed}:development:{r['id']}")))
                used.add(pick["group"])
                for key in ("condition", "family", "domain"):
                    seen[key, pick[key]] += 1
                selected.append(pick)
    return sorted(selected, key=lambda r: digest(f"{seed}:development:{r['id']}"))


def publish(pool, destination, lock, sampling, retrieved_at, development_rows=DEVELOPMENT_ROWS):
    """Write a payload from a pool with row IDs assigned: the held-out partition whole as private/test.jsonl, and
    public/validation.jsonl as the development sample (or the whole validation partition when development_rows is None)."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}; publish into a new directory")
    seed, rows, sampling = sampling["seed"], dict(pool), dict(sampling)
    if development_rows is not None:
        withheld = heldout_language(pool["validation"] + pool["heldout"])
        sample = select_development(pool["validation"], withheld, development_rows, seed)
        rows["validation"] = [dict(row, id=f"validation-{i:06d}") for i, row in enumerate(sample)]
        pool_counts = Counter(row["language"] for row in pool["validation"] + pool["heldout"])
        sampling["development"] = {
            "rows": len(sample), "one_row_per_source_group": True, "heldout_language": withheld,
            "heldout_language_rule": "lower-median pool row count: languages sorted by (pool rows, code), position (n - 1) // 2",
            "pool_rows_per_language": dict(sorted(pool_counts.items())), "pool_validation_rows": len(pool["validation"]),
            "unpublished": "the validation partition's other rows, including every row of the withheld language",
            "pool_ids": [row["id"] for row in sample]}
    audit = split_audit(rows)
    if not audit["passed"]:
        raise ValueError(f"Cross-split leakage: {audit}")
    paths = {"validation": "public/validation.jsonl", "heldout": "private/test.jsonl"}
    for split, relative in paths.items():
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        # ASCII JSON escapes preserve decoded text, and keep U+0085/U+2028/U+2029 inside one
        # record even for consumers that use str.splitlines() instead of physical file lines.
        path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows[split]), encoding="utf-8")
    sample_note = ("" if development_rows is None else
                   f"validation.jsonl is a development sample of {development_rows} passages (half human, half AI, one per "
                   "source group) for checking a server and the metric; it is not training data, and the scored held-out "
                   "set includes at least one language that does not appear in it. ")
    schema = ("# Binary text detection data\n\nEach JSONL row has id, text, label (0 human, 1 AI), language, "
              "family (an anonymous generator ID), domain, condition, and group. Only text is sent to /detect. "
              "Texts are unmodified and can include Unicode perturbations. " + sample_note +
              "Groups identify the source document of related variants. "
              "Repeated human text is included once; AI variants retain their condition membership. "
              "The score equally weights language × family × condition cells at a global human threshold.\n")
    (destination / "public/SCHEMA.md").write_text(schema)
    manifest = {"schema_version": 1, "task": "ai-text-detection", "dataset_schema": SCHEMA, "source": lock,
                "retrieved_at": retrieved_at, "sampling": sampling,
                "splits": {s: len(rs) for s, rs in rows.items()},
                "counts": {s: {"human": sum(r["label"] == 0 for r in rs), "ai": sum(r["label"] == 1 for r in rs),
                               "groups": len({r["group"] for r in rs}),
                               "languages": dict(sorted(Counter(r["language"] for r in rs).items())),
                               "domains": dict(sorted(Counter(r["domain"] for r in rs).items())),
                               "families": dict(sorted(Counter(r["family"] for r in rs).items())),
                               "human_conditions": dict(sorted(Counter(r["condition"] for r in rs if not r["label"]).items())),
                               "ai_conditions": dict(sorted(Counter(r["condition"] for r in rs if r["label"]).items()))}
                           for s, rs in rows.items()}, "split_audit": audit,
                "files": {str(p.relative_to(destination)): file_meta(p) for p in sorted(destination.rglob("*")) if p.is_file()}}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    verify(destination)
    return manifest


def prepare(cache, destination, validation_per_cell=4, heldout_per_cell=20, development_rows=DEVELOPMENT_ROWS, seed=SEED):
    if min(validation_per_cell, heldout_per_cell) < 1:
        raise ValueError("Both split sizes must be positive")
    destination, cache = Path(destination), Path(cache)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}; prepare into a new directory")
    lock = fetch_sources(cache)
    logging.info("Verified all pinned source files")
    base = list(iter_array(cache / "binary_general_open.json"))
    for row in base:
        check_row(row)
    selected, exclusions = select_groups(base, validation_per_cell, heldout_per_cell, seed)
    groups = sorted({group for _, group in selected.values()}, key=lambda g: digest(f"id:{seed}:{g}"))
    neutral_group = {g: f"source-{i:05d}" for i, g in enumerate(groups)}
    rows = {"validation": [], "heldout": []}
    human_seen = {split: set() for split in rows}
    for condition in CONDITIONS:
        logging.info("Preparing %s", condition)
        count = 0
        for index, variant in enumerate(iter_array(cache / f"binary_{condition}_open.json")):
            count += 1
            if index not in selected:
                continue
            check_row(variant)
            if not aligned(base[index], variant, condition):
                raise ValueError(f"Variant/source alignment failure: {condition}, row {index}")
            split, group = selected[index]
            common = {"language": LANGUAGES[variant["lang"]], "domain": variant["domain"].lower(),
                      "group": neutral_group[group], "condition": condition}
            # Repeated human anchors are one negative, regardless of how many generators/attacks reuse them.
            human = variant["human_written_text"]
            key = digest(human)
            if key not in human_seen[split]:
                rows[split].append({**common, "text": human, "label": 0, "family": "human"})
                human_seen[split].add(key)
            rows[split].append({**common, "text": variant["llm_generated_text"], "label": 1,
                                "family": MODELS[variant["model"]]})
        if count != len(base):
            raise ValueError(f"Variant row count mismatch: {condition}, {count} != {len(base)}")
    for split, samples in rows.items():
        # Pool row order and IDs are fixed here; the development sample is drawn from these IDs.
        random.Random(f"{seed}:{split}:row-order").shuffle(samples)
        for i, row in enumerate(samples):
            row["id"] = f"{split}-{i:06d}"
    sampling = {"seed": seed, "validation_groups_per_language_generator_domain": validation_per_cell,
                "heldout_groups_per_language_generator_domain": heldout_per_cell, "upstream_split": "test",
                "excluded_groups": exclusions, "generator_aliases": MODELS,
                "source_row_indices": {neutral_group[group]: index for index, (_, group) in selected.items()}}
    return publish(rows, destination, lock, sampling, datetime.now(timezone.utc).date().isoformat(), development_rows)


def develop(pool, destination, development_rows=DEVELOPMENT_ROWS):
    """Publish the development payload from a verified pool payload (one whose public/validation.jsonl is the
    whole validation partition, such as the payload `prepare --pool-only` writes)."""
    manifest = verify(pool)
    if "development" in manifest["sampling"]:
        raise ValueError("develop needs a pool payload, not one that already carries a development sample")
    return publish(load_rows(pool), destination, manifest["source"], manifest["sampling"], manifest["retrieved_at"], development_rows)


def load_rows(directory):
    directory = Path(directory)
    return {s: [json.loads(line) for line in (directory / p).read_text().splitlines()]
            for s, p in (("validation", "public/validation.jsonl"), ("heldout", "private/test.jsonl"))}


def check_cells(samples, per_cell):
    """Every language/generator/condition/domain cell present with per_cell positives; no repeated human negative."""
    expected = {(lang, family, condition, domain.lower()) for lang in LANGUAGES.values() for family in MODELS.values()
                for condition in CONDITIONS for domain in DOMAINS}
    cells = Counter((r["language"], r["family"], r["condition"], r["domain"]) for r in samples if r["label"] == 1)
    if set(cells) != expected:
        raise ValueError("Missing or unexpected language/generator/condition/domain cells")
    if set(cells.values()) != {per_cell}:
        raise ValueError("Unbalanced evaluation cells")
    if len({r["text"] for r in samples if r["label"] == 0}) != sum(r["label"] == 0 for r in samples):
        raise ValueError("Repeated human negative")


def check_development(sample, development):
    """Balanced labels, one row per source group, and every language except the withheld one."""
    withheld = development["heldout_language"]
    if len(sample) != development["rows"] or Counter(r["label"] for r in sample) != {0: len(sample) // 2, 1: len(sample) // 2}:
        raise ValueError("Unbalanced development sample")
    if len({r["group"] for r in sample}) != len(sample):
        raise ValueError("Development sample repeats a source group")
    if withheld not in LANGUAGES.values() or {r["language"] for r in sample} != set(LANGUAGES.values()) - {withheld}:
        raise ValueError("Development sample must cover every language except the withheld one")


def verify(destination):
    destination = Path(destination)
    manifest = json.loads((destination / "manifest.json").read_text())
    if manifest.get("dataset_schema") != SCHEMA or manifest.get("source") != json.loads(SOURCE_LOCK.read_text()):
        raise ValueError("Wrong dataset schema or pinned source")
    for relative, expected in manifest["files"].items():
        if file_meta(destination / relative) != expected:
            raise ValueError(f"Payload checksum mismatch: {relative}")
    rows = load_rows(destination)
    for split, samples in rows.items():
        if len(samples) != manifest["splits"][split] or len({r["id"] for r in samples}) != len(samples):
            raise ValueError("Wrong split size or duplicate IDs")
        for row in samples:
            if (type(row.get("label")) is not int or row["label"] not in (0, 1)
                    or row.get("language") not in LANGUAGES.values() or row.get("condition") not in CONDITIONS
                    or not isinstance(row.get("text"), str) or not row["text"].strip()):
                raise ValueError("Invalid evaluation row")
    check_cells(rows["heldout"], manifest["sampling"]["heldout_groups_per_language_generator_domain"])
    if "development" in manifest["sampling"]:
        check_development(rows["validation"], manifest["sampling"]["development"])
    else:
        check_cells(rows["validation"], manifest["sampling"]["validation_groups_per_language_generator_domain"])
    audit = split_audit(rows)
    if not audit["passed"] or audit != manifest["split_audit"]:
        raise ValueError("Invalid split audit")
    return manifest


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("prepare", "develop", "verify"))
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--cache", type=Path, help="prepare: directory of pinned source files (downloaded when absent)")
    parser.add_argument("--pool", type=Path, help="develop: a verified pool payload written by `prepare --pool-only`")
    parser.add_argument("--validation-per-cell", type=int, default=4)
    parser.add_argument("--heldout-per-cell", type=int, default=20)
    parser.add_argument("--development-rows", type=int, default=DEVELOPMENT_ROWS)
    parser.add_argument("--pool-only", action="store_true", help="prepare: ship the whole pool instead of a development sample")
    args = parser.parse_args()
    if args.pool is not None and args.command != "develop":
        parser.error("--pool is develop's input pool; `prepare --pool-only` ships the whole pool")
    if args.pool_only and args.command != "prepare":
        parser.error("--pool-only applies to prepare")
    if args.command == "prepare":
        if args.cache is None:
            parser.error("prepare requires --cache")
        manifest = prepare(args.cache, args.destination, args.validation_per_cell, args.heldout_per_cell,
                           None if args.pool_only else args.development_rows)
    elif args.command == "develop":
        if args.pool is None:
            parser.error("develop requires --pool")
        manifest = develop(args.pool, args.destination, args.development_rows)
    else:
        manifest = verify(args.destination)
    summary = {"splits": manifest["splits"], "counts": manifest["counts"], "split_audit": manifest["split_audit"]}
    if "development" in manifest["sampling"]:
        summary["development"] = {k: v for k, v in manifest["sampling"]["development"].items() if k != "pool_ids"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
