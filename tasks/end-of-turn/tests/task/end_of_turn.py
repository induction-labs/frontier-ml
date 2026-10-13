"""Operator-only import of lossless conversation audio and pinned consensus labels.

The task uses an actor-disjoint partition of the labelled development source.
It is not the official test leaderboard. Authentication uses ambient HF login.
"""
from __future__ import annotations
import argparse
from collections import Counter
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from turnbench_gold import Conversation, events_for_conversation

UPSTREAM_REVISION = "76ccd045f121ccfa921abac2ad3107027e736911"
DEV_REPO = "mundo-ai/turn-benchmark-dev"
DEV_REVISION = "8fa18a24be51528a45397b35cbcaecd84202062b"
TEST_REPO = "mundo-ai/turn-benchmark-test"
TEST_REVISION = "0c12a77a0134449757e4fd4144913bfcff0c376d"
PUBLIC_IDS = frozenset(("20", "21", "22", "94", "95", "122", "124", "125"))
SPLIT_VERSION = "eot-dev-split-v1"
ANNOTATION_COLUMNS = [f"speaker_{s}_annotation_{a}" for s in (1, 2) for a in "abc"]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def flac_info(blob):
    """Read STREAMINFO without decoding, resampling, or quantizing the audio."""
    if blob[:4] != b"fLaC":
        raise ValueError("Expected original lossless FLAC input")
    offset = 4
    while offset + 4 <= len(blob):
        kind, last = blob[offset] & 0x7f, bool(blob[offset] & 0x80)
        length = int.from_bytes(blob[offset + 1:offset + 4], "big")
        block = blob[offset + 4:offset + 4 + length]
        if kind == 0 and len(block) == 34:
            bits = int.from_bytes(block[10:18], "big")
            return {"sample_rate": bits >> 44, "channels": ((bits >> 41) & 7) + 1,
                    "bits_per_sample": ((bits >> 36) & 31) + 1, "frames": bits & ((1 << 36) - 1)}
        if last:
            break
        offset += 4 + length
    raise ValueError("Invalid FLAC STREAMINFO")


def gold_from_row(row, duration):
    annotations = {(s, a): [(e["start_s"], e["end_s"], e["label"], e["text"])
                   for e in row[f"speaker_{s}_annotation_{a}"]] for s in (1, 2) for a in "abc"}
    if not any(annotations.values()):
        return None
    conv = Conversation(str(row["conversation_id"]), duration, annotations)
    events = asdict(events_for_conversation(conv))
    return {k: events[k] for k in ("eot_positive_events", "eot_negative_spans", "eot_excluded")}


def source_shards(repo, revision, cache_dir=None):
    from huggingface_hub import HfApi, hf_hub_download
    files = sorted(n for n in HfApi().list_repo_files(repo, repo_type="dataset", revision=revision) if n.endswith(".parquet"))
    if not files:
        raise ValueError(f"No parquet shards in {repo}@{revision}")
    def fetch(name):
        return hf_hub_download(repo, name, repo_type="dataset", revision=revision, cache_dir=cache_dir)
    with ThreadPoolExecutor(max_workers=3) as pool:
        yield from pool.map(fetch, files)


def prepare_split(dest, *, repo, revision, split, cache_dir=None, selected=None):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    columns = ["conversation_id", "metadata", "speaker_1_audio", "speaker_2_audio", *ANNOTATION_COLUMNS]
    for shard in source_shards(repo, revision, cache_dir):
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=1, columns=columns):
            raw = batch.to_pylist()[0]
            cid = str(raw["conversation_id"])
            if selected is not None and not selected(cid):
                continue
            if not cid.isdecimal():
                raise ValueError("Unexpected conversation identifier")
            audio, infos, hashes = {}, [], {}
            for speaker in (1, 2):
                key = f"speaker_{speaker}"
                blob = raw[key + "_audio"]["bytes"]
                info = flac_info(blob)
                if (info["sample_rate"], info["channels"], info["bits_per_sample"]) != (48000, 1, 24):
                    raise ValueError(f"{cid}: expected original 48 kHz mono 24-bit FLAC")
                relative = f"audio/{cid}/{key}.flac"
                target = dest / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(blob)
                audio[key], hashes[key] = relative, hashlib.sha256(blob).hexdigest()
                infos.append(info)
            if infos[0] != infos[1]:
                raise ValueError(f"{cid}: channels are not aligned")
            duration = infos[0]["frames"] / infos[0]["sample_rate"]
            gold = gold_from_row(raw, duration)
            rows.append({"id": cid, "conversation_id": cid, "duration_s": duration, "audio": audio,
                         "audio_sha256": hashes, "metadata": raw["metadata"], "gold": gold,
                         "annotations": {name: raw[name] for name in ANNOTATION_COLUMNS}})
            print(json.dumps({"split": split, "conversation": cid, "seconds": duration, "labelled": gold is not None}), flush=True)
    rows.sort(key=lambda row: int(row["id"]))
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate conversations in source")
    (dest / f"{split}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    # The license and attribution travel in the dataset's manifest.json (see prepare), never in public/: the
    # agent-visible payload must not name the held-out source or revision (docs/TASKS.md; 2026-09-30).
    license_path = hf_hub_download(repo, "LICENSE", repo_type="dataset", revision=revision, cache_dir=cache_dir)
    return {"repo": repo, "revision": revision, "conversations": len(rows), "license_text": Path(license_path).read_text(),
            "attribution_notice": (f"TurnBench conversation data and annotations, Mundo AI Inc.\n"
                                   f"Source: https://huggingface.co/datasets/{repo}/tree/{revision}\n"
                                   f"Consensus derived with SesameAILabs/turnbench at {UPSTREAM_REVISION}.\n"
                                   "Audio is copied losslessly; EOT regions and JSONL packaging are adaptations.\n"
                                   "The accompanying Dataset Public License v1.0 applies.\n"),
            "labelled_conversations": sum(row["gold"] is not None for row in rows),
            "conversation_ids": [row["id"] for row in rows], "hours": sum(row["duration_s"] for row in rows) / 3600}


def actors(row):
    return {row["metadata"][f"speaker_{s}_actor_id"] for s in (1, 2)}


def choose_public_ids(rows):
    """Metadata-only partition: whole actor-connected groups, 8 public conversations.

    Exhaustively minimize count error, absent conversation types, then L1 type
    distribution error. A salted hash resolves ties. No gold labels are read.
    """
    groups = []
    for row in sorted(rows, key=lambda row: int(row["id"])):
        joined = [row]
        group_actors = actors(row)
        retained = []
        for group in groups:
            if group_actors & set().union(*(actors(item) for item in group)):
                joined.extend(group)
                group_actors.update(*[actors(item) for item in group])
            else:
                retained.append(group)
        groups = retained + [joined]
    # Merge transitively after a bridge joins two existing groups.
    changed = True
    while changed:
        changed = False
        for i, j in combinations(range(len(groups)), 2):
            if set().union(*(actors(r) for r in groups[i])) & set().union(*(actors(r) for r in groups[j])):
                groups[i].extend(groups.pop(j))
                changed = True
                break
    totals = Counter(r["metadata"]["conversation_type"] for r in rows)
    best = None
    for mask in range(1, 2 ** len(groups) - 1):
        selected = [r for i, group in enumerate(groups) if mask >> i & 1 for r in group]
        counts = Counter(r["metadata"]["conversation_type"] for r in selected)
        remaining = totals - counts
        ids = sorted((r["id"] for r in selected), key=int)
        key = (abs(len(selected) - 8), len(totals.keys() - counts.keys()) + len(totals.keys() - remaining.keys()),
               sum(abs(counts[k] / len(selected) - totals[k] / len(rows)) for k in sorted(totals)),
               hashlib.sha256((SPLIT_VERSION + ":" + ",".join(ids)).encode()).hexdigest())
        if best is None or key < best[0]:
            best = (key, ids)
    if best is None:
        raise ValueError("Need at least two independent speaker groups")
    return frozenset(best[1])


def prepare_public(destination):
    return prepare_split(destination, repo=DEV_REPO, revision=DEV_REVISION, split="validation", selected=lambda cid: cid in PUBLIC_IDS)


def prepare_private(destination):
    return prepare_split(destination, repo=DEV_REPO, revision=DEV_REVISION, split="test", selected=lambda cid: cid not in PUBLIC_IDS)


def prepare(root, *, cache_dir=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"format": "conversation-eot-v2", "scorer_revision": UPSTREAM_REVISION,
                "status": "preparing", "split_version": SPLIT_VERSION,
                "evaluation_origin": "speaker-disjoint partition of official dev; not official test",
                "sources": {"dev": {"repo": DEV_REPO, "revision": DEV_REVISION}},
                "scalar_adapter": "Recall at FPR <= 0.05 with the threshold swept over the returned per-frame "
                                  "probability streams (1 s late window); the upstream matcher is not used."}
    write_json(root / "manifest.json", manifest)
    manifest["validation"] = prepare_split(root / "public", repo=DEV_REPO, revision=DEV_REVISION, split="validation", cache_dir=cache_dir,
                                           selected=lambda cid: cid in PUBLIC_IDS)
    manifest["test"] = prepare_split(root / "private", repo=DEV_REPO, revision=DEV_REVISION, split="test", cache_dir=cache_dir,
                                     selected=lambda cid: cid not in PUBLIC_IDS)
    manifest["attribution"] = {"notice": manifest["validation"].pop("attribution_notice").strip(),
                               "license": "Dataset Public License v1.0 (Mundo AI Inc.)",
                               "license_text": manifest["validation"].pop("license_text")}
    manifest["test"].pop("attribution_notice"), manifest["test"].pop("license_text")
    audit = verify(root)
    manifest["status"] = "ready"
    manifest["split_audit"] = audit
    write_json(root / "manifest.json", manifest)
    return audit


def verify(root):
    root = Path(root)
    result = {"verified": True, "scorer_revision": UPSTREAM_REVISION}
    all_rows, split_rows = [], {}
    for name, folder in (("validation", "public"), ("test", "private")):
        rows = [json.loads(line) for line in (root / folder / f"{name}.jsonl").read_text().splitlines() if line.strip()]
        all_rows.extend(rows)
        split_rows[name] = rows
        totals = {"conversations": len(rows), "labelled": 0, "eot_positive": 0, "eot_negative": 0, "eot_excluded": 0}
        for row in rows:
            for speaker in ("speaker_1", "speaker_2"):
                blob = (root / folder / row["audio"][speaker]).read_bytes()
                if hashlib.sha256(blob).hexdigest() != row["audio_sha256"][speaker]:
                    raise ValueError("Audio checksum mismatch")
                info = flac_info(blob)
                if info["frames"] / info["sample_rate"] != row["duration_s"]:
                    raise ValueError("Audio duration mismatch")
            raw = {"conversation_id": row["id"], **row["annotations"]}
            if gold_from_row(raw, row["duration_s"]) != row["gold"]:
                raise ValueError("Consensus differs from pinned scorer")
            if row["gold"] is not None:
                totals["labelled"] += 1
                for key, source in (("eot_positive", "eot_positive_events"), ("eot_negative", "eot_negative_spans"), ("eot_excluded", "eot_excluded")):
                    totals[key] += len(row["gold"][source])
        result[name] = totals
    if choose_public_ids(all_rows) != PUBLIC_IDS or {r["id"] for r in split_rows["validation"]} != PUBLIC_IDS:
        raise ValueError("Prepared split differs from the metadata-only partition")
    public_actors = set().union(*(actors(r) for r in split_rows["validation"]))
    private_actors = set().union(*(actors(r) for r in split_rows["test"]))
    if public_actors & private_actors:
        raise ValueError("Speaker leakage between splits")
    public_hashes = {h for r in split_rows["validation"] for h in r["audio_sha256"].values()}
    private_hashes = {h for r in split_rows["test"] for h in r["audio_sha256"].values()}
    if public_hashes & private_hashes:
        raise ValueError("Audio leakage between splits")
    result["speaker_overlap"] = 0
    result["audio_overlap"] = 0
    result["public_speakers"], result["private_speakers"] = len(public_actors), len(private_actors)
    result["heldout_ready"] = result["test"]["labelled"] == result["test"]["conversations"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "prepare-public", "prepare-private", "prepare-unlabelled-test"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--cache", default=None)
    args = parser.parse_args()
    if args.command == "prepare-public":
        result = prepare_public(args.root)
    elif args.command == "prepare-private":
        result = prepare_private(args.root)
    elif args.command == "prepare-unlabelled-test":
        result = prepare_split(args.root, repo=TEST_REPO, revision=TEST_REVISION, split="test", cache_dir=args.cache)
    elif args.command == "prepare":
        result = prepare(args.root, cache_dir=args.cache)
    else:
        result = verify(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
