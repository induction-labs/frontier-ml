"""Import the operator's MIDI catalog and render it with amt_render's shared pipeline.

    uv run --with mido==1.3.3 --with pretty-midi==0.2.10 --with setuptools==75.8.0 \
      --with scipy==1.16.2 --with soundfile==0.13.1 python \
      tasks/music-transcription/pipeline/amt_catalog.py --catalog <catalog.xlsx> \
      --midi-zip <files.zip> --seed-file data/music-transcription.private-seed \
      --soundfont-dir <sf2-directory> --out data/music-transcription

Source files, credits, plans and audits are operator-only. Nothing here is imported
by validate.py. General MIDI programs are zero-based; channel 10 is percussion.
"""
import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import re
import shutil
import subprocess
from xml.etree import ElementTree as ET
from zipfile import ZipFile

DURATION = 30.0
STARTS_PER_SECOND = 1000  # sample uniformly over eligible millisecond start positions
MIN_PRESENT_SECONDS = 0.1  # one note must sound for at least 100 ms inside the window
# Retain these selections from the published 47-clip validation split. Keeping
# the original plan first preserves every excerpt, mix and soundfont assignment.
PREVIOUS_VALIDATION_SELECTIONS = frozenset({"N032", "N063", "N086", "N087", "N088", "N135", "N149", "N175", "N197"})
# Catalog styles/titles, not a listening-based genre annotation. Order preserves
# the two retained public IDs; each replacement trades places with its test clip.
VALIDATION_SELECTIONS = {
    "N067": "ragtime", "N076": "electronic/game music", "N063": "harmonium march",
    "N116": "strings and percussion", "N117": "rock instrumental", "N118": "jazz instrumental",
    "N135": "Romantic minuet", "N144": "contemporary chamber music",
    "N194": "recorder country dance", "N200": "contemporary ensemble",
}
# Exact taxonomy in the workbook's Coverage!A8:E16, not eight-program GM families.
PROGRAMS = {
    "piano": (0, 1, 2, 3, 4, 5),
    "mallets": (8, 9, 10, 11, 12, 13),
    "organ": (16, 17, 18, 19, 20, 21),
    "guitar": (24, 25, 26, 27, 28, 29, 30),
    "bass": (32, 33, 34, 35, 36, 37, 38, 39),
    "strings": (40, 41, 42, 44, 45, 48, 49),
    "brass": (56, 57, 58, 59, 60, 61),
    "reed": tuple(range(64, 72)),
    "pipe": (72, 73, 74, 75, 79),
}
PROGRAM_CLASS = {p: name for name, programs in PROGRAMS.items() for p in programs}
FONT_FILES = {"fluidr3": "FluidR3_GM.sf2", "musescore": "MuseScore_General.sf2",
              "generaluser": "GeneralUser-GS.sf2", "timgm6mb": "TimGM6mb.sf2"}
CHANGES = ("30-second excerpt sampled uniformly from eligible millisecond starts with maximum pitched class "
           "coverage, then maximum pitched GM-program coverage; short sources zero-padded. "
           "Notes crossing the left boundary are "
           "rearticulated at zero and offsets are clipped at 30 seconds. Original GM programs, pitches, "
           "velocities and tempo-derived timing retained. Playback controllers, pitch bends, SysEx and "
           "source effects omitted. Nonstandard drum-kit programs 2 and 27 normalized to standard GM kit 0; "
           "orchestral kit 48 retained. Each part rendered through the shared AMT FluidSynth/mix pipeline. "
           "Drums retained in audio only; exact same-class/pitch/onset unisons share one reference label.")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def program_class(program, is_drum=False):
    if is_drum:
        return None
    if program not in PROGRAM_CLASS:
        raise ValueError(f"Unmapped pitched GM program {program}")
    return PROGRAM_CLASS[program]


def workbook_rows(path):
    """Read cached XLSX cell values by sheet/column; no spreadsheet runtime needed."""
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(e.itertext()) for e in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        rels = {r.attrib["Id"]: r.attrib["Target"]
                for r in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))}
        sheets = ET.fromstring(archive.read("xl/workbook.xml")).find("s:sheets", ns)
        result = {}
        for sheet in sheets:
            rel = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            target = rels[rel]
            target = target.lstrip("/") if target.startswith("/") else "xl/" + target
            rows = []
            for row in ET.fromstring(archive.read(target)).findall(".//s:sheetData/s:row", ns):
                values = {}
                for cell in row:
                    value = cell.find("s:v", ns)
                    value = value.text if value is not None else ""
                    if cell.get("t") == "s":
                        value = strings[int(value)]
                    elif cell.get("t") == "inlineStr":
                        value = "".join(cell.find("s:is", ns).itertext())
                    if value:
                        values[re.sub(r"\d+$", "", cell.get("r"))] = value
                if values:
                    rows.append(values)
            result[sheet.get("name")] = rows
    return result


def parse_midi(raw):
    """Pair notes on (port, channel, pitch), capturing the program at note-on.

    Tempo is global; program state is per port/channel, including across tracks.
    Overlapping same-pitch attacks pair FIFO. Unmatched offs are counted, never
    turned into notes. Hanging/nonpositive notes and non-GM banks fail closed.
    """
    import mido
    midi = mido.MidiFile(file=io.BytesIO(raw))
    if midi.type not in (0, 1) or midi.ticks_per_beat <= 0:
        raise ValueError("Only synchronous, metrical MIDI files are supported")
    events, names = [], {}
    ignored = Counter()
    for track_index, track in enumerate(midi.tracks):
        tick, port = 0, 0
        for index, event in enumerate(track):
            tick += event.time
            if event.type == "midi_port":
                port = event.port
            if event.type == "track_name":
                names[track_index] = event.name
            events.append((tick, track_index, index, port, event))
    programs = defaultdict(int)
    banks = defaultdict(lambda: [0, 0])
    active = defaultdict(deque)
    tracks = {}
    tempo, last_tick, seconds = 500000, 0, 0.0
    extra_offs, defaults, explicit = 0, set(), set()
    note_ons = Counter()
    for tick, track_index, _, port, event in sorted(events):
        seconds += mido.tick2second(tick - last_tick, midi.ticks_per_beat, tempo)
        last_tick = tick
        if event.type == "set_tempo":
            tempo = event.tempo
            continue
        if not hasattr(event, "channel"):
            continue
        channel = (port, event.channel)
        if event.type == "program_change":
            programs[channel] = event.program
            explicit.add(channel)
        elif event.type == "control_change":
            if event.control in (0, 32):
                banks[channel][event.control == 32] = event.value
            ignored[event.type] += 1
        elif event.type == "note_on" and event.velocity > 0:
            drum = event.channel == 9
            if any(banks[channel]) and not (drum and banks[channel] in ([120, 0], [128, 0])):
                raise ValueError(f"Non-default MIDI bank on {channel}: {banks[channel]}")
            program = programs[channel]
            family = program_class(program, drum)
            key = (track_index, port, event.channel, program)
            if channel not in explicit and not drum:
                defaults.add(key)
            if key not in tracks:
                render_program = 0 if drum and program in (2, 27) else program
                tracks[key] = {"instrument": family, "program": render_program, "source_program": program, "is_drum": drum,
                               "role": "drums" if drum else "source", "source_track": track_index,
                               "source_name": names.get(track_index, ""), "port": port,
                               "channel": event.channel, "source_bank": list(banks[channel]), "notes": []}
            active[(*channel, event.note)].append((seconds, event.velocity, key))
            note_ons["drum" if drum else "pitched"] += 1
        elif event.type in ("note_on", "note_off"):
            queue = active[(*channel, event.note)]
            if not queue:
                extra_offs += 1
                continue
            onset, velocity, key = queue.popleft()
            if seconds <= onset:
                raise ValueError("Nonpositive source note duration")
            tracks[key]["notes"].append([onset, seconds, event.note, velocity])
        else:
            ignored[event.type] += 1
    if any(active.values()):
        raise ValueError("Unclosed MIDI notes")
    parts = list(tracks.values())
    for part in parts:
        part["notes"].sort()
    if not any(t["notes"] and not t["is_drum"] for t in parts):
        raise ValueError("No pitched notes")
    duration = max(n[1] for t in parts for n in t["notes"])
    return {"duration": duration, "tracks": parts}, {
        "note_ons": dict(note_ons), "extra_note_offs": extra_offs,
        "default_program_parts": len(defaults), "omitted_playback_events": dict(ignored),
        "programs": sorted({t["program"] for t in parts if not t["is_drum"]}),
    }


def labels(piece):
    # One event per audible class/pitch/onset, including unisons across source parts.
    unique = {}
    for track in piece["tracks"]:
        if track["is_drum"]:
            continue
        for onset, offset, pitch, _ in track["notes"]:
            key = (onset, track["instrument"], pitch)
            unique[key] = max(unique.get(key, 0), offset)
    return [{"onset": onset, "offset": offset, "pitch": pitch, "instrument": instrument}
            for (onset, instrument, pitch), offset in sorted(unique.items())]


def crop(piece, start):
    parts, crossing = [], 0
    for track in piece["tracks"]:
        notes = []
        for onset, offset, pitch, velocity in track["notes"]:
            if offset <= start or onset >= start + DURATION:
                continue
            left, right = round(max(0, onset - start), 4), round(min(DURATION, offset - start), 4)
            if right <= left:
                continue
            crossing += onset < start
            notes.append([left, right, pitch, velocity])
        if notes:
            parts.append({**track, "notes": notes})
    return {"duration": DURATION, "tracks": parts}, crossing


def _merge_intervals(intervals):
    """Union of inclusive integer intervals."""
    merged = []
    for low, high in sorted(intervals):
        if low > high:
            continue
        if merged and low <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    return merged


def excerpt_intervals(piece):
    """All eligible start intervals, in integer milliseconds, without an onset-only shortlist.

    A program is present when at least one of its notes overlaps the window by
    100 ms. Intersecting every program's valid starts can be impossible (instruments
    may occur in distant sections), so maximize classes first and programs second.
    If a window contains all source programs, only such windows remain eligible.
    """
    last = max(0, math.floor((piece["duration"] - DURATION) * STARTS_PER_SECOND + 1e-8))
    by_program = defaultdict(list)
    for track in piece["tracks"]:
        if track["is_drum"]:
            continue
        spans = by_program[track["program"]]
        for onset, offset, _, _ in track["notes"]:
            if offset - onset < MIN_PRESENT_SECONDS - 1e-9:
                continue
            low = math.ceil((onset + MIN_PRESENT_SECONDS - DURATION) * STARTS_PER_SECOND - 1e-8)
            high = math.floor((offset - MIN_PRESENT_SECONDS) * STARTS_PER_SECOND + 1e-8)
            spans.append((max(0, low), min(last, high)))
    events = defaultdict(list)
    events[0], events[last + 1] = [], []
    for program, spans in by_program.items():
        for low, high in _merge_intervals(spans):
            events[low].append((program, True))
            events[high + 1].append((program, False))
    active, best, eligible = set(), (0, 0), []
    boundaries = sorted(events)
    for low, end in zip(boundaries, boundaries[1:]):
        for program, present in events[low]:
            if present:
                active.add(program)
            else:
                active.remove(program)
        coverage = (len({program_class(p) for p in active}), len(active))
        if not coverage[0]:
            continue
        if coverage > best:
            best, eligible = coverage, []
        if coverage == best:
            eligible.append((low, end - 1))
    if not eligible:
        raise ValueError("No interval with pitched instruments present")
    return _merge_intervals(eligible), best, sorted(by_program)


def choose_excerpt(piece, rng):
    """Uniform seeded draw from all eligible millisecond starts, including sustained notes."""
    intervals, best, required = excerpt_intervals(piece)
    count = sum(high - low + 1 for low, high in intervals)
    pick = rng.randrange(count)
    for low, high in intervals:
        if pick <= high - low:
            start = (low + pick) / STARTS_PER_SECOND
            break
        pick -= high - low + 1
    excerpt, crossing = crop(piece, start)
    refs = labels(excerpt)
    if not refs or len(refs) > 10000:
        raise ValueError("Selected excerpt must contain 1–10,000 reference notes")
    present = sorted({t["program"] for t in excerpt["tracks"] if not t["is_drum"]})
    classes = {program_class(p) for p in present}
    if len(classes) < best[0] or len(present) < best[1]:
        raise ValueError("Cropping lost required instrument coverage")
    audit = {"policy": "uniform_maximum_instrument_coverage", "start_resolution_s": 1 / STARTS_PER_SECOND,
             "minimum_note_overlap_s": MIN_PRESENT_SECONDS, "eligible_start_count": count,
             "source_programs": required, "present_programs": present,
             "missing_programs": sorted(set(required) - set(present)),
             "missing_classes": sorted({program_class(p) for p in required} - classes),
             "all_source_programs_fit": best[1] == len(required)}
    return excerpt, start, crossing, audit


def load_catalog(catalog, midi_zip):
    sheets = workbook_rows(catalog)
    selections = {r["A"]: r for r in sheets["Selections"] if re.fullmatch(r"N\d{3}", r.get("A", ""))}
    downloads = {(r["A"], r["K"]): r for r in sheets["Downloads"]
                 if re.fullmatch(r"N\d{3}", r.get("A", ""))}
    mapping = {r["A"]: tuple(map(int, r["E"].split(","))) for r in sheets["Coverage"] if r.get("A") in PROGRAMS}
    if mapping != PROGRAMS:
        raise ValueError("Workbook taxonomy differs from the explicit GM mapping")
    with ZipFile(midi_zip) as archive:
        rights = {r["id"]: r for r in json.loads(archive.read("rights-and-attribution.json"))}
        entries = list(csv.DictReader(io.StringIO(archive.read("manifest.csv").decode())))
        members = [r["bundle_member"] for r in entries]
        if len(set(members)) != len(members) or set(members) != {
                n for n in archive.namelist() if n.lower().endswith((".mid", ".midi"))}:
            raise ValueError("Archive MIDI members do not match its manifest")
        if set(rights) != set(selections) or {r["selection_id"] for r in entries} != set(selections):
            raise ValueError("Workbook and archive selections differ")
        sources, seen = [], set()
        for entry in entries:
            selection = entry["selection_id"]
            raw = archive.read(entry["bundle_member"])
            sha = digest(raw)
            if sha != entry["sha256"] or sha in seen:
                raise ValueError(f"Hash mismatch or duplicate source: {entry['bundle_member']}")
            seen.add(sha)
            row = downloads[(selection, sha)]
            piece, audit = parse_midi(raw)
            if audit["programs"] != sorted(map(int, row["E"].split(","))):
                raise ValueError(f"Program audit differs: {entry['bundle_member']}")
            if audit["note_ons"]["pitched"] != int(row["V"]):
                raise ValueError(f"Pitched note count differs: {entry['bundle_member']}")
            families = {t["instrument"] for t in piece["tracks"] if not t["is_drum"]}
            if families != {v.strip() for v in re.split(r"[,;]", row["U"])}:
                raise ValueError(f"Class audit differs: {entry['bundle_member']}")
            credit = rights[selection]
            if credit["composer"] != selections[selection]["D"] or credit["license"] != selections[selection]["F"]:
                raise ValueError(f"Workbook and archive credits differ: {selection}")
            sources.append({"selection_id": selection, "member": entry["bundle_member"], "sha256": sha,
                            "credit": credit, "piece": piece, "audit": audit})
        if len(sources) != len(downloads):
            raise ValueError("Workbook download count differs from archive")
    return sources


def assign_splits(sources, seed, validation_count):
    """Keep composer AND related-work connected components in one split.

    Seeded subset-sum trials find the requested file count, then prefer class
    proportions close to the full catalog. All nine classes must occur in both.
    """
    parents = list(range(len(sources)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen = {}
    for i, source in enumerate(sources):
        for field in ("composer", "split_group"):
            key = (field, source["credit"][field].strip().casefold())
            if key in seen:
                parents[root(i)] = root(seen[key])
            seen[key] = i
    groups = defaultdict(list)
    for i in range(len(sources)):
        groups[root(i)].append(i)
    groups = list(groups.values())
    families = [{t["instrument"] for t in s["piece"]["tracks"] if not t["is_drum"]} for s in sources]
    total = Counter(c for cs in families for c in cs)
    rng = random.Random(f"{seed}:catalog-split")
    best = None
    for _ in range(512):
        rng.shuffle(groups)
        subsets = {0: []}
        for group in groups:
            for count, subset in sorted(list(subsets.items()), reverse=True):
                new = count + len(group)
                if new <= validation_count and new not in subsets:
                    subsets[new] = subset + group
        if validation_count not in subsets:
            continue
        chosen = subsets[validation_count]
        coverage = Counter(c for i in chosen for c in families[i])
        if any(not 0 < coverage[c] < total[c] for c in PROGRAMS):
            continue
        loss = sum((coverage[c] / total[c] - validation_count / len(sources)) ** 2 for c in PROGRAMS)
        if best is None or loss < best[0]:
            best = (loss, set(chosen))
    if best is None:
        raise ValueError("Cannot form composer-disjoint splits with all nine classes at the requested count")
    return ["validation" if i in best[1] else "test" for i in range(len(sources))]


def reduce_validation(clips):
    """Keep ten published examples and append the other 37 to the existing test set."""
    retained = [c for c in clips["validation"] if c["source"]["selection_id"] in PREVIOUS_VALIDATION_SELECTIONS]
    coverage = Counter(k for c in retained for k in c["classes"])
    if (len(clips["validation"]) != 47 or len(retained) != 10
            or any(coverage[k] < 2 for k in PROGRAMS) or not any(c["drums"] for c in retained)):
        raise ValueError("The published validation subset must contain ten clips, all nine classes twice, and drums")
    result = {"validation": [], "test": list(clips["test"])}
    for clip in clips["validation"]:
        split = "validation" if clip["source"]["selection_id"] in PREVIOUS_VALIDATION_SELECTIONS else "test"
        result[split].append({**clip, "id": f"{split}-{len(result[split]) + 1:04d}",
                              "split": split, "previous_public_id": clip["id"]})
    for field in ("composer", "split_group"):
        groups = [{c["source"]["credit"][field].strip().casefold() for c in rows} for rows in result.values()]
        if set.intersection(*groups):
            raise ValueError(f"Reduced validation split overlaps test {field}")
    return result


def diversify_validation(clips):
    """Swap eight clips to broaden styles without changing any rendered performance."""
    result = {split: list(rows) for split, rows in clips.items()}
    if len(result["validation"]) != len(VALIDATION_SELECTIONS):
        raise ValueError("Genre rebalance requires the published ten-clip validation split")
    for i, selection in enumerate(VALIDATION_SELECTIONS):
        outgoing = result["validation"][i]
        if outgoing["source"]["selection_id"] == selection:
            continue
        matches = [j for j, c in enumerate(result["test"]) if c["source"]["selection_id"] == selection]
        if len(matches) != 1:
            raise ValueError(f"Expected one test clip for validation selection {selection}")
        j = matches[0]
        incoming = result["test"][j]
        result["validation"][i] = {**incoming, "id": outgoing["id"], "split": "validation"}
        result["test"][j] = {**outgoing, "id": incoming["id"], "split": "test"}
    coverage = Counter(k for c in result["validation"] for k in c["classes"])
    if any(coverage[k] < 2 for k in PROGRAMS) or not any(c["drums"] for c in result["validation"]):
        raise ValueError("Genre rebalance must preserve all nine classes twice and drums")
    for field in ("composer", "split_group"):
        groups = [{c["source"]["credit"][field].strip().casefold() for c in rows} for rows in result.values()]
        if set.intersection(*groups):
            raise ValueError(f"Genre rebalance overlaps test {field}")
    return result


def plan(seed, catalog, midi_zip, validation_count=10):
    import amt_render
    sources = load_catalog(catalog, midi_zip)
    splits = assign_splits(sources, seed, 47 if validation_count == 10 else validation_count)
    clips = {"validation": [], "test": []}
    for source, split in zip(sources, splits, strict=True):
        rng = random.Random(f"{seed}:{source['sha256']}:excerpt")
        piece, start, crossing, sampling = choose_excerpt(source["piece"], rng)
        index = len(clips[split])
        fonts = ("fluidr3",) if split == "validation" else ("musescore", "generaluser", "timgm6mb")
        classes = sorted({t["instrument"] for t in piece["tracks"] if not t["is_drum"]})
        clips[split].append({"id": f"{split}-{index + 1:04d}", "split": split,
                             "soundfont": fonts[index % len(fonts)], "piece": piece,
                             "mix": amt_render._mix(rng, piece["tracks"]), "classes": classes,
                             "drums": any(t["is_drum"] for t in piece["tracks"]),
                             "source": {k: source[k] for k in ("selection_id", "member", "sha256", "credit", "audit")},
                             "crop_start_s": start, "rearticulated_notes": crossing, "excerpt_sampling": sampling})
    for split in clips:
        if {c for clip in clips[split] for c in clip["classes"]} != set(PROGRAMS):
            raise ValueError(f"Excerpt selection lost instrument coverage in {split}")
    return diversify_validation(reduce_validation(clips)) if validation_count == 10 else clips


def write_metadata(out, clips, catalog, midi_zip, soundfont_hashes, seed, fluidsynth_version, package_versions=None):
    out = Path(out)
    provenance = []
    for split, split_clips in clips.items():
        for clip in split_clips:
            provenance.append({"id": clip["id"], "split": split, **clip["source"],
                               "crop_start_s": clip["crop_start_s"], "rearticulated_notes": clip["rearticulated_notes"],
                               "excerpt_sampling": clip["excerpt_sampling"],
                               "soundfont": clip["soundfont"], "classes": clip["classes"],
                               "drums": clip["drums"], "changes": CHANGES})
            if "previous_public_id" in clip:
                provenance[-1]["previous_public_id"] = clip["previous_public_id"]
    file_records = {str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": digest(p.read_bytes())}
                    for folder in (out / "public", out / "private") for p in sorted(folder.rglob("*")) if p.is_file()}
    source_sets = {split: {c["source"]["sha256"] for c in rows} for split, rows in clips.items()}
    composer_sets = {split: {c["source"]["credit"]["composer"].strip().casefold() for c in rows}
                     for split, rows in clips.items()}
    groups = {split: {c["source"]["credit"]["split_group"].strip().casefold() for c in rows}
              for split, rows in clips.items()}
    audit = {"composer_overlap": len(set.intersection(*composer_sets.values())),
             "group_overlap": len(set.intersection(*groups.values())),
             "input_fingerprint_overlap": len(set.intersection(*source_sets.values()))}
    fingerprints = {split: {digest(json.dumps(labels(c["piece"]), sort_keys=True).encode()) for c in rows}
                    for split, rows in clips.items()}
    audit["label_fingerprint_overlap"] = len(set.intersection(*fingerprints.values()))
    audit["passed"] = not any(audit.values())
    if not audit["passed"]:
        raise ValueError(f"Split contamination: {audit}")
    audit["limitations"] = ("Source compositions and MIDI are public and may occur in model training corpora. "
                            "Composer-disjoint excerpts do not establish contamination freedom. "
                            "The catalog's listening review remains open. Previous procedural baseline results do not apply.")
    moved = [c for c in clips["test"] if "previous_public_id" in c]
    if moved:
        audit["previously_public_test_clips"] = len(moved)
        audit["limitations"] += (f" {len(moved)} test clips were previously public validation examples; moving them does not "
                                  "make them unseen to earlier runs. Their original FluidR3 audio is retained.")
    diversified = [c["source"]["selection_id"] for c in clips["validation"]] == list(VALIDATION_SELECTIONS)
    dataset_version = "niche-catalog-v4" if diversified else "niche-catalog-v3" if moved else "niche-catalog-v2"
    manifest = {"schema_version": 1, "task": "music-transcription", "dataset_version": dataset_version,
                "source": {"catalog": Path(catalog).name, "catalog_sha256": digest(Path(catalog).read_bytes()),
                           "archive": Path(midi_zip).name, "archive_sha256": digest(Path(midi_zip).read_bytes()),
                           "selections": len({p['selection_id'] for p in provenance}), "midi_files": len(provenance),
                           "license": "Per-file; see provenance credit/license/license_url", "provenance": provenance},
                "splits": {"validation": len(clips["validation"]), "heldout": len(clips["test"])},
                "split_audit": audit, "render": {"pipeline": "amt_render.render_audio", "changes": CHANGES,
                    "seed_sha256": digest(seed.encode()), "sample_rate": 44100, "channels": 1, "subtype": "PCM_16",
                    "soundfont_sha256": soundfont_hashes, "fluidsynth_version": fluidsynth_version,
                    "python_packages": package_versions or {
                        name: version(name) for name in ("numpy", "scipy", "soundfile", "pretty-midi", "mido")}},
                "files": file_records}
    if diversified:
        manifest["validation_selection"] = {
            "basis": "Catalog style descriptions and titles; genre labels are not listening-verified",
            "styles": VALIDATION_SELECTIONS,
            "clips_per_instrument": dict(Counter(k for c in clips["validation"] for k in c["classes"])),
            "clips_with_drums": sum(c["drums"] for c in clips["validation"]),
        }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def freeze_inputs(out, clips, catalog, midi_zip):
    out = Path(out)
    sources = out.parent / f"{out.name}-sources"
    sources.mkdir(parents=True, exist_ok=True)
    for path in (Path(catalog), Path(midi_zip)):
        if path.resolve() != (sources / path.name).resolve():
            shutil.copy2(path, sources / path.name)
    (sources / "plan.json").write_text(json.dumps(clips, separators=(",", ":"), ensure_ascii=False))


def write_schema(out):
    (Path(out) / "public/SCHEMA.md").write_text(
        "# Audio transcription examples\n\n"
        "Each validation.jsonl row contains id, audio (relative FLAC path), duration_s, and notes. "
        "Audio is 30-second mono, 44.1 kHz, 16-bit FLAC. Each note contains onset and offset in seconds, "
        "integer MIDI pitch, and instrument: piano, mallets, organ, guitar, bass, strings, brass, reed or pipe. "
        "Percussion is audible in some clips but is not labelled. Multiple parts may share a class. "
        "Simultaneous same-class/pitch unisons count once. You may use these examples however you want, including "
        "for training, fine-tuning, calibration and model selection.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--midi-zip", type=Path, required=True)
    parser.add_argument("--seed-file", type=Path, required=True)
    parser.add_argument("--soundfont-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validation-count", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    import amt_render
    seed = args.seed_file.read_text().strip()
    if not seed:
        raise ValueError("Empty private seed")
    fonts = {name: str(args.soundfont_dir / file) for name, file in FONT_FILES.items()}
    for path in fonts.values():
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    hashes = {name: digest(Path(path).read_bytes()) for name, path in fonts.items()}
    if hashes != amt_render.SOUNDFONT_SHA256:
        raise ValueError("Local soundfont hashes differ from the shared AMT renderer")
    clips = plan(seed, args.catalog, args.midi_zip, args.validation_count)
    # Freeze inputs/plan beside the publishable payload for exact reconstruction.
    freeze_inputs(args.out, clips, args.catalog, args.midi_zip)
    for split, jobs in clips.items():
        rendered = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(lambda clip: amt_render.render_audio(clip, fonts), jobs):
                rendered[result["id"]] = result
                print(f"{split}: {len(rendered)}/{len(jobs)} {result['id']}", flush=True)
        amt_render._write_split(jobs, rendered, args.out, split, public=split == "validation")
    write_schema(args.out)
    version = subprocess.check_output(["fluidsynth", "--version"], text=True).strip()
    manifest = write_metadata(args.out, clips, args.catalog, args.midi_zip, hashes, seed, version)
    print(json.dumps({"splits": manifest["splits"], "split_audit": manifest["split_audit"],
                      "evaluation_sha256": manifest["files"]["private/test.jsonl"]["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
