"""Note lists and the multi-instrument transcription metric (standard library only; shipped with validate.py).
"""
import math

INSTRUMENTS = ("piano", "mallets", "organ", "guitar", "bass", "strings", "brass", "reed", "pipe")
ONSET_TOLERANCE = 0.05
OFFSET_RATIO = 0.2
OFFSET_MIN_TOLERANCE = 0.05
MAX_NOTES = 10000  # per clip


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def parse_notes(response):
    """The validated note list of a /transcribe response; raises ValueError on anything malformed."""
    if not isinstance(response, dict) or not isinstance(response.get("notes"), list):
        raise ValueError('Response must be {"notes": [...]}')
    notes = response["notes"]
    if len(notes) > MAX_NOTES:
        raise ValueError(f"At most {MAX_NOTES} notes per clip")
    parsed = []
    for index, note in enumerate(notes):
        if not isinstance(note, dict):
            raise ValueError(f"notes[{index}] must be an object")
        onset = _number(note.get("onset"), f"notes[{index}].onset")
        offset = _number(note.get("offset"), f"notes[{index}].offset")
        pitch = note.get("pitch")
        instrument = note.get("instrument")
        if onset < 0 or offset <= onset:
            raise ValueError(f"notes[{index}] needs 0 <= onset < offset")
        if isinstance(pitch, bool) or not isinstance(pitch, int) or not 0 <= pitch <= 127:
            raise ValueError(f"notes[{index}].pitch must be an integer MIDI pitch 0-127")
        if instrument not in INSTRUMENTS:
            raise ValueError(f"notes[{index}].instrument must be one of {', '.join(INSTRUMENTS)}")
        parsed.append({"onset": onset, "offset": offset, "pitch": pitch, "instrument": instrument})
    return parsed


def _rounded_distance(a, b):
    """|a - b| rounded to 4 decimals like numpy.around (round half to even of the scaled value)."""
    return round(abs(a - b) * 10000) / 10000


def _max_matching(adjacency, right_count):
    """Size of a maximum bipartite matching (Kuhn's augmenting paths, iterative depth-first search)."""
    owner = [-1] * right_count  # right vertex -> matched left vertex
    partner = [-1] * len(adjacency)  # left vertex -> matched right vertex
    matched = 0
    for start in range(len(adjacency)):
        parent = {}  # right vertex -> the left vertex it was reached from
        visited = {start}
        stack = [(start, iter(adjacency[start]))]
        found = None
        while stack and found is None:
            left, edges = stack[-1]
            for right in edges:
                if right in parent:
                    continue
                parent[right] = left
                if owner[right] == -1:
                    found = right
                    break
                if owner[right] not in visited:
                    visited.add(owner[right])
                    stack.append((owner[right], iter(adjacency[owner[right]])))
                    break
            else:
                stack.pop()
        right = found
        while right is not None and right != -1:  # flip the augmenting path
            left = parent[right]
            previous = partner[left]
            owner[right], partner[left] = left, right
            right = previous
        matched += found is not None
    return matched


def match_count(reference, estimate, *, with_offsets=False, by_instrument=True):
    """Maximum number of one-to-one matches between reference and estimated notes (same pitch, same class unless
    by_instrument is False, onsets within 50 ms; with_offsets also requires mir_eval's offset criterion)."""
    groups = {}
    for side, notes in ((0, reference), (1, estimate)):
        for note in notes:
            key = (note["instrument"] if by_instrument else None, note["pitch"])
            groups.setdefault(key, ([], []))[side].append(note)
    total = 0
    for refs, ests in groups.values():
        if not refs or not ests:
            continue
        ests = sorted(ests, key=lambda n: n["onset"])
        onsets = [n["onset"] for n in ests]
        adjacency = []
        for ref in refs:
            low = _bisect(onsets, ref["onset"] - ONSET_TOLERANCE - 0.001)
            edges = []
            for j in range(low, len(ests)):
                if onsets[j] > ref["onset"] + ONSET_TOLERANCE + 0.001:
                    break
                if _rounded_distance(ref["onset"], onsets[j]) > ONSET_TOLERANCE:
                    continue
                if with_offsets:
                    tolerance = max(OFFSET_RATIO * (ref["offset"] - ref["onset"]), OFFSET_MIN_TOLERANCE)
                    if _rounded_distance(ref["offset"], ests[j]["offset"]) > tolerance:
                        continue
                edges.append(j)
            adjacency.append(edges)
        total += _max_matching(adjacency, len(ests))
    return total


def _bisect(values, target):
    low, high = 0, len(values)
    while low < high:
        middle = (low + high) // 2
        if values[middle] < target:
            low = middle + 1
        else:
            high = middle
    return low


def f1(matches, predicted, reference):
    return 2 * matches / (predicted + reference) if matches else 0.0


def clip_counts(reference, estimate):
    """Per-class (matches, predicted, reference) for one clip, plus class-agnostic and onset-offset diagnostics."""
    counts = {}
    for instrument in INSTRUMENTS:
        refs = [n for n in reference if n["instrument"] == instrument]
        ests = [n for n in estimate if n["instrument"] == instrument]
        counts[instrument] = [match_count(refs, ests) if refs and ests else 0, len(ests), len(refs)]
    return {"classes": counts,
            "agnostic": [match_count(reference, estimate, by_instrument=False), len(estimate), len(reference)],
            "with_offsets": [match_count(reference, estimate, with_offsets=True), len(estimate), len(reference)]}


def aggregate(records):
    """Pool per-clip counts: the headline macro F1 over classes present in the reference, and diagnostics."""
    pooled = {c: [0, 0, 0] for c in INSTRUMENTS}
    agnostic, offsets = [0, 0, 0], [0, 0, 0]
    for record in records:
        for instrument, values in record["classes"].items():
            pooled[instrument] = [a + b for a, b in zip(pooled[instrument], values)]
        agnostic = [a + b for a, b in zip(agnostic, record["agnostic"])]
        offsets = [a + b for a, b in zip(offsets, record["with_offsets"])]
    present = [c for c in INSTRUMENTS if pooled[c][2] > 0]
    if not present:
        raise ValueError("No reference notes to score")
    per_class = {c: {"f1": f1(*pooled[c]), "precision": pooled[c][0] / pooled[c][1] if pooled[c][1] else 0.0,
                     "recall": pooled[c][0] / pooled[c][2] if pooled[c][2] else 0.0,
                     "matches": pooled[c][0], "predicted": pooled[c][1], "reference": pooled[c][2]} for c in INSTRUMENTS}
    return {"score": sum(per_class[c]["f1"] for c in present) / len(present),
            "classes_scored": present, "per_instrument": per_class,
            # Diagnostics only: class-agnostic onset F1 and class-aware onset+offset F1, pooled over all notes.
            "instrument_agnostic_onset_f1": f1(*agnostic),
            "onset_offset_f1": f1(*offsets)}
