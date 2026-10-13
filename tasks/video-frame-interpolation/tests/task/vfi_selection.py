"""Fixed, score-independent interval sampling shared by acquisition and grading."""
import hashlib
import json

from vfi_protocol import FACTORS

DEFAULT_SELECTION = {"method": "sha256-id-v1", "per_factor": 90, "seed": 0}


def select_intervals(samples, selection=None):
    """Keep complete intervals, balanced by factor; never select using model errors.

    Accept normalized data rows or saved per-interval metric rows. Membership is
    independent of input order; returned rows retain the dataset's original order.
    None explicitly requests the full set, including for public interface checks.
    """
    if selection is None:
        return samples
    if (not isinstance(selection, dict) or set(selection) != {"method", "per_factor", "seed"}
            or selection["method"] != "sha256-id-v1"
            or type(selection["per_factor"]) is not int or selection["per_factor"] <= 0
            or type(selection["seed"]) is not int or selection["seed"] < 0):
        raise ValueError("Invalid fixed interval selection")
    groups = {factor: [] for factor in FACTORS}
    seen = set()
    for sample in samples:
        sid = sample["id"]
        factor = sample["frame_count"] - 1 if "frame_count" in sample else sample["interpolation_factor"]
        if not isinstance(sid, str) or not sid or sid in seen or factor not in groups:
            raise ValueError("Selection requires unique interval IDs and factors 4, 8, 16")
        seen.add(sid)
        key = hashlib.sha256(json.dumps(["vfi-subset-v1", selection["seed"], sid]).encode()).hexdigest()
        groups[factor].append((key, sid))
    chosen = set()
    for rows in groups.values():
        if len(rows) < selection["per_factor"]:
            raise ValueError("Too few intervals for the requested per-factor selection")
        chosen.update(sid for _, sid in sorted(rows)[:selection["per_factor"]])
    return [sample for sample in samples if sample["id"] in chosen]
