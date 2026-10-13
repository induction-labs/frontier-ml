# Vendored from p-doom/inverse-dynamics-model a22f421b575ae74c56ed25db02a30ba150a2864b.
# Modification: import path only. Apache-2.0; see LICENSE-apache-2.0.txt.
#!/usr/bin/env python3
"""IDM action parsing, ground truth extraction, and scoring.

The action space is unified: KeyPress, MouseClick, MouseMove, and MouseScroll.
Mouse displacement values are normalized 0-1000 integer deltas.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict

from idm_upstream_keylog import parse_keylog_events, ts_to_frame_idx


NORM_SCALE = 1000  # normalize deltas to this range


def compute_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return precision, recall, f1


def norm1000(value: float, dimension: int) -> int:
    """Normalize a pixel delta to 0-1000 scale and round."""
    return round(value / dimension * NORM_SCALE)


# ---------------------------------------------------------------------------
# GT extraction
# ---------------------------------------------------------------------------


def extract_mouse_gt(
    raw_actions: list,
    start_s: float,
    fps: int,
    n_frames: int,
    video_w: int,
    video_h: int,
) -> list[dict]:
    """Extract per-frame MouseMove and MouseScroll GT from raw clip actions.

    Args:
        raw_actions: raw [timestamp_us, [event_type, payload]] list from clip JSON
        start_s: clip start time in seconds
        fps: frame extraction rate
        n_frames: total number of frames in clip
        video_w: original video width in pixels (for normalization)
        video_h: original video height in pixels (for normalization)
    Returns:
        List of {frame, type, details} dicts, sorted by frame.
    """
    # Accumulate per-frame deltas
    move_dx: dict[int, float] = defaultdict(float)
    move_dy: dict[int, float] = defaultdict(float)
    scroll_dy: dict[int, float] = defaultdict(float)

    start_us = int(start_s * 1_000_000)
    for ts_us, payload in raw_actions:
        event_type = payload[0]
        frame_idx = ts_to_frame_idx(int(ts_us) - start_us, fps)
        if frame_idx < 0 or frame_idx >= n_frames:
            continue

        if event_type == "MouseMove":
            dx, dy = payload[1][0], payload[1][1]
            if not (math.isnan(dx) or math.isnan(dy)):
                move_dx[frame_idx] += dx
                move_dy[frame_idx] += dy
        elif event_type == "MouseScroll":
            dy = payload[1][1]
            dx_scroll = payload[1][0]
            # Prefer vertical, fall back to horizontal
            delta = dy if dy != 0 else dx_scroll
            if not math.isnan(delta):
                # Flip sign: raw dy<0 = scroll down (content up), but standard
                # convention is positive = scroll down, so negate.
                scroll_dy[frame_idx] -= delta

    # Build event list
    events = []

    # MouseMove events
    all_move_frames = set(move_dx.keys()) | set(move_dy.keys())
    for f in sorted(all_move_frames):
        dx = move_dx.get(f, 0)
        dy = move_dy.get(f, 0)
        if dx == 0 and dy == 0:
            continue
        # Normalize to 0-1000 scale
        dx_n = norm1000(dx, video_w)
        dy_n = norm1000(dy, video_h)
        details = f"{dx_n},{dy_n}"
        events.append(
            {"frame": f"F{f:02d}", "type": "MouseMove", "details": details}
        )

    # MouseScroll events (normalize by height)
    for f in sorted(scroll_dy.keys()):
        dz = scroll_dy[f]
        if dz == 0:
            continue
        dz_n = norm1000(dz, video_h)
        details = str(dz_n)
        events.append(
            {"frame": f"F{f:02d}", "type": "MouseScroll", "details": details}
        )

    events.sort(key=lambda x: (x["frame"], x["type"]))
    return events


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_action_response(text: str) -> list[dict]:
    """Parse model response into action events."""
    if not text:
        return []

    # Strip markdown fences
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Handle thinking tags
    if "<think>" in text:
        parts = text.split("</think>")
        text = parts[-1].strip() if len(parts) > 1 else text

    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    json_text = m.group()

    try:
        arr = json.loads(json_text)
    except json.JSONDecodeError:
        return []

    events = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        frame = item.get("frame", "")
        etype = item.get("type", "")
        details = str(item.get("details", ""))

        if etype not in ("MouseMove", "MouseScroll", "KeyPress", "MouseClick"):
            continue
        if not re.match(r"^F\d{2}$", frame):
            continue

        events.append({"frame": frame, "type": etype, "details": details})

    return events


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


# Per-type matching rules for unified IDM action scoring.
#   tol_low / tol_high: bounds on (pred_frame - gt_frame). Asymmetric allowed.
#   match_detail: if True, prediction's detail must equal GT's detail.
INLINE_MATCH_RULES = {
    "KeyPress": {"tol_low": -2, "tol_high": 3, "match_detail": True},
    "MouseClick": {"tol_low": -2, "tol_high": 3, "match_detail": True},
    "MouseScroll": {"tol_low": 0, "tol_high": 0, "match_detail": True},
    "MouseMove": {"tol_low": 0, "tol_high": 0, "match_detail": False},
}


def _match_per_type_rules(gt: list[dict], preds: list[dict], rules: dict) -> dict:
    """Greedy bipartite matching using per-type rules.

    Each rule: tol_low/tol_high on (pred.frame - gt.frame), match_detail flag.
    """
    used_preds = set()
    matches = []
    unmatched_gt = []
    for g in gt:
        rule = rules.get(g["type"])
        if rule is None:
            unmatched_gt.append(g)
            continue
        tol_low, tol_high = rule["tol_low"], rule["tol_high"]
        match_detail = rule["match_detail"]
        gd = g.get("detail", "")
        gf = g["frame"]
        best_idx, best_dist = None, max(abs(tol_low), abs(tol_high)) + 1
        for j, p in enumerate(preds):
            if j in used_preds or p["type"] != g["type"]:
                continue
            if match_detail and p.get("detail", "") != gd:
                continue
            off = p["frame"] - gf
            if not (tol_low <= off <= tol_high):
                continue
            d = abs(off)
            if d < best_dist:
                best_idx, best_dist = j, d
        if best_idx is not None:
            used_preds.add(best_idx)
            matches.append({"gt": g, "pred": preds[best_idx], "frame_dist": best_dist})
        else:
            unmatched_gt.append(g)
    unmatched_preds = [p for j, p in enumerate(preds) if j not in used_preds]
    return {
        "matches": matches,
        "unmatched_gt": unmatched_gt,
        "unmatched_preds": unmatched_preds,
    }



def _frame_int(frame) -> int:
    if isinstance(frame, int):
        return frame
    digits = re.sub(r"[^0-9]", "", str(frame))
    return int(digits) if digits else 0


def _scroll_direction(detail: str) -> str:
    value = str(detail).strip()
    try:
        numeric = float(value)
    except ValueError:
        return value
    if numeric > 0:
        return "down"
    if numeric < 0:
        return "up"
    return ""


def _matching_action(action: dict) -> dict | None:
    action_type = action.get("type", "")
    frame = _frame_int(action.get("frame", action.get("frame_idx", 0)))
    detail = str(action.get("details", action.get("detail", "")))

    if action_type in ("KeyPress", "MouseClick"):
        return {"frame": frame, "type": action_type, "detail": detail}
    if action_type == "MouseScroll":
        direction = _scroll_direction(detail)
        if direction in ("up", "down"):
            return {"frame": frame, "type": action_type, "detail": direction}
        return None
    if action_type == "MouseMove":
        return {"frame": frame, "type": action_type, "detail": ""}
    return None


def _to_matching_actions(actions: list[dict]) -> list[dict]:
    converted = []
    for action in actions:
        item = _matching_action(action)
        if item is not None:
            converted.append(item)
    return converted


def _summarize_matches(gt: list[dict], preds: list[dict], matches: list[dict]) -> dict:
    stats: dict[str, dict] = {}
    for action_type in ("KeyPress", "MouseClick", "MouseScroll", "MouseMove"):
        gt_count = sum(1 for item in gt if item["type"] == action_type)
        pred_count = sum(1 for item in preds if item["type"] == action_type)
        tp = sum(1 for match in matches if match["gt"]["type"] == action_type)
        fp = pred_count - tp
        fn = gt_count - tp
        if gt_count == 0 and pred_count == 0:
            precision = recall = f1 = 1.0
        else:
            precision = tp / (tp + fp) if (tp + fp) else 0
            recall = tp / (tp + fn) if (tp + fn) else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall)
                else 0
            )
        stats[action_type] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    action_types = ("KeyPress", "MouseClick", "MouseScroll", "MouseMove")
    tp = sum(stats[t]["tp"] for t in action_types)
    fp = sum(stats[t]["fp"] for t in action_types)
    fn = sum(stats[t]["fn"] for t in action_types)
    if tp == 0 and fp == 0 and fn == 0:
        precision = recall = f1 = 1.0
    else:
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0
        )
    stats["overall"] = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return stats


def score_action_lists(gt_actions: list[dict], pred_actions: list[dict]) -> dict:
    """Score already-normalized GT/pred action lists across all action types."""
    gt = _to_matching_actions(gt_actions)
    preds = _to_matching_actions(pred_actions)
    result = _match_per_type_rules(gt, preds, INLINE_MATCH_RULES)
    return _summarize_matches(gt, preds, result["matches"])


def score_unified_inline(
    gt_mouse: list[dict],
    gt_key_click: list[dict],
    preds: list[dict],
) -> dict:
    """Score mouse GT plus key/click GT against a unified prediction list."""
    return score_action_lists([*gt_key_click, *gt_mouse], preds)


def _extract_key_click_gt(
    raw_actions: list,
    start_s: float,
    fps: int,
    n_frames: int,
) -> list[dict]:
    """Extract KeyPress/MouseClick GT relative to a clip start time."""
    start_us = int(start_s * 1_000_000)
    rel_actions = []
    for event in raw_actions:
        if not isinstance(event, list) or len(event) < 2:
            continue
        try:
            ts_us = int(event[0]) - start_us
        except Exception:
            continue
        if ts_us < 0:
            continue
        rel_actions.append([ts_us, event[1]])

    return parse_keylog_events(rel_actions, fps, n_frames)


def score_actions(
    raw_actions: list,
    start_s: float,
    fps: int,
    n_frames: int,
    video_w: int,
    video_h: int,
    preds: list[dict],
) -> dict:
    """Score all four IDM action types together."""
    mouse_gt = extract_mouse_gt(raw_actions, start_s, fps, n_frames, video_w, video_h)
    key_click_gt = _extract_key_click_gt(raw_actions, start_s, fps, n_frames)
    stats = score_unified_inline(mouse_gt, key_click_gt, preds)
    stats["mouse_dynamics"] = score_mouse_dynamics(mouse_gt, preds)
    return stats


def _parse_move_details(details: str) -> tuple[float, float] | None:
    """Parse MouseMove details 'dx,dy' into (dx, dy)."""
    parts = details.split(",")
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def _parse_scroll_details(details: str) -> float | None:
    """Parse MouseScroll details into dz."""
    try:
        return float(details)
    except ValueError:
        return None


def _direction_match(gt_val: float, pred_val: float) -> bool:
    """Check if sign (direction) matches. Both zero = match, otherwise signs must agree."""
    if gt_val == 0 and pred_val == 0:
        return True
    if gt_val == 0 or pred_val == 0:
        return False  # one is zero, other isn't — wrong direction
    return (gt_val > 0) == (pred_val > 0)


def score_mouse_dynamics(
    gt: list[dict],
    preds: list[dict],
) -> dict:
    """Score MouseMove/MouseScroll diagnostics against GT.

    This is not the headline IDM eval. Use score_actions() for the unified
    KeyPress/MouseClick/MouseMove/MouseScroll score.
    """
    # Index by (frame, type)
    gt_by_key: dict[tuple[str, str], dict] = {}
    for g in gt:
        key = (g["frame"], g["type"])
        gt_by_key[key] = g

    pred_by_key: dict[tuple[str, str], dict] = {}
    for p in preds:
        key = (p["frame"], p["type"])
        pred_by_key[key] = p

    all_keys = set(gt_by_key.keys()) | set(pred_by_key.keys())

    # Accumulators
    type_stats: dict[str, dict] = {}
    for etype in ("MouseMove", "MouseScroll"):
        type_stats[etype] = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "direction_correct": 0,
            "direction_total": 0,
            "l2_errors": [],  # L2 distance for MouseMove, abs error for MouseScroll
            "exact_matches": 0,
            "close_matches": 0,  # within ±10 normalized units
            "gt_vals": [],  # for R² computation
            "pred_vals": [],  # for R² computation
            "cos_sims": [],  # cos(pred, gt) per TP with both nonzero
            "mag_ratios": [],  # |pred|/|gt| per TP with both nonzero
        }

    for key in all_keys:
        _, etype = key
        has_gt = key in gt_by_key
        has_pred = key in pred_by_key

        if etype not in type_stats:
            continue

        stats = type_stats[etype]

        if has_gt and has_pred:
            stats["tp"] += 1
            g = gt_by_key[key]
            p = pred_by_key[key]

            if etype == "MouseMove":
                gt_vals = _parse_move_details(g["details"])
                pred_vals = _parse_move_details(p["details"])
                if gt_vals and pred_vals:
                    gt_dx, gt_dy = gt_vals
                    pred_dx, pred_dy = pred_vals
                    # Direction
                    dx_dir = _direction_match(gt_dx, pred_dx)
                    dy_dir = _direction_match(gt_dy, pred_dy)
                    stats["direction_total"] += 2
                    stats["direction_correct"] += int(dx_dir) + int(dy_dir)
                    # L2 distance between vectors
                    l2 = math.sqrt((pred_dx - gt_dx) ** 2 + (pred_dy - gt_dy) ** 2)
                    stats["l2_errors"].append(l2)
                    # Cos similarity & magnitude ratio (only when both nonzero)
                    gt_mag = math.hypot(gt_dx, gt_dy)
                    pred_mag = math.hypot(pred_dx, pred_dy)
                    if gt_mag > 1e-6 and pred_mag > 1e-6:
                        stats["cos_sims"].append(
                            (pred_dx * gt_dx + pred_dy * gt_dy) / (pred_mag * gt_mag)
                        )
                        stats["mag_ratios"].append(pred_mag / gt_mag)
                    # Collect for R²
                    stats["gt_vals"].append((gt_dx, gt_dy))
                    stats["pred_vals"].append((pred_dx, pred_dy))
                    if g["details"] == p["details"]:
                        stats["exact_matches"] += 1
                    if abs(gt_dx - pred_dx) <= 10 and abs(gt_dy - pred_dy) <= 10:
                        stats["close_matches"] += 1

            elif etype == "MouseScroll":
                gt_dz = _parse_scroll_details(g["details"])
                pred_dz = _parse_scroll_details(p["details"])
                if gt_dz is not None and pred_dz is not None:
                    stats["direction_total"] += 1
                    stats["direction_correct"] += int(_direction_match(gt_dz, pred_dz))
                    # Absolute error (1D, so L2 = L1)
                    stats["l2_errors"].append(abs(pred_dz - gt_dz))
                    # Collect for R²
                    stats["gt_vals"].append((gt_dz,))
                    stats["pred_vals"].append((pred_dz,))
                    if g["details"] == p["details"]:
                        stats["exact_matches"] += 1
                    if abs(gt_dz - pred_dz) <= 10:
                        stats["close_matches"] += 1

        elif has_gt and not has_pred:
            stats["fn"] += 1
            # Treat missed predictions as predicting 0,0 (or 0 for scroll)
            g = gt_by_key[key]
            if etype == "MouseMove":
                gt_vals = _parse_move_details(g["details"])
                if gt_vals:
                    gt_dx, gt_dy = gt_vals
                    stats["gt_vals"].append((gt_dx, gt_dy))
                    stats["pred_vals"].append((0, 0))
                    l2 = math.sqrt(gt_dx**2 + gt_dy**2)
                    stats["l2_errors"].append(l2)
                    stats["direction_total"] += 2
                    # 0 only matches 0
                    stats["direction_correct"] += int(gt_dx == 0) + int(gt_dy == 0)
                    if abs(gt_dx) <= 10 and abs(gt_dy) <= 10:
                        stats["close_matches"] += 1
            elif etype == "MouseScroll":
                gt_dz = _parse_scroll_details(g["details"])
                if gt_dz is not None:
                    stats["gt_vals"].append((gt_dz,))
                    stats["pred_vals"].append((0,))
                    stats["l2_errors"].append(abs(gt_dz))
                    stats["direction_total"] += 1
                    stats["direction_correct"] += int(gt_dz == 0)
                    if abs(gt_dz) <= 10:
                        stats["close_matches"] += 1
        elif has_pred and not has_gt:
            stats["fp"] += 1

    # Compute summary metrics
    summary = {}
    total_tp, total_fp, total_fn = 0, 0, 0
    all_l2 = []

    for etype, stats in type_stats.items():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        total_tp += tp
        total_fp += fp
        total_fn += fn

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        dir_acc = (
            stats["direction_correct"] / stats["direction_total"]
            if stats["direction_total"] > 0
            else 0
        )
        l2_errs = stats["l2_errors"]
        mean_l2 = sum(l2_errs) / len(l2_errs) if l2_errs else 0
        all_l2.extend(l2_errs)
        exact_acc = stats["exact_matches"] / tp if tp > 0 else 0
        close_acc = stats["close_matches"] / (tp + fn) if (tp + fn) > 0 else 0

        # R² (coefficient of determination) on matched predictions
        r_squared = 0.0
        gt_v = stats["gt_vals"]
        pred_v = stats["pred_vals"]
        if len(gt_v) > 1:
            # Flatten all components into single lists
            gt_flat = [c for tup in gt_v for c in tup]
            pred_flat = [c for tup in pred_v for c in tup]
            ss_res = sum((g - p_) ** 2 for g, p_ in zip(gt_flat, pred_flat))
            mean_gt = sum(gt_flat) / len(gt_flat)
            ss_tot = sum((g - mean_gt) ** 2 for g in gt_flat)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Cos similarity & magnitude ratio (TPs where both pred and GT are nonzero)
        cos_list = stats["cos_sims"]
        mag_list = stats["mag_ratios"]
        n_cos = len(cos_list)
        if n_cos > 0:
            cos_sorted = sorted(cos_list)
            mag_sorted = sorted(mag_list)
            cos_median = cos_sorted[n_cos // 2]
            mag_median = mag_sorted[n_cos // 2]
            cos_mean = sum(cos_list) / n_cos
            frac_cos_pos = sum(1 for c in cos_list if c > 0) / n_cos
            frac_cos_p9 = sum(1 for c in cos_list if c > 0.9) / n_cos
            frac_within_2x = sum(1 for r in mag_list if 0.5 <= r <= 2.0) / n_cos
            frac_within_1_5x = sum(1 for r in mag_list if 1 / 1.5 <= r <= 1.5) / n_cos
        else:
            cos_median = cos_mean = mag_median = 0.0
            frac_cos_pos = frac_cos_p9 = frac_within_2x = frac_within_1_5x = 0.0

        summary[etype] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "direction_accuracy": round(dir_acc, 4),
            "mean_l2": round(mean_l2, 2),
            "r_squared": round(r_squared, 4),
            "exact_match_accuracy": round(exact_acc, 4),
            "close_match_accuracy": round(close_acc, 4),
            "cos_n": n_cos,
            "cos_median": round(cos_median, 4),
            "cos_mean": round(cos_mean, 4),
            "frac_cos_positive": round(frac_cos_pos, 4),
            "frac_cos_above_0_9": round(frac_cos_p9, 4),
            "mag_ratio_median": round(mag_median, 4),
            "frac_within_2x": round(frac_within_2x, 4),
            "frac_within_1_5x": round(frac_within_1_5x, 4),
            "_cos_sims": cos_list,  # raw values, for cross-clip aggregation
            "_mag_ratios": mag_list,
        }

    # Overall
    op = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    or_ = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    of1 = 2 * op * or_ / (op + or_) if (op + or_) > 0 else 0
    overall_l2 = sum(all_l2) / len(all_l2) if all_l2 else 0

    summary["overall"] = {
        "precision": round(op, 4),
        "recall": round(or_, 4),
        "f1": round(of1, 4),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "mean_l2": round(overall_l2, 2),
    }

    return summary
