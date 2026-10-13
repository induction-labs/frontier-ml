# SPDX-License-Identifier: Apache-2.0; see LICENSE.txt.
# Modified: validation-only dependencies and strict JSON responses.
from __future__ import annotations

def ts_to_frame_idx(ts_us: int, fps: int) -> int:
    """Convert a microsecond timestamp to its frame index."""
    return (ts_us * fps) // 1_000_000

KEY_NAME_MAP = {
    "Return": "Return",
    "Escape": "Escape",
    "Backspace": "Backspace",
    "Space": "Space",
    "Tab": "Tab",
    "ShiftLeft": "Shift",
    "ShiftRight": "Shift",
    "ControlLeft": "Ctrl",
    "ControlRight": "Ctrl",
    "Alt": "Alt",
    "AltLeft": "Alt",
    "AltRight": "Alt",
    "AltGr": "AltGr",
    "MetaLeft": "Cmd",
    "MetaRight": "Cmd",
    "UpArrow": "UpArrow",
    "DownArrow": "DownArrow",
    "LeftArrow": "LeftArrow",
    "RightArrow": "RightArrow",
    "CapsLock": "CapsLock",
    "Delete": "Delete",
    "Home": "Home",
    "End": "End",
    "PageUp": "PageUp",
    "PageDown": "PageDown",
}

MODIFIER_KEYS = {"Shift", "Ctrl", "Alt", "AltGr", "Cmd"}

def normalize_key_name(raw: str) -> str:
    """Map raw key names (e.g. 'KeyA', 'MetaLeft') to display names."""
    if raw in KEY_NAME_MAP:
        return KEY_NAME_MAP[raw]
    if raw.startswith("Key") and len(raw) == 4:
        return raw[3].upper()
    if raw.startswith("Digit") and len(raw) == 6:
        return raw[5]
    if raw.startswith("F") and raw[1:].isdigit():
        return raw  # F1, F2, ...
    return raw

def format_key_with_modifiers(key: str, held_modifiers: set[str]) -> str:
    """Combine held modifiers with a key press, e.g. 'Cmd+C'."""
    if key in MODIFIER_KEYS:
        return key
    parts = []
    for mod in ["Cmd", "Ctrl", "Alt", "Shift"]:
        if mod in held_modifiers:
            parts.append(mod)
    parts.append(key)
    return "+".join(parts)

def parse_keylog_events(entries: list, fps: int, num_frames: int) -> list[dict]:
    """Convert raw msgpack keylog entries to action events.

    Emits ONE event per recorded press in the msgpack (KeyPress, MousePress,
    MouseScroll). Within a single frame, duplicate (key, modifiers) pairs are
    deduplicated so OS auto-repeat at ~30 Hz collapses to one KP per key per
    frame at 5 fps.

    Modifier keys (Shift/Cmd/Ctrl/Alt) do not emit on their own; they are
    attached to subsequent non-modifier KeyPress events via the details field.

    Returns list of {"frame_idx": int, "type": str, "details": str}.
    """
    held_modifiers: set[str] = set()

    # Sort by timestamp, then original order to make modifier state deterministic
    sortable = []
    for order_i, entry in enumerate(entries):
        if not isinstance(entry, list) or len(entry) < 2:
            continue
        try:
            ts = int(entry[0])
        except (TypeError, ValueError):
            continue
        ev = entry[1]
        if not isinstance(ev, list) or len(ev) < 1:
            continue
        sortable.append((ts, order_i, ev))
    sortable.sort(key=lambda x: (x[0], x[1]))

    events: list[dict] = []
    kp_seen: set[tuple[int, str]] = set()  # (frame_idx, detail) — dedup within frame
    mc_seen: set[tuple[int, str]] = set()  # (frame_idx, button) — dedup within frame
    scroll_dy_per_frame: dict[int, float] = {}

    for ts, _, ev in sortable:
        frame_idx = ts_to_frame_idx(ts, fps)
        if frame_idx < 0 or frame_idx >= num_frames:
            continue

        etype = str(ev[0])
        payload = ev[1] if len(ev) > 1 else None

        if etype == "ContextChanged":
            # Switching to uncaptured app — clear held modifier state so any
            # later events in the captured app don't carry stale modifiers.
            ctx = ev[1] if len(ev) > 1 else None
            if isinstance(ctx, list) and "UNCAPTURED" in ctx:
                held_modifiers.clear()

        elif etype == "KeyPress":
            raw_key = _parse_key_name(payload)
            if raw_key == "UNKNOWN":
                continue
            key = normalize_key_name(raw_key)
            if key in MODIFIER_KEYS:
                held_modifiers.add(key)
                continue
            detail = format_key_with_modifiers(key, held_modifiers)
            dedup_key = (frame_idx, detail)
            if dedup_key not in kp_seen:
                kp_seen.add(dedup_key)
                events.append(
                    {"frame_idx": frame_idx, "type": "KeyPress", "details": detail}
                )

        elif etype == "KeyRelease":
            raw_key = _parse_key_name(payload)
            if raw_key == "UNKNOWN":
                continue
            key = normalize_key_name(raw_key)
            if key in MODIFIER_KEYS:
                held_modifiers.discard(key)

        elif etype == "MousePress":
            button = _parse_button(payload)
            if button == "UNKNOWN":
                continue
            dedup_key = (frame_idx, button)
            if dedup_key not in mc_seen:
                mc_seen.add(dedup_key)
                events.append(
                    {"frame_idx": frame_idx, "type": "MouseClick", "details": button}
                )

        elif etype == "MouseScroll":
            if isinstance(payload, list) and len(payload) >= 2:
                try:
                    dx, dy = float(payload[0]), float(payload[1])
                except (TypeError, ValueError):
                    dx, dy = 0.0, 0.0
                # Use dy (vertical) if nonzero, else dx (horizontal)
                delta = dy if dy != 0 else dx
                if delta != 0:
                    scroll_dy_per_frame[frame_idx] = (
                        scroll_dy_per_frame.get(frame_idx, 0.0) + delta
                    )

    # Emit scroll events: group consecutive frames into mouses, net dy per mouse
    scroll_frames = sorted(scroll_dy_per_frame.keys())
    if scroll_frames:
        mouse_start = scroll_frames[0]
        mouse_dy = scroll_dy_per_frame[scroll_frames[0]]
        for k in range(1, len(scroll_frames)):
            f_cur = scroll_frames[k]
            if f_cur <= scroll_frames[k - 1] + 1:
                # Consecutive frame — accumulate into mouse
                mouse_dy += scroll_dy_per_frame[f_cur]
            else:
                # Gap — emit previous mouse and start new one
                if mouse_dy != 0:
                    events.append(
                        {
                            "frame_idx": mouse_start,
                            "type": "MouseScroll",
                            "details": "down" if mouse_dy < 0 else "up",
                        }
                    )
                mouse_start = f_cur
                mouse_dy = scroll_dy_per_frame[f_cur]
        # Emit last scroll
        if mouse_dy != 0:
            events.append(
                {
                    "frame_idx": mouse_start,
                    "type": "MouseScroll",
                    "details": "down" if mouse_dy < 0 else "up",
                }
            )

    events.sort(key=lambda x: (x["frame_idx"], x["type"]))
    return events

def _parse_key_name(payload) -> str:
    if isinstance(payload, list) and len(payload) >= 2:
        return str(payload[1])
    return "UNKNOWN"

def _parse_button(payload) -> str:
    if isinstance(payload, list) and len(payload) >= 1:
        b = str(payload[0])
        if b in ("Left", "Right", "Middle"):
            return b
    return "UNKNOWN"
