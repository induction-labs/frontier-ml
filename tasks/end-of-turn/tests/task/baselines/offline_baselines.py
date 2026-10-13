"""Diagnostic training-free baseline emitting causal 20 ms probability streams; not a trained solution.

python tasks/end-of-turn/baselines/offline_baselines.py --data DATA/public/validation.jsonl --out predictions.json
Requires numpy and soundfile. Speech is RMS energy above a slowly rising running noise floor (causal). The
probability is half the trailing-silence ramp on the speaker's own channel (1 after --ramp seconds of silence)
plus, unless --own-channel-only, half a ramp over the partner's speech since this speaker fell silent (1 after
0.3 s). The scorer chooses the threshold, so there is no operating point to select here.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turn_taking import audio_path, load_rows, score, submission

HOP_S = 0.02
MIN_RMS = 0.002  # absolute speech threshold on 24-bit full-scale audio
FLOOR_RATIO = 3.0  # speech is this many times the running noise floor
FLOOR_RISE = 1.002  # per frame: the floor drifts up about 10% per second so it can follow a changing room
PARTNER_S = 0.3  # partner speech after this speaker's offset that counts as a full handover


def speech_frames(path):
    """Causal per-frame speech flags: RMS energy against the quietest recent frame, drifting upward slowly."""
    audio, rate = sf.read(path, dtype="float32")
    step = int(HOP_S * rate)
    rms = np.sqrt(np.mean(audio[:len(audio) // step * step].reshape(-1, step).astype(np.float64) ** 2, axis=1))
    floor, flags = (float(rms[0]) if len(rms) else 0.0), []
    for value in rms:
        floor = min(float(value), floor * FLOOR_RISE)
        flags.append(bool(value > max(MIN_RMS, FLOOR_RATIO * floor)))
    return flags


def predict(own, partner, ramp_s, *, own_channel_only=False):
    """One speaker's stream from its own speech flags and (optionally) the partner's; frame i uses frames <= i only."""
    probabilities, silence_s, partner_s, spoke = [], 0.0, 0.0, False
    for own_active, partner_active in zip(own, partner):  # aligned channels: equal frame counts
        spoke |= own_active
        if own_active:
            silence_s = partner_s = 0.0
        else:
            silence_s += HOP_S
            partner_s += HOP_S if partner_active else 0.0
        silence = min(1.0, silence_s / ramp_s)
        handover = 0.0 if own_channel_only else min(1.0, partner_s / PARTNER_S)
        probabilities.append((silence if own_channel_only else 0.5 * silence + 0.5 * handover) if spoke else 0.0)
    # floor(duration / hop) frames end within one hop of the audio, so a trailing partial frame needs no padding.
    return {"hop_seconds": HOP_S, "eot_probability": probabilities}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ramp", type=float, default=1.0, help="seconds of own silence at which that term saturates")
    parser.add_argument("--own-channel-only", action="store_true", help="pure silence timer: ignore the partner channel")
    args = parser.parse_args()
    rows = load_rows(args.data, require_gold=False)
    predictions = []
    for row in rows:
        flags = {speaker: speech_frames(audio_path(row["audio"][speaker], args.data.parent)) for speaker in ("speaker_1", "speaker_2")}
        predictions.append({"id": row["id"],
                            "speaker_1": predict(flags["speaker_1"], flags["speaker_2"], args.ramp, own_channel_only=args.own_channel_only),
                            "speaker_2": predict(flags["speaker_2"], flags["speaker_1"], args.ramp, own_channel_only=args.own_channel_only)})
    args.out.write_text(json.dumps(submission(predictions)) + "\n")
    if all(row["gold"] is not None for row in rows):
        result = score(rows, predictions)
        result["details"].pop("curve")  # thousands of points; the export keeps the streams themselves
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
