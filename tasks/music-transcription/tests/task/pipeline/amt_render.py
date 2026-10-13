"""Operator-only data build for music-transcription 

    # Local catalog builds: see amt_catalog.py. Remote CPU rendering uses the same render_audio function.
    uv run --with mido==1.3.3 modal run tasks/music-transcription/pipeline/amt_render.py::prepare \
        --seed-file data/music-transcription.private-seed --out data/music-transcription \
        --catalog <catalog.xlsx> --midi-zip <files.zip>
    uv run modal run tasks/music-transcription/pipeline/amt_render.py::calibrate
"""
import hashlib
import json
import random
import sys
import struct
from functools import lru_cache
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import amt_compose  # noqa: E402
import amt_catalog  # noqa: E402

SAMPLE_RATE = 44100
GENERALUSER_COMMIT = "684543d5e5efaef08d02be50dcda8d552478fa60"
SOUNDFONTS = {  # container paths; exact asset hashes are checked below
    "fluidr3": "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "musescore": "/sf2/MuseScore_General.sf2",
    "generaluser": "/sf2/GeneralUser-GS.sf2",
    "timgm6mb": "/usr/share/sounds/sf2/TimGM6mb.sf2",
}
SOUNDFONT_SHA256 = {
    "fluidr3": "74594e8f4250680adf590507a306655a299935343583256f3b722c48a1bc1cb0",
    "musescore": "ee51d2c4b1525e70f19a45909c4fd7a2e26d91d115fa89dbf5a6bc413d8b9bf3",
    "generaluser": "9575028c7a1f589f5770fccc8cff2734566af40cd26ed836944e9a5152688cfe",
    "timgm6mb": "c5378b62028c920cb11e4803327983fee2f2cdff5dc89c708e39da417e51c854",
}
SPLITS = {  # split -> (clip count, soundfonts, instrument-count strata)
    "validation": (40, ("fluidr3",), {1: 4, 2: 8, 3: 10, 4: 10, 5: 8}),
    "test": (200, ("musescore", "generaluser", "timgm6mb"), {1: 20, 2: 40, 3: 50, 4: 50, 5: 40}),
}

image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("fluidsynth", "fluid-soundfont-gm", "timgm6mb-soundfont", "curl", "ca-certificates")
         .run_commands(
             "mkdir -p /sf2",
             "curl -sSfL -o /sf2/MuseScore_General.sf2 "
             "https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf2",
             f"curl -sSfL -o /sf2/GeneralUser-GS.sf2 "
             f"https://raw.githubusercontent.com/mrbumpy409/GeneralUser-GS/{GENERALUSER_COMMIT}/GeneralUser-GS.sf2")
         .pip_install("numpy==2.2.6", "scipy==1.16.2", "soundfile==0.13.1", "pretty_midi==0.2.10",
                      "mido==1.3.3", "setuptools==75.8.0")
         .add_local_python_source("amt_compose", "amt_catalog"))
app = modal.App("amt-render", image=image)


def _balanced_classes(rng, count, usage):
    chosen = []
    for _ in range(count):
        options = [c for c in amt_compose.INSTRUMENTS if c not in chosen]
        weights = [1.0 / (1 + usage[c]) ** 2 for c in options]
        pick = rng.choices(options, weights)[0]
        chosen.append(pick)
        usage[pick] += 1
    return chosen


def _mix(rng, tracks):
    parts = []
    for track in tracks:
        eq = []
        if rng.random() < 0.8:
            eq.append(("lowshelf", rng.uniform(80, 300), rng.uniform(-6, 6), 0.7))
        for _ in range(rng.choice((0, 1, 2))):
            eq.append(("peak", rng.uniform(250, 6000), rng.uniform(-7, 7), rng.uniform(0.6, 2.5)))
        if rng.random() < 0.8:
            eq.append(("highshelf", rng.uniform(3000, 10000), rng.uniform(-8, 5), 0.7))
        role = track["role"]
        gain = 0.0 if role in ("melody", "solo") else rng.uniform(-9, -1)
        if track["is_drum"]:
            gain = rng.uniform(-12, -3)
        parts.append({"gain_db": round(gain, 2), "eq": eq,
                      "drive": round(rng.uniform(1.5, 5.0), 2) if rng.random() < 0.15 else 0.0})
    master = {"reverb_rt60": rng.uniform(0.25, 2.4), "reverb_wet": rng.uniform(0.03, 0.35),
              "predelay": rng.uniform(0.004, 0.04), "reverb_tone": rng.uniform(2500, 9000),
              "lowpass": rng.uniform(6000, 18000) if rng.random() < 0.4 else None,
              "highpass": rng.uniform(25, 90) if rng.random() < 0.5 else None,
              "noise_snr_db": rng.uniform(30, 60) if rng.random() < 0.5 else None,
              "rms_db": rng.uniform(-24, -14), "noise_seed": rng.randrange(2 ** 31)}
    return {"parts": parts, "master": master}


def procedural_plan(seed, split):
    """Historical generator, retained for comparison only. Catalog builds do not call it."""
    count, fonts, strata = SPLITS[split]
    rng = random.Random(f"{seed}:{split}:plan")
    sizes = [n for n, k in strata.items() for _ in range(k)]
    assert len(sizes) == count
    rng.shuffle(sizes)
    font_order = [fonts[i % len(fonts)] for i in range(count)]
    rng.shuffle(font_order)
    usage = {c: 0 for c in amt_compose.INSTRUMENTS}
    clips = []
    for index, size in enumerate(sizes):
        clip_rng = random.Random(f"{seed}:{split}:{index}")
        classes = _balanced_classes(clip_rng, size, usage)
        drums = size >= 2 and clip_rng.random() < 0.4
        piece = amt_compose.compose(clip_rng, classes, drums=drums)
        clips.append({"id": f"{split}-{index + 1:04d}", "split": split, "soundfont": font_order[index],
                      "classes": sorted(classes), "drums": drums, "piece": piece, "mix": _mix(clip_rng, piece["tracks"])})
    return clips


def _biquad(kind, freq, gain_db, q, rate):
    """RBJ cookbook biquad as one second-order section."""
    import numpy as np
    a = 10 ** (gain_db / 40)
    w = 2 * np.pi * freq / rate
    cos, alpha = np.cos(w), np.sin(w) / (2 * q)
    if kind == "peak":
        b = [1 + alpha * a, -2 * cos, 1 - alpha * a]
        d = [1 + alpha / a, -2 * cos, 1 - alpha / a]
    elif kind == "lowshelf":
        s = 2 * np.sqrt(a) * alpha
        b = [a * ((a + 1) - (a - 1) * cos + s), 2 * a * ((a - 1) - (a + 1) * cos), a * ((a + 1) - (a - 1) * cos - s)]
        d = [(a + 1) + (a - 1) * cos + s, -2 * ((a - 1) + (a + 1) * cos), (a + 1) + (a - 1) * cos - s]
    else:  # highshelf
        s = 2 * np.sqrt(a) * alpha
        b = [a * ((a + 1) + (a - 1) * cos + s), -2 * a * ((a - 1) + (a + 1) * cos), a * ((a + 1) + (a - 1) * cos - s)]
        d = [(a + 1) - (a - 1) * cos + s, 2 * ((a - 1) - (a + 1) * cos), (a + 1) - (a - 1) * cos - s]
    return np.array([[b[0] / d[0], b[1] / d[0], b[2] / d[0], 1.0, d[1] / d[0], d[2] / d[0]]])


@lru_cache(maxsize=8)
def soundfont_presets(soundfont):
    """Inspect SF2 preset headers so FluidSynth cannot silently substitute a missing patch."""
    raw = Path(soundfont).read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"sfbk":
        raise ValueError(f"Not an SF2 soundfont: {soundfont}")
    offset = 12
    while offset + 8 <= len(raw):
        tag, size = struct.unpack_from("<4sI", raw, offset)
        start = offset + 8
        if tag == b"LIST" and raw[start:start + 4] == b"pdta":
            at = start + 4
            while at + 8 <= start + size:
                key, length = struct.unpack_from("<4sI", raw, at)
                if key == b"phdr":
                    # Last 38-byte record is the EOP sentinel, not a playable preset.
                    return frozenset(struct.unpack_from("<HH", raw, j + 20)
                                     for j in range(at + 8, at + 8 + length - 38, 38))
                at += 8 + length + length % 2
        offset += 8 + size + size % 2
    raise ValueError(f"Missing preset headers: {soundfont}")


def _synth(track, soundfont, workdir):
    """One part through FluidSynth (no built-in reverb or chorus) as float mono at 44.1 kHz."""
    import subprocess
    import numpy as np
    import pretty_midi
    import soundfile
    preset = (track["program"], 128 if track["is_drum"] else 0)
    if preset not in soundfont_presets(soundfont):
        raise ValueError(f"Missing soundfont program/bank {preset} in {soundfont}")
    midi = pretty_midi.PrettyMIDI(resolution=1920, initial_tempo=120.0)
    instrument = pretty_midi.Instrument(program=track["program"], is_drum=track["is_drum"])
    for onset, offset, pitch, velocity in track["notes"]:
        instrument.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=onset, end=offset))
    midi.instruments.append(instrument)
    source, target = workdir / "part.mid", workdir / "part.wav"
    midi.write(str(source))
    subprocess.run(["fluidsynth", "-ni", "-q", "-r", str(SAMPLE_RATE), "-g", "0.5", "-R", "0", "-C", "0",
                    "-O", "float", "-T", "wav", "-F", str(target), soundfont, str(source)],
                   check=True, capture_output=True)
    audio, rate = soundfile.read(str(target), dtype="float64", always_2d=True)
    assert rate == SAMPLE_RATE
    return audio.mean(axis=1)


def render_audio(clip, soundfonts=None):
    """Identical local/remote rendering, accepting explicit local soundfont paths."""
    import tempfile
    import numpy as np
    import soundfile
    from scipy import signal
    soundfonts = SOUNDFONTS if soundfonts is None else soundfonts
    length = int(round(amt_compose.DURATION * SAMPLE_RATE))
    mix = np.zeros(length)
    with tempfile.TemporaryDirectory() as directory:
        workdir = Path(directory)
        for track, settings in zip(clip["piece"]["tracks"], clip["mix"]["parts"], strict=True):
            audio = _synth(track, soundfonts[clip["soundfont"]], workdir)[:length]
            audio = np.pad(audio, (0, length - len(audio)))
            if track["notes"] and np.max(np.abs(audio)) <= 0:
                # The labels would list notes nobody can hear (e.g. a preset missing from the bank).
                raise RuntimeError(f"{clip['id']}: program {track['program']} rendered silence in {clip['soundfont']}")
            if np.max(np.abs(audio)) <= 0:
                continue
            audio = audio / np.sqrt(np.mean(audio ** 2) + 1e-12) * 0.1  # parts start level-matched
            for kind, freq, gain, q in settings["eq"]:
                audio = signal.sosfilt(_biquad(kind, freq, gain, q, SAMPLE_RATE), audio)
            if settings["drive"]:
                audio = np.tanh(settings["drive"] * audio) / settings["drive"]
            mix += audio * 10 ** (settings["gain_db"] / 20)
    master = clip["mix"]["master"]
    noise = np.random.default_rng(master["noise_seed"])
    tail = np.arange(int(master["reverb_rt60"] * SAMPLE_RATE)) / SAMPLE_RATE
    impulse = noise.standard_normal(len(tail)) * np.exp(-6.9078 * tail / master["reverb_rt60"])
    impulse = signal.sosfilt(signal.butter(2, master["reverb_tone"], "low", fs=SAMPLE_RATE, output="sos"), impulse)
    impulse = np.concatenate([np.zeros(int(master["predelay"] * SAMPLE_RATE)), impulse])
    wet = signal.fftconvolve(mix, impulse)[:length]
    wet *= np.sqrt(np.mean(mix ** 2) / (np.mean(wet ** 2) + 1e-12))
    out = (1 - master["reverb_wet"]) * mix + master["reverb_wet"] * wet
    if master["highpass"]:
        out = signal.sosfilt(signal.butter(2, master["highpass"], "high", fs=SAMPLE_RATE, output="sos"), out)
    if master["lowpass"]:
        out = signal.sosfilt(signal.butter(4, master["lowpass"], "low", fs=SAMPLE_RATE, output="sos"), out)
    out *= 10 ** (master["rms_db"] / 20) / np.sqrt(np.mean(out ** 2) + 1e-12)
    if master["noise_snr_db"]:
        pink = signal.lfilter([0.049922035, -0.095993537, 0.050612699, -0.004408786],
                              [1, -2.494956002, 2.017265875, -0.522189400], noise.standard_normal(length))
        pink *= np.sqrt(np.mean(out ** 2)) / np.sqrt(np.mean(pink ** 2)) * 10 ** (-master["noise_snr_db"] / 20)
        out += pink
    out = np.where(np.abs(out) > 0.9, np.sign(out) * (0.9 + 0.1 * np.tanh((np.abs(out) - 0.9) / 0.1)), out)
    buffer = __import__("io").BytesIO()
    soundfile.write(buffer, np.clip(out, -1, 1).astype(np.float32), SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    buffer.seek(0)
    stored, _ = soundfile.read(buffer, dtype="float64")
    return {"id": clip["id"], "flac": buffer.getvalue(), "peak": float(np.max(np.abs(out))),
            "rms_db": float(20 * np.log10(np.sqrt(np.mean(stored ** 2)) + 1e-12))}


@app.function(cpu=2, memory=4096, timeout=1200)
def render_clip(clip):
    return render_audio(clip)


@app.function(timeout=600)
def soundfont_digests():
    hashes = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in SOUNDFONTS.items()}
    if hashes != SOUNDFONT_SHA256:
        raise ValueError("Renderer soundfont hashes differ from the frozen build")
    return hashes


@app.function(timeout=600)
def renderer_environment():
    import subprocess
    from importlib.metadata import version
    return {"fluidsynth": subprocess.check_output(["fluidsynth", "--version"], text=True).strip(),
            "packages": {name: version(name) for name in ("numpy", "scipy", "soundfile", "pretty-midi", "mido")}}


@app.function(cpu=2, memory=4096, timeout=1200)
def onset_lag(soundfont, program):
    """Median lag (s) from MIDI onset to the first sample within 30 dB of the note's peak, over isolated notes."""
    import tempfile
    import numpy as np
    times = [0.5 + 1.5 * i for i in range(8)]
    pitches = [48, 55, 60, 64, 67, 72, 76, 79]
    low, high = amt_compose.RANGES.get(program, (0, 127))
    notes = [[t, t + 0.8, min(max(p, low), high), 100] for t, p in zip(times, pitches)]
    with tempfile.TemporaryDirectory() as directory:
        audio = _synth({"program": program, "is_drum": False, "notes": notes}, SOUNDFONTS[soundfont], Path(directory))
    lags = []
    for t in times:
        start = int(t * SAMPLE_RATE)
        window = np.abs(audio[start - 2000:start + int(1.0 * SAMPLE_RATE)])
        if window.max() <= 0:
            continue
        first = int(np.argmax(window >= window.max() * 10 ** (-30 / 20)))
        lags.append((first - 2000) / SAMPLE_RATE)
    return {"soundfont": soundfont, "program": program, "median_lag": float(np.median(lags)) if lags else None}


@app.local_entrypoint()
def calibrate():
    jobs = [(font, program) for font in SOUNDFONTS for programs in amt_catalog.PROGRAMS.values() for program in programs]
    results = list(onset_lag.starmap(jobs))
    print(json.dumps(soundfont_digests.remote(), indent=1))
    for result in results:
        print(json.dumps(result))


def _write_split(clips, rendered, out, split, public):
    folder = out / ("public" if public else "private")
    (folder / "audio").mkdir(parents=True, exist_ok=True)
    rows = []
    for clip in clips:
        result = rendered[clip["id"]]
        name = f"audio/{clip['id']}.flac"
        (folder / name).write_bytes(result["flac"])
        row = {"id": clip["id"], "audio": name, "duration_s": amt_compose.DURATION,
               "notes": amt_catalog.labels(clip["piece"])}
        if not public:  # trusted-only diagnostics; the verifier never sends them
            row["meta"] = {"soundfont": clip["soundfont"], "classes": clip["classes"], "drums": clip["drums"],
                           "tracks": [{k: v for k, v in t.items() if k != "notes"} for t in clip["piece"]["tracks"]],
                           "mix": clip["mix"], "rms_db": round(result["rms_db"], 2)}
            if "source" in clip:
                row["meta"].update(source=clip["source"], crop_start_s=clip["crop_start_s"],
                                    rearticulated_notes=clip["rearticulated_notes"], excerpt_sampling=clip["excerpt_sampling"])
        rows.append(row)
    target = folder / ("validation.jsonl" if public else "test.jsonl")
    target.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
    return rows


@app.local_entrypoint()
def prepare(seed_file: str, out: str, catalog: str, midi_zip: str, validation_count: int = 10):
    seed = Path(seed_file).read_text().strip()
    out = Path(out)
    digests = soundfont_digests.remote()
    summary = {"soundfont_sha256": digests}
    planned = amt_catalog.plan(seed, catalog, midi_zip, validation_count)
    amt_catalog.freeze_inputs(out, planned, catalog, midi_zip)
    for split in ("validation", "test"):
        clips = planned[split]
        rendered = {r["id"]: r for r in render_clip.map(clips)}
        rows = _write_split(clips, rendered, out, split, public=split == "validation")
        notes = [n for r in rows for n in r["notes"]]
        summary[split] = {"clips": len(rows), "notes": len(notes),
                          "per_instrument": {c: sum(n["instrument"] == c for n in notes) for c in amt_compose.INSTRUMENTS},
                          "clips_per_instrument": {c: sum(c in {n["instrument"] for n in r["notes"]} for r in rows)
                                                   for c in amt_compose.INSTRUMENTS}}
    # Beside the dataset folder, not in it: the upload publishes only public/, private/, artifacts/ and manifest.json.
    (out.parent / f"{out.name}.build-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    amt_catalog.write_schema(out)
    environment = renderer_environment.remote()
    amt_catalog.write_metadata(out, planned, catalog, midi_zip, digests, seed,
                               environment["fluidsynth"], environment["packages"])
    print(json.dumps(summary, indent=1))
