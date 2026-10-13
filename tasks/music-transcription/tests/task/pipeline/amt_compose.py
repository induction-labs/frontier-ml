"""Operator-only procedural composer for music-transcription.

Generates audio off of sampled instruments
"""
import math

DURATION = 30.0
INSTRUMENTS = ("piano", "mallets", "organ", "guitar", "bass", "strings", "brass", "reed", "pipe")
# General MIDI programs rendered for each class (0-based program numbers).
PROGRAMS = {
    "piano": (0, 1, 2, 4, 5),
    "mallets": (8, 9, 10, 11, 12, 13),
    "organ": (16, 17, 18, 19, 20, 21),
    "guitar": (24, 25, 26, 27, 28, 29, 30),
    "bass": (32, 33, 34, 35, 38),
    "strings": (40, 41, 42, 48),  # 49 (slow string ensemble) attacks up to 34 ms late
    "brass": (56, 57, 58, 60, 61),
    "reed": (64, 65, 66, 67, 68, 69, 70, 71),
    "pipe": (72, 73, 74, 75, 79),
}
# Playable MIDI pitch range per program (inclusive).
RANGES = {
    0: (28, 100), 1: (28, 100), 2: (28, 100), 4: (28, 96), 5: (28, 96),
    8: (60, 101), 9: (79, 105), 10: (60, 96), 11: (53, 89), 12: (45, 96), 13: (65, 101),
    16: (36, 96), 17: (36, 96), 18: (36, 96), 19: (36, 96), 20: (36, 93), 21: (41, 89),
    24: (40, 81), 25: (40, 81), 26: (40, 81), 27: (40, 81), 28: (40, 76), 29: (40, 81), 30: (40, 81),
    32: (28, 55), 33: (28, 60), 34: (28, 60), 35: (28, 60), 38: (28, 60),
    40: (55, 96), 41: (48, 84), 42: (36, 72), 48: (36, 91), 49: (36, 91),
    56: (55, 82), 57: (40, 72), 58: (29, 58), 60: (41, 77), 61: (43, 79),
    64: (56, 87), 65: (49, 80), 66: (44, 75), 67: (36, 68), 68: (58, 88), 69: (52, 81), 70: (34, 72), 71: (50, 89),
    72: (74, 102), 73: (60, 96), 74: (60, 86), 75: (60, 91), 79: (60, 86),
}
MIN_DURATION = 0.03

MODES = {"major": (0, 2, 4, 5, 7, 9, 11), "minor": (0, 2, 3, 5, 7, 8, 10),
         "dorian": (0, 2, 3, 5, 7, 9, 10), "mixolydian": (0, 2, 4, 5, 7, 9, 10)}
PROGRESSIONS = {
    "major": ((1, 5, 6, 4), (1, 6, 4, 5), (1, 4, 5, 1), (6, 4, 1, 5), (1, 4, 6, 5), (2, 5, 1, 6), (1, 3, 6, 4),
              (4, 1, 5, 6), (1, 5, 4, 5), (1, 2, 4, 5), (1, 6, 2, 5), (4, 5, 3, 6)),
    "minor": ((1, 6, 3, 7), (1, 4, 5, 1), (1, 7, 6, 7), (1, 6, 4, 5), (1, 4, 7, 3), (6, 7, 1, 1), (1, 3, 4, 5)),
    "dorian": ((1, 4, 1, 4), (1, 2, 1, 7), (1, 4, 7, 1)),
    "mixolydian": ((1, 7, 4, 1), (1, 4, 7, 4), (1, 5, 7, 4)),
}
MELODY_CELLS = {
    4: ((1, 1, 1, 1), (2, 1, 1), (1, 1, 2), (0.5, 0.5, 1, 1, 1), (1.5, 0.5, 2), (1, 0.5, 0.5, 2),
        (0.5,) * 8, (3, 1), (1, 1, 0.5, 0.5, 1), (0.75, 0.25, 1, 1, 1), (0.25, 0.25, 0.25, 0.25, 1, 1, 1), (2, 2),
        (1, 2, 1), (0.5, 1, 0.5, 2), (0.5, 0.5, 0.5, 0.5, 2), (1.5, 1.5, 1)),
    3: ((1, 1, 1), (2, 1), (1, 2), (0.5, 0.5, 1, 1), (3,), (1.5, 0.5, 1), (0.5,) * 6, (1, 0.5, 0.5, 1),
        (0.25, 0.25, 0.5, 1, 1)),
}
COMP_RHYTHMS = {  # (start beat, length in beats) within a bar
    4: (((0, 4),), ((0, 2), (2, 2)), ((0, 1), (1, 1), (2, 1), (3, 1)), ((0.5, 0.5), (1.5, 0.5), (2.5, 0.5), (3.5, 0.5)),
        ((0, 1.5), (1.5, 2.5)), ((0, 1), (1.5, 1), (2.5, 1.5)), ((0, 0.5), (0.5, 0.5), (1.5, 1), (2.5, 0.5), (3, 1))),
    3: (((0, 3),), ((0, 1), (1, 1), (2, 1)), ((1, 1), (2, 1)), ((0, 2), (2, 1)), ((0, 1.5), (1.5, 1.5))),
}
DRUMS = {"kick": 36, "snare": 38, "hat": 42, "open_hat": 46, "ride": 51, "crash": 49, "tom": 45}


def clamp(value, low, high):
    return max(low, min(high, value))


class Piece:
    """Harmony and time grid shared by every part."""

    def __init__(self, rng):
        self.rng = rng
        self.meter = 4 if rng.random() < 0.75 else 3
        self.tempo = rng.uniform(62, 168)
        self.beat = 60.0 / self.tempo
        self.swing = 0.5 if self.tempo > 140 or rng.random() < 0.7 else rng.uniform(0.58, 0.67)
        self.key = rng.randrange(12)
        self.mode = rng.choices(("major", "minor", "dorian", "mixolydian"), (0.5, 0.3, 0.1, 0.1))[0]
        self.scale = MODES[self.mode]
        self.lead_in = rng.choice((0.0, rng.uniform(0.05, 1.2)))
        bar = self.meter * self.beat
        self.bars = int(math.ceil((DURATION - self.lead_in) / bar)) + 1
        chords_per_bar = 2 if (self.tempo < 90 and rng.random() < 0.4) else 1
        progression = list(rng.choice(PROGRESSIONS[self.mode]))
        if rng.random() < 0.5:
            progression += list(rng.choice(PROGRESSIONS[self.mode]))
        self.sevenths = rng.random() < 0.3
        self.harmony = []  # (start beat, length in beats, degree)
        span = self.meter / chords_per_bar
        for index in range(self.bars * chords_per_bar):
            self.harmony.append((index * span, span, progression[index % len(progression)]))

    def time(self, beats):
        """Seconds for a beat position, with swung off-beat eighths."""
        whole = math.floor(beats)
        fraction = beats - whole
        if self.swing != 0.5 and abs(fraction - 0.5) < 1e-9:
            fraction = self.swing
        return self.lead_in + (whole + fraction) * self.beat

    def degree_pitch_class(self, degree, step):
        """Pitch class of the scale tone `step` scale steps above chord root `degree` (1-based)."""
        index = degree - 1 + step
        pitch = self.key + self.scale[index % 7] + 12 * (index // 7)
        if self.mode == "minor" and degree == 5 and step % 7 == 2:  # dominant chord gets the leading tone
            pitch += 1
        return pitch % 12

    def chord(self, degree):
        steps = (0, 2, 4, 6) if self.sevenths else (0, 2, 4)
        return [self.degree_pitch_class(degree, s) for s in steps]

    def chord_at(self, beat):
        for start, length, degree in self.harmony:
            if start <= beat + 1e-9 < start + length:
                return degree
        return self.harmony[-1][2]

    def scale_pitches(self, low, high):
        return [p for p in range(low, high + 1) if (p - self.key) % 12 in self.scale]


def pitches_of(pitch_classes, low, high):
    return [p for p in range(low, high + 1) if p % 12 in pitch_classes]


def voicing(pitch_classes, low, high, size, previous, rng):
    """A close voicing of the chord in [low, high], led smoothly from the previous voicing."""
    candidates = pitches_of(pitch_classes, low, high)
    if len(candidates) < size:
        return candidates
    options = []
    for start in range(len(candidates) - size + 1):
        chord = candidates[start:start + size]
        if chord[-1] - chord[0] > 19:
            continue
        cost = (sum(abs(a - b) for a, b in zip(chord, previous)) if previous and len(previous) == size
                else abs(sum(chord) / size - (low + high) / 2))
        options.append((cost + rng.random() * 2, chord))
    if not options:
        return candidates[:size]
    return min(options)[1]


class Part:
    def __init__(self, piece, rng, instrument, program, role):
        self.piece, self.rng = piece, rng
        self.instrument, self.program, self.role = instrument, program, role
        self.low, self.high = RANGES[program]
        self.notes = []
        self.level = rng.uniform(62, 100)
        self.jitter = rng.uniform(0.003, 0.012)
        self.legato = rng.uniform(0.75, 0.98) if rng.random() < 0.7 else rng.uniform(0.35, 0.6)

    def add(self, start_beat, length_beats, pitch, accent=0.0, legato=None, offset_seconds=0.0):
        piece, rng = self.piece, self.rng
        if not self.low <= pitch <= self.high:
            return
        onset = piece.time(start_beat) + offset_seconds + clamp(rng.gauss(0, self.jitter), -0.025, 0.025)
        end = piece.time(start_beat + length_beats)
        length = max(MIN_DURATION * 2, (end - piece.time(start_beat)) * (legato or self.legato))
        velocity = int(clamp(round(self.level + accent + rng.gauss(0, 6)), 32, 127))
        self.notes.append([max(0.0, onset), max(0.0, onset) + length, pitch, velocity])

    def register(self, low_fraction, high_fraction):
        span = self.high - self.low
        return self.low + int(span * low_fraction), self.low + int(math.ceil(span * high_fraction))


def melody(part, low_fraction=0.3, high_fraction=0.85, rest=0.12):
    piece, rng = part.piece, part.rng
    low, high = part.register(low_fraction, high_fraction)
    scale = piece.scale_pitches(low, high)
    if len(scale) < 3:
        scale = piece.scale_pitches(part.low, part.high)
    index = len(scale) // 2
    phrase_rhythm = None
    for bar in range(piece.bars):
        if bar % 2 == 0 or phrase_rhythm is None:
            phrase_rhythm = [rng.choice(MELODY_CELLS[piece.meter]) for _ in range(2)] if (
                phrase_rhythm is None or rng.random() < 0.55) else phrase_rhythm
        cell = phrase_rhythm[bar % 2]
        beat = bar * piece.meter
        for length in cell:
            if rng.random() >= rest:
                chord = piece.chord(piece.chord_at(beat))
                strong = abs(beat - round(beat)) < 1e-9 and round(beat) % 2 == 0
                move = rng.choices((-2, -1, 0, 1, 2, -4, 4, -3, 3), (10, 25, 6, 25, 10, 3, 3, 4, 4))[0]
                index = clamp(index + move, 0, len(scale) - 1)
                if strong and scale[index] % 12 not in chord:
                    nearby = [i for i in range(max(0, index - 2), min(len(scale), index + 3)) if scale[i] % 12 in chord]
                    if nearby:
                        index = min(nearby, key=lambda i: abs(i - index))
                if index in (0, len(scale) - 1) and rng.random() < 0.6:
                    index = len(scale) // 2
                part.add(beat, length, scale[index], accent=8 if strong else 0)
            beat += length


def bass_line(part):
    piece, rng = part.piece, part.rng
    style = rng.choice(("whole", "half", "root_fifth", "walking", "eighths", "syncopated"))
    top = min(part.high, part.low + 19)

    def near(pitch_class, target):
        options = pitches_of({pitch_class}, part.low, top)
        return min(options, key=lambda p: abs(p - target)) if options else None

    previous = part.low + 7
    for start, length, degree in piece.harmony:
        root_pc, fifth_pc = piece.degree_pitch_class(degree, 0), piece.degree_pitch_class(degree, 4)
        root = near(root_pc, previous)
        fifth = near(fifth_pc, root + 7)
        if root is None:
            continue
        beats = int(round(length))
        if style == "whole" or beats < 2:
            part.add(start, length, root, accent=6)
        elif style == "half":
            half = length / 2
            part.add(start, half, root, accent=6)
            part.add(start + half, half, fifth if rng.random() < 0.6 else root)
        elif style == "root_fifth":
            for b in range(beats):
                part.add(start + b, 1, root if b % 2 == 0 else fifth, accent=6 if b == 0 else 0)
        elif style == "walking":
            scale = piece.scale_pitches(part.low, top)
            position = min(range(len(scale)), key=lambda i: abs(scale[i] - root))
            for b in range(beats):
                pitch = root if b == 0 else scale[clamp(position + rng.choice((-1, 1, 2)), 0, len(scale) - 1)]
                if b > 0:
                    position = scale.index(pitch)
                part.add(start + b, 1, pitch, accent=6 if b == 0 else 0, legato=0.9)
        elif style == "eighths":
            for e in range(beats * 2):
                part.add(start + e / 2, 0.5, root if rng.random() < 0.8 else fifth, accent=6 if e == 0 else 0,
                         legato=0.7)
        else:
            pattern = ((0, 1.5, root), (1.5, 0.5, root), (2, 1, fifth), (3, 1, root)) if beats >= 4 else (
                (0, 1.5, root), (1.5, 1.5, fifth))
            for offset, size, pitch in pattern:
                if offset < length:
                    part.add(start + offset, min(size, length - offset), pitch, accent=6 if offset == 0 else 0)
        previous = root


def block_chords(part, low_fraction=0.25, high_fraction=0.7, size=None, strum=False, pad=False):
    piece, rng = part.piece, part.rng
    low, high = part.register(low_fraction, high_fraction)
    size = size or rng.choice((3, 3, 4))
    patterns = COMP_RHYTHMS[piece.meter]
    pattern = patterns[0] if pad else rng.choice(patterns)
    direction = rng.choice((1, -1))
    previous = None
    for bar in range(piece.bars):
        if not pad and rng.random() < 0.15:
            pattern = rng.choice(patterns)
        for offset, length in pattern:
            beat = bar * piece.meter + offset
            chord = voicing(set(piece.chord(piece.chord_at(beat))), low, high, size, previous, rng)
            previous = chord
            order = chord if direction > 0 else chord[::-1]
            spread = rng.uniform(0.008, 0.022) if strum else 0.0
            for position, pitch in enumerate(order):
                part.add(beat, length, pitch, accent=6 if offset == 0 else 0, legato=0.98 if pad else None,
                         offset_seconds=position * spread)
            if strum and rng.random() < 0.5:
                direction = -direction


def arpeggio(part, low_fraction=0.2, high_fraction=0.8):
    piece, rng = part.piece, part.rng
    low, high = part.register(low_fraction, high_fraction)
    step = 0.25 if (piece.tempo < 105 and rng.random() < 0.5) else 0.5
    shape = rng.choice(("up", "down", "updown", "alberti"))
    for bar in range(piece.bars):
        for n in range(int(round(piece.meter / step))):
            beat = bar * piece.meter + n * step
            tones = voicing(set(piece.chord(piece.chord_at(beat))), low, high, 4, None, rng) or [low]
            tones = sorted(tones)
            if shape == "up":
                pitch = tones[n % len(tones)]
            elif shape == "down":
                pitch = tones[::-1][n % len(tones)]
            elif shape == "updown":
                cycle = tones + tones[-2:0:-1]
                pitch = cycle[n % len(cycle)]
            else:
                pitch = tones[(0, 2, 1, 2)[n % 4] % len(tones)]
            part.add(beat, step, pitch, accent=6 if n == 0 else 0, legato=rng.uniform(0.8, 1.6))


def counter_line(part):
    piece, rng = part.piece, part.rng
    low, high = part.register(0.2, 0.7)
    previous = (low + high) // 2
    length = rng.choice((1, 2, piece.meter))
    for bar in range(piece.bars):
        for n in range(int(piece.meter // length)):
            beat = bar * piece.meter + n * length
            if rng.random() < 0.1:
                continue
            tones = pitches_of(set(piece.chord(piece.chord_at(beat))), low, high)
            if not tones:
                continue
            pitch = min(tones, key=lambda p: abs(p - previous) + rng.random() * 3)
            previous = pitch
            part.add(beat, length, pitch, accent=4 if n == 0 else 0)


def solo_keyboard(part):
    """Melody in the upper register plus left-hand accompaniment (solo piano or guitar)."""
    melody(part, 0.55, 0.95, rest=0.08)
    if part.rng.random() < 0.5:
        block_chords(part, 0.15, 0.5)
    else:
        arpeggio(part, 0.05, 0.45)


def drum_kit(piece, rng):
    notes = []
    style = rng.choice(("rock", "eighth_hats", "sixteenth_hats", "half_time", "ride")) if piece.meter == 4 else "waltz"
    if style == "sixteenth_hats" and piece.tempo > 115:
        style = "eighth_hats"
    level = rng.uniform(60, 100)

    def hit(beat, name, accent=0):
        onset = max(0.0, piece.time(beat) + clamp(rng.gauss(0, 0.006), -0.02, 0.02))
        notes.append([onset, onset + 0.1, DRUMS[name], int(clamp(round(level + accent + rng.gauss(0, 8)), 30, 127))])

    for bar in range(piece.bars):
        base = bar * piece.meter
        if bar % 4 == 0 and rng.random() < 0.4:
            hit(base, "crash", 10)
        if style == "waltz":
            hit(base, "kick", 8)
            hit(base + 1, "snare")
            hit(base + 2, "snare")
            for b in range(3):
                hit(base + b, "hat", -10)
            continue
        kicks = (0, 2) if style != "half_time" else (0,)
        snares = (1, 3) if style != "half_time" else (2,)
        for k in kicks:
            hit(base + k, "kick", 8)
        if rng.random() < 0.3:
            hit(base + 2.5, "kick")
        for s in snares:
            hit(base + s, "snare", 6)
        cymbal = "ride" if style == "ride" else "hat"
        step = 0.25 if style == "sixteenth_hats" else 0.5
        for n in range(int(4 / step)):
            hit(base + n * step, cymbal if (n * step) % 1 or rng.random() > 0.1 else "open_hat", -12 if n % 2 else -4)
        if bar % 4 == 3 and rng.random() < 0.5:
            for n in range(4):
                hit(base + 3 + n / 4, "tom" if n % 2 else "snare", -4)
    return notes


MELODIC = ("pipe", "reed", "brass", "strings", "mallets", "guitar", "piano", "organ")
MELODY_WEIGHT = {"pipe": 5, "reed": 5, "brass": 4, "strings": 4, "mallets": 2, "guitar": 2, "piano": 2, "organ": 1}
SUPPORT_ROLES = {
    "piano": ("chords", "arpeggio", "chords"),
    "guitar": ("strum", "arpeggio", "strum"),
    "organ": ("pad", "chords"),
    "mallets": ("arpeggio", "counter"),
    "strings": ("pad", "counter", "pad"),
    "brass": ("counter", "pad"),
    "reed": ("counter",),
    "pipe": ("counter",),
}


def assign_roles(classes, rng):
    roles = {}
    if "bass" in classes:
        roles["bass"] = "bass"
    rest = [c for c in classes if c != "bass"]
    if len(classes) == 1:
        only = classes[0]
        roles[only] = "solo" if only in ("piano", "guitar") else ("bass" if only == "bass" else "melody")
        return roles
    if rest:
        lead = rng.choices(rest, [MELODY_WEIGHT[c] for c in rest])[0]
        roles[lead] = "melody"
        for c in rest:
            if c != lead:
                roles[c] = rng.choice(SUPPORT_ROLES[c])
    return roles


def compose(rng, classes, drums=False):
    """One piece: {"tempo", "meter", "key", "mode", "duration", "tracks": [{instrument, program, is_drum, role,
    notes: [[onset, offset, pitch, velocity], ...]}]} with at most one track per class."""
    if len(set(classes)) != len(classes) or not set(classes) <= set(INSTRUMENTS):
        raise ValueError("classes must be distinct instrument classes")
    piece = Piece(rng)
    tracks = []
    for instrument, role in assign_roles(list(classes), rng).items():
        program = rng.choice(PROGRAMS[instrument])
        part = Part(piece, rng, instrument, program, role)
        if role == "melody":
            part.level = clamp(part.level + 8, 0, 110)
            melody(part)
        elif role == "bass":
            bass_line(part)
        elif role == "chords":
            block_chords(part)
        elif role == "strum":
            block_chords(part, 0.1, 0.75, size=rng.choice((4, 5, 6)), strum=True)
        elif role == "pad":
            part.level = clamp(part.level - 8, 45, 127)
            block_chords(part, 0.2, 0.7, pad=True)
        elif role == "arpeggio":
            arpeggio(part)
        elif role == "counter":
            counter_line(part)
        elif role == "solo":
            solo_keyboard(part)
        tracks.append({"instrument": instrument, "program": program, "is_drum": False, "role": role,
                       "notes": finish(part.notes)})
    if drums:
        tracks.append({"instrument": None, "program": 0, "is_drum": True, "role": "drums",
                       "notes": finish(drum_kit(piece, rng))})
    return {"tempo": round(piece.tempo, 3), "meter": piece.meter, "key": piece.key, "mode": piece.mode,
            "swing": round(piece.swing, 3), "duration": DURATION, "tracks": tracks}


def finish(notes):
    """Clip to the excerpt, resolve same-pitch overlaps (a re-struck pitch ends the previous note) and drop
    notes too short to sound. Returns notes sorted by onset, times rounded to 0.1 ms."""
    kept = []
    by_pitch = {}
    for note in sorted(notes, key=lambda n: (n[0], n[2])):
        by_pitch.setdefault(note[2], []).append(note)
    for pitch, group in by_pitch.items():
        struck = []
        for note in group:  # a re-strike within 60 ms of the last kept onset would sound as one note: drop it
            if not struck or note[0] - struck[-1][0] >= 0.06:
                struck.append(note)
        for index, (onset, offset, _, velocity) in enumerate(struck):
            if index + 1 < len(struck):
                offset = min(offset, struck[index + 1][0] - 0.01)
            offset = min(offset, DURATION)
            if onset >= DURATION or offset - onset < MIN_DURATION:
                continue
            kept.append([round(onset, 4), round(offset, 4), pitch, velocity])
    return sorted(kept)


def labels(piece):
    """Reference notes of a composed piece: every pitched note, in onset order."""
    rows = []
    for track in piece["tracks"]:
        if track["is_drum"]:
            continue
        for onset, offset, pitch, _ in track["notes"]:
            rows.append({"onset": onset, "offset": offset, "pitch": pitch, "instrument": track["instrument"]})
    rows.sort(key=lambda n: (n["onset"], n["instrument"], n["pitch"]))
    return rows
