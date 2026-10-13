"""TurnBench consensus and floor construction, vendored without algorithm changes.

Upstream: https://github.com/SesameAILabs/turnbench/blob/76ccd045f121ccfa921abac2ad3107027e736911/turnbench/gold.py
Revision: 76ccd045f121ccfa921abac2ad3107027e736911

Only the data-container dependency and CLI were removed for local scoring.

MIT License

Copyright (c) 2026 Freeman Jiang, Ramon Sanabria, Soham Deshmukh, Bandhav Veluri, Simon Michael Vuch Williams, Elliott K. Suen, Garreth Lee, Kevin Yoonho Choi, Takuya Umeki, Riku Kubo, Sathvik Udupa, Chien-yu Huang, Shih-Yun Shan Kuan, Zhuoyan Tao, Satyapriya Krishna, Sefik Emre Eskimez, Yu Tsao, Hung-yi Lee, Shinji Watanabe

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import bisect
from dataclasses import dataclass
from itertools import combinations
from statistics import median

UPSTREAM_REVISION = "76ccd045f121ccfa921abac2ad3107027e736911"
ANNOTATORS = ("a", "b", "c")
SPEAKERS = (1, 2)
Annotation = tuple[float, float, str, str]


@dataclass(frozen=True)
class Conversation:
    """Raw annotations for one two-speaker conversation, in source order."""

    conversation_id: str
    duration_s: float
    annotations: dict[tuple[int, str], list[Annotation]]


# Maps the fine-grained annotator labels to the canonical event taxonomy used
# for evaluation (canonical -> fine labels that collapse into it). Labels not
# listed are ignored when building consensus. Laughter and Awkward Silence are
# their own buckets: tt-benchmark's turn-region builder treats them specially
# (leading laughter doesn't claim the floor; awkward silence is a break, not a
# turn). EOT derivation here currently uses `Turn` only — the Turn ∪ Laughter
# floor rule is deferred (see turnbench/README.md).
LABEL_MAP: dict[str, tuple[str, ...]] = {
    "Turn": (
        "Normal Turn",
        "Regular Turn",  # defensive alias; occurs in neither released split
        "Strong Floor Hold",
        "Bounded Response",
        "Filler",
        "Overlap",
    ),
    "Laughter": ("Laughter",),
    "AwkwardSilence": ("Awkward Silence",),
    # An interruption means the other speaker was actually interrupted: the
    # floor changes hands. Only the floor-taking fine labels are scored as
    # Interruption; non-floor-taking attempts are their own class, which the
    # INT task excludes rather than scores (high-precision gold — at onset
    # the two are indistinguishable, so firing on an attempt is neither
    # rewarded nor penalised). This deliberately narrows tt-benchmark's
    # original any-vocalization definition.
    "Interruption": (
        "Floor-taking Competitive Interruption",
        "Floor-taking Cooperative Interruption",
    ),
    "NonFloorTakingInterruption": (
        "Non-floor Taking Competitive Interruption",
        "Non-floor Taking Cooperative Interruption",
    ),
    "Backchannel": (
        "Acknowledgement Backchannel",
        "Continuer Backchannel",
        "Reaction Backchannel",
    ),
    "NonContent": (
        "Non-Speech Noise",
        "Channel Bleed",
        "Speech, Non-Linguistic",
    ),
}
CANONICAL = {
    fine: canonical for canonical, fines in LABEL_MAP.items() for fine in fines
}

# The taxonomy is hierarchical for floor purposes: "Turn" is the coarser
# level, and floor-taking interruptions are turns at that level. The *turn
# view* asks the coarser question — "does this span CLAIM the floor?" — so a
# span labelled Normal Turn / Floor-taking Interruption / Overlap reaches a
# floor-claiming majority there even though the label view sees disagreement. The label view
# (CANONICAL) still serves the INT task, where that distinction is the point.
TURN_LABELS = LABEL_MAP["Turn"] + LABEL_MAP["Interruption"]
TURN_CANONICAL = {fine: "Turn" for fine in TURN_LABELS}

# Other-speaker sounds that are NOT a floor grab -> Interruption-task negatives.
INT_NEGATIVE_LABELS = ("Backchannel", "NonContent")
# Both interruption classes mark a barge-in when truncating pauses.
INTERRUPTION_LABELS = ("Interruption", "NonFloorTakingInterruption")

MIN_AGREEMENT = 2  # of 3 annotators an event needs (3 = unanimous, 2 = majority)
TIME_TOLERANCE_S = 0.2  # max disagreement on start OR end across annotators
OVERLAP_WINDOW_S = 0.5  # max start-time gap to match events across annotators

# The scoring windows around gold anchors (used by turnbench/score.py; carried in
# the exported artifact so external scorers don't duplicate them). TAU_PRE_S
# is a matching tolerance — the gold boundary is only annotation-exact — and
# TAU_MAX_S the latency deadline.
TAU_PRE_S = 0.25
TAU_MAX_S = 3.00

@dataclass(frozen=True)
class ConsensusEvent:
    """One majority-agreed annotated segment."""

    speaker: int
    start: float
    end: float
    label: str


@dataclass(frozen=True)
class AnchorEvent:
    """A scoring anchor: a window centred at `time_s` on `speaker`'s channel."""

    speaker: int
    time_s: float


@dataclass(frozen=True)
class Interval:
    speaker: int
    start: float
    end: float


@dataclass(frozen=True)
class ConsensusViews:
    """The two consensus granularities."""

    turn_events: list[ConsensusEvent]  # majority floor-claiming spans (label "Turn")
    turn_excluded: list[Interval]  # no floor-level agreement
    events: list[ConsensusEvent]  # majority canonical-label events (label view)
    excluded: list[Interval]  # no fine-level agreement


@dataclass(frozen=True)
class ConversationEvents:
    eot_positive_events: list[AnchorEvent]  # real turn ends (floor handed over)
    eot_negative_spans: list[Interval]  # believable mid-turn pauses
    int_positive_events: list[AnchorEvent]  # interruptions (interrupter takes floor)
    int_negative_spans: list[Interval]  # backchannel / bleed extents
    eot_excluded: list[Interval]  # turn-view disputes; EOT predictions masked
    int_excluded: list[Interval]  # label-view disputes; INT predictions masked


# ---- stage 1: annotation tracks -> consensus events -------------------------


def map_events(
    events: list[tuple[float, float, str]],
    canonical: dict[str, str],
) -> list[tuple[float, float, str]]:
    """Drop unmapped fine labels; rewrite the rest to canonical labels.

    Accepts (start, end, label) or (start, end, label, text) tuples; any
    trailing fields (the turnbench.data.Annotation transcript text) are dropped.
    """
    return [
        (start, end, canonical[label])
        for start, end, label, *_ in events
        if label in canonical
    ]


def merge_intervals(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Merge overlapping (start, end) intervals; returns them sorted."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def find_consensus(
    per_annotator: dict[str, list[tuple[float, float, str]]], speaker: int
) -> tuple[list[ConsensusEvent], list[Interval]]:
    """MIN_AGREEMENT-of-3 consensus over one speaker's canonical events.

    Anchor on each annotator's events in turn: gather the nearest unused
    same-label event from each *later* annotator (within OVERLAP_WINDOW_S of the
    anchor's start), then accept the largest group including the anchor whose
    starts and ends each span at most TIME_TOLERANCE_S, provided it reaches
    MIN_AGREEMENT members. The gold boundary is the median across the group.

    A dissenting annotator's event is settled by the majority, so an unmatched
    event becomes an excluded interval ONLY where it overlaps no accepted
    consensus event — a genuine no-majority region. Disagreement the majority
    already resolved is not re-introduced as uncertainty; otherwise it would
    mask the very event it dissents from.
    """
    events = {annotator: per_annotator[annotator] for annotator in ANNOTATORS}
    used: dict[str, set[int]] = {annotator: set() for annotator in ANNOTATORS}
    consensus: list[ConsensusEvent] = []

    def nearest(target_start: float, label: str, annotator: str) -> int | None:
        best_index, best_distance = None, float("inf")
        for index, (start, _, candidate_label) in enumerate(events[annotator]):
            if index in used[annotator] or candidate_label != label:
                continue
            distance = abs(start - target_start)
            if distance < best_distance and distance <= OVERLAP_WINDOW_S:
                best_distance, best_index = distance, index
        return best_index

    def largest_agreeing(
        group: dict[str, tuple[int, float, float]], anchor: str
    ) -> tuple[str, ...]:
        """The biggest subset of `group` that includes `anchor` and whose
        starts and ends each span at most TIME_TOLERANCE_S."""
        members = list(group)
        for size in range(len(members), 0, -1):
            for subset in combinations(members, size):
                if anchor not in subset:
                    continue
                starts = [group[annotator][1] for annotator in subset]
                ends = [group[annotator][2] for annotator in subset]
                if (
                    max(starts) - min(starts) <= TIME_TOLERANCE_S
                    and max(ends) - min(ends) <= TIME_TOLERANCE_S
                ):
                    return subset
        return ()

    for anchor_position, anchor in enumerate(ANNOTATORS):
        for anchor_index, (start, end, label) in enumerate(events[anchor]):
            if anchor_index in used[anchor]:
                continue
            group = {anchor: (anchor_index, start, end)}
            for other in ANNOTATORS[anchor_position + 1 :]:
                match_index = nearest(start, label, other)
                if match_index is not None:
                    match_start, match_end, _ = events[other][match_index]
                    group[other] = (match_index, match_start, match_end)
            agreeing = largest_agreeing(group, anchor)
            if len(agreeing) < MIN_AGREEMENT:
                continue
            starts = [group[annotator][1] for annotator in agreeing]
            ends = [group[annotator][2] for annotator in agreeing]
            consensus.append(
                ConsensusEvent(
                    speaker, round(median(starts), 4), round(median(ends), 4), label
                )
            )
            for annotator in agreeing:
                used[annotator].add(group[annotator][0])

    consensus_spans = [(event.start, event.end) for event in consensus]
    unmatched = [
        (start, end)
        for annotator in ANNOTATORS
        for index, (start, end, _) in enumerate(events[annotator])
        if index not in used[annotator]
        and not any(
            span_start < end and start < span_end
            for span_start, span_end in consensus_spans
        )
    ]
    excluded = [
        Interval(speaker, round(start, 4), round(end, 4))
        for start, end in merge_intervals(unmatched)
    ]
    return consensus, excluded


def consensus_for_conversation(
    conv: Conversation,
    canonical: dict[str, str] = CANONICAL,
) -> tuple[list[ConsensusEvent], list[Interval]]:
    """Build the consensus events + excluded intervals for one conversation,
    judged at the granularity of `canonical` (the label view by default, the
    coarser turn view with TURN_CANONICAL).

    Reads the three annotator tracks per speaker from `conv.annotations`.
    Returns (consensus events across both speakers, excluded intervals).
    """
    events: list[ConsensusEvent] = []
    excluded: list[Interval] = []
    for speaker in SPEAKERS:
        per_annotator = {
            annotator: map_events(conv.annotations[(speaker, annotator)], canonical)
            for annotator in ANNOTATORS
        }
        speaker_events, speaker_excluded = find_consensus(per_annotator, speaker)
        events.extend(speaker_events)
        excluded.extend(speaker_excluded)
    return events, excluded


# ---- stage 2: consensus views -> scored event sets ---------------------------


def first_start_after(start_times: list[float], time_s: float) -> float:
    """Smallest start strictly greater than time_s, or +inf. `start_times` must be sorted."""
    index = bisect.bisect_right(start_times, time_s)
    return start_times[index] if index < len(start_times) else float("inf")


@dataclass(frozen=True)
class SpeakerTurns:
    """One speaker's turn-view `Turn` segments, sorted by start time."""

    speaker: int
    segments: list[ConsensusEvent]

    @property
    def start_times(self) -> list[float]:
        return [segment.start for segment in self.segments]


def collect_turns(events: list[ConsensusEvent], speaker: int) -> SpeakerTurns:
    segments = sorted(
        (
            event
            for event in events
            if event.label == "Turn" and event.speaker == speaker
        ),
        key=lambda event: event.start,
    )
    return SpeakerTurns(speaker, segments)


def truncated_pause(
    span: Interval, views: ConsensusViews, other_speaker: int
) -> Interval | None:
    """Cut the pause at the first evidence against "a quiet within-turn hold":
    an excluded interval in either view (disputed is unknown, not silent), the
    speaker's own non-floor vocalisation, or an other-speaker Interruption.
    Returns None when nothing believable remains."""
    cut_times = [span.end]
    for interval in views.turn_excluded + views.excluded:
        if interval.start < span.end and interval.end > span.start:
            cut_times.append(interval.start)
    for event in views.events:
        own_vocalisation = event.speaker == span.speaker and event.label != "Turn"
        other_barge_in = (
            event.speaker == other_speaker and event.label in INTERRUPTION_LABELS
        )
        if (own_vocalisation or other_barge_in) and span.start < event.start < span.end:
            cut_times.append(event.start)
    end = min(cut_times)
    if end <= span.start:
        return None
    return Interval(span.speaker, span.start, end)


def build_conversation_events(views: ConsensusViews) -> ConversationEvents:
    """Derive the EOT and Interruption event sets from the consensus views."""
    speaker1_turns = collect_turns(views.turn_events, 1)
    speaker2_turns = collect_turns(views.turn_events, 2)

    eot_positive_events: list[AnchorEvent] = []
    eot_negative_spans: list[Interval] = []
    for speaker_turns in [speaker1_turns, speaker2_turns]:
        other_speaker_turns = (
            speaker2_turns if speaker_turns.speaker == 1 else speaker1_turns
        )
        self_start_times = speaker_turns.start_times
        other_start_times = other_speaker_turns.start_times
        for segment in speaker_turns.segments:
            # An end strictly inside another of the speaker's own segments
            # (e.g. a floor-taking interruption within their longer turn) is
            # not a turn boundary at all.
            inside_own_segment = any(
                own is not segment and own.start < segment.end < own.end
                for own in speaker_turns.segments
            )
            if inside_own_segment:
                continue
            # A real EOT: the floor leaves this speaker — the other speaker
            # takes over before this speaker resumes, is already mid-turn as
            # this segment ends (an overlapping take-over), or nobody speaks
            # again. Otherwise it is a mid-turn pause.
            next_self = first_start_after(self_start_times, segment.end)
            next_other = first_start_after(other_start_times, segment.end)
            other_is_speaking = any(
                other.start < segment.end < other.end
                for other in other_speaker_turns.segments
            )
            if next_self == float("inf") or next_other < next_self or other_is_speaking:
                anchor = AnchorEvent(speaker_turns.speaker, segment.end)
                if anchor not in eot_positive_events:
                    eot_positive_events.append(anchor)
            else:
                pause = truncated_pause(
                    Interval(speaker_turns.speaker, segment.end, next_self),
                    views,
                    other_speaker_turns.speaker,
                )
                if pause is not None and pause not in eot_negative_spans:
                    eot_negative_spans.append(pause)

    int_positive_events = [
        AnchorEvent(event.speaker, event.start)
        for event in views.events
        if event.label == "Interruption"
    ]
    int_negative_spans = [
        Interval(event.speaker, event.start, event.end)
        for event in views.events
        if event.label in INT_NEGATIVE_LABELS
    ]
    # Consensus non-floor-taking attempts are neither positives nor
    # negatives: at onset they are indistinguishable from real interruptions,
    # so firing on one is neither rewarded nor penalised.
    int_excluded = views.excluded + [
        Interval(event.speaker, event.start, event.end)
        for event in views.events
        if event.label == "NonFloorTakingInterruption"
    ]

    return ConversationEvents(
        eot_positive_events,
        eot_negative_spans,
        int_positive_events,
        int_negative_spans,
        eot_excluded=views.turn_excluded,
        int_excluded=int_excluded,
    )


def events_for_conversation(conv: Conversation) -> ConversationEvents:
    """One conversation's annotation tracks -> the scored EOT / INT event sets.

    Consensus is built twice, at the granularity each task needs: the turn
    view (floor-claiming vs not) drives turns and EOT; the label view drives
    the INT task.
    """
    events, excluded = consensus_for_conversation(conv)
    turn_events, turn_excluded = consensus_for_conversation(
        conv, canonical=TURN_CANONICAL
    )
    return build_conversation_events(
        ConsensusViews(turn_events, turn_excluded, events, excluded)
    )
