#!/usr/bin/env python3
"""Evaluate your action-reconstruction server on the public validation clip.

    python /task/validate.py --server-url http://127.0.0.1:8000

Posts each clip as multipart/form-data (fields `video` and `id`) to POST /actions,
exactly as the trusted final client does, scores the returned actions and reports
micro action F1 across all clips and action types. It measures quality only. The `id`
field is the clip's position in the batch (`clip-000`, `clip-001`, ...), never the
clip's name: the dataset names clips after their upstream files, whose suffix is the
curator's category label, and the server must not see it.

Time limits are those of final grading: clips are sent one at a time, each request
must answer within 300 seconds (`--request-timeout-seconds`; raise it only to debug a
slow CPU prototype) and the whole batch within `--timeout-seconds` (3600, the budget
for all held-out clips together).

Scoring never aborts because of one clip or one action:
- Actions are parsed as by the upstream evaluator (`parse_actions`): items that are not
  objects, have an unknown `type` or a `frame` not of the form `Fnn` are dropped, extra
  keys are ignored and `details` is read as a string. Frame labels past the labelled
  window (F00-F24) are not rejected: they are matched like any other prediction, so a
  KeyPress/MouseClick there can still match a label within the frame tolerance and
  anything else counts as a false positive. Trim output to F00-F24.
- A request that fails (non-200, connection error, invalid JSON) is retried twice on the
  same body after a short backoff, inside the clip's time window; a clip that still fails,
  times out (never retried) or returns something other than a JSON array scores as an empty
  prediction (all of its labels become false negatives) and is listed under
  `details.failed_clips` (request failures also under `details.failed_requests`). More than
  5% failed clips fails the evaluation.
Deviations from the canonical schema are reported per clip under `warning` and
`dropped_actions`; fix them locally, they cost precision or recall in grading.
"""
import json
import math
import re
import time
import uuid
from pathlib import Path

from actions import compute_prf, score_actions
from frontier_ml.validation.cli import public_main
from frontier_ml.validation.errors import VerificationFailure
from frontier_ml.validation.evaluation import FailedRequests, RequestFailure, attempt_request
from frontier_ml.validation.progress import progress

TASK = "idm-training"
DEFAULT_DATA = "/task/public/validation.jsonl"
ACTION_TYPES = {'KeyPress', 'MouseClick', 'MouseMove', 'MouseScroll'}
REQUEST_TIMEOUT_SECONDS = 300  # per clip, as in final grading
BATCH_TIMEOUT_SECONDS = 3600  # all clips together, as in final grading
MAX_FAILED_CLIP_PERCENT = 5  # more failed clips than this is a broken server, not a hiccup
FRAME = re.compile(r'F\d{2}')


def request_id(index):
    """The `id` form field of the request for the clip at `index`: its position in the batch, nothing else.

    The dataset rows are named after their upstream files, whose suffix (`_scroll_drag`, `_keystroke-heavy`,
    ...) is the curator's clip category and the name of the clip's public label file. That name never reaches
    the server; each response is mapped back to its row by position for scoring, and the row's name stays in
    the operator-facing per-clip details.
    """
    return f'clip-{index:03d}'


def build_request(sample, data_root, index):
    """The multipart body and headers of the request for the clip at `index`: the clip's position as `id`
    (request_id) and its MP4 as `video`, read from below `data_root`."""
    root = Path(data_root)
    video = (root / sample['video']).resolve()
    if not video.is_relative_to(root.resolve()):
        raise ValueError('Video escapes dataset root')
    boundary = uuid.uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="id"\r\n\r\n{request_id(index)}\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="video"; filename="clip.mp4"\r\n'
            'Content-Type: video/mp4\r\n\r\n').encode() + video.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
    return body, {'Content-Type': f'multipart/form-data; boundary={boundary}'}


def validate_actions(actions, n_frames):
    """Strict canonical-schema check; raises ValueError describing the first deviation.

    Only used to produce a per-clip warning. Scoring uses parse_actions and never
    rejects a clip or an evaluation because of a non-canonical action.
    """
    if not isinstance(actions, list):
        raise ValueError('Response must be a JSON array of canonical action objects')
    for index, event in enumerate(actions):
        where = f'action {index}'
        if not isinstance(event, dict) or set(event) != {'frame', 'type', 'details'}:
            raise ValueError(f'{where}: each action needs exactly frame, type, details')
        frame, kind, detail = event['frame'], event['type'], event['details']
        if not isinstance(frame, str) or not FRAME.fullmatch(frame):
            raise ValueError(f'{where}: frame must be a label F00-F99')
        if int(frame[1:]) >= n_frames:
            raise ValueError(f'{where}: {frame} is past the labelled window F00-F{n_frames - 1:02d}; only a '
                             'KeyPress/MouseClick within the frame tolerance can match there, anything else is a false positive')
        if kind not in ACTION_TYPES or not isinstance(detail, str):
            raise ValueError(f'{where}: invalid canonical action type/details')
        if kind == 'MouseMove' and not re.fullmatch(r'-?\d+,-?\d+', detail):
            raise ValueError(f'{where}: MouseMove requires integer normalized dx,dy')
        if kind == 'MouseScroll' and not re.fullmatch(r'-?\d+', detail):
            raise ValueError(f'{where}: MouseScroll requires an integer normalized amount')
        if kind == 'MouseClick' and detail not in {'Left', 'Right', 'Middle'}:
            raise ValueError(f'{where}: unknown mouse button')
        if kind == 'KeyPress' and not detail:
            raise ValueError(f'{where}: KeyPress details cannot be empty')
    return actions


def parse_actions(response):
    """The actions the upstream evaluator would score, and how many items were dropped.

    Mirrors the upstream response parser: keep objects whose type is known and whose
    frame is a label `Fnn` (an integer 0-99 is accepted as well), read details as a
    string, ignore extra keys and drop everything else. Frames past the labelled window
    are kept; the pinned matcher decides what they match. A JSON object wrapping exactly
    one array (such as {"actions": [...]}) is unwrapped; any other shape raises ValueError
    and the clip counts as failed.
    """
    if isinstance(response, dict):
        arrays = [value for value in response.values() if isinstance(value, list)]
        if len(arrays) != 1:
            raise ValueError('Response must be a JSON array of action objects')
        response = arrays[0]
    if not isinstance(response, list):
        raise ValueError('Response must be a JSON array of action objects')
    kept, dropped = [], 0
    for item in response:
        frame = item.get('frame') if isinstance(item, dict) else None
        if isinstance(frame, int) and not isinstance(frame, bool) and 0 <= frame <= 99:
            frame = f'F{frame:02d}'
        if isinstance(frame, str):
            frame = frame.strip()
        if not isinstance(frame, str) or not FRAME.fullmatch(frame) or item.get('type') not in ACTION_TYPES:
            dropped += 1
            continue
        kept.append({'frame': frame, 'type': item['type'], 'details': str(item.get('details', ''))})
    return kept, dropped


def score(samples, predictions, config=None):
    """Micro action F1 over all clips.

    A prediction is a decoded response, or the RequestFailure (or exception) recorded for a request that
    produced none; such a clip scores as an empty prediction.
    """
    if not samples or len(samples) != len(predictions):
        raise ValueError('Must produce exactly one response per clip')
    totals = {'tp': 0, 'fp': 0, 'fn': 0}
    by_type = {kind: {'tp': 0, 'fp': 0, 'fn': 0} for kind in sorted(ACTION_TYPES)}
    per_clip, failed, dropped_total = [], [], 0
    for index, (sample, response) in enumerate(zip(samples, predictions, strict=True)):
        error = warning = None
        actions, dropped = [], 0
        if isinstance(response, RequestFailure):  # no usable response after the shared retries; recorded by run()
            error = f'{response.reason}: {response.error}'
        elif isinstance(response, BaseException):
            error = f'{type(response).__name__}: {response}'
        else:
            try:
                actions, dropped = parse_actions(response)
            except ValueError as exc:
                error = f'{type(exc).__name__}: {exc}'
        if error is not None:
            failed.append({'id': sample['id'], 'error': error})
        else:
            try:
                validate_actions(response, sample['n_frames'])
            except (ValueError, TypeError) as exc:
                warning = str(exc)
        dropped_total += dropped
        result = score_actions(sample['actions'], sample['start_s'], sample['fps'],
                               sample['n_frames'], sample['video_w'], sample['video_h'], actions)
        for name in totals:
            totals[name] += result['overall'][name]
            for kind in by_type:
                by_type[kind][name] += result[kind][name]
        per_clip.append({'id': sample['id'], 'request_id': request_id(index), **result['overall'],
                         'per_type': {kind: result[kind] for kind in by_type},
                         'mouse_dynamics': result['mouse_dynamics'],
                         'dropped_actions': dropped, 'error': error, 'warning': warning})
    if failed and len(failed) * 100 > MAX_FAILED_CLIP_PERCENT * len(samples):
        raise VerificationFailure(f'{len(failed)} of {len(samples)} clips had no scoreable response, more than '
                                  f'{MAX_FAILED_CLIP_PERCENT}%; first: {failed[0]["id"]}: {failed[0]["error"]}')
    # Sum counts across clips, then compute PRF, as in the reference evaluator.
    # In particular, an entirely empty dataset result has F1 zero, even though
    # the action-list scorer returns F1 one for a single empty/empty clip.
    precision, recall, value = compute_prf(**totals)
    for counts in by_type.values():
        p, r, f1 = compute_prf(**counts)
        counts.update(precision=p, recall=r, f1=f1)
    return {'score': value, 'details': {'metric': 'micro_action_f1', **totals,
                                      'precision': precision, 'recall': recall,
                                      'per_type': by_type, 'failed_clips': failed,
                                      'failed_clip_limit_percent': MAX_FAILED_CLIP_PERCENT,
                                      'dropped_actions': dropped_total, 'clips': per_clip}}


def run(transport, data_path, config):
    config = config or {}
    path = Path(data_path)
    samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not samples:
        raise ValueError('Evaluation dataset is empty')
    expected = config.get('expected_samples')
    if not config.get('validation_mode') and expected is not None and len(samples) != int(expected):
        raise ValueError('Held-out split count mismatch')
    per_request = float(config.get('request_timeout_seconds', REQUEST_TIMEOUT_SECONDS))
    budget = float(config.get('inference_timeout_seconds', BATCH_TIMEOUT_SECONDS))
    if not all(value > 0 and math.isfinite(value) for value in (per_request, budget)):
        raise ValueError('Request and batch timeouts must be positive and finite')
    predictions = []
    # The shared per-item policy (frontier_ml.validation.evaluation): a request that fails at the transport level is
    # retried twice on the identical body, a clip that still fails scores as an empty prediction, and more than
    # MAX_FAILED_CLIP_PERCENT of the clips failing that way fails the run closed (VerificationFailure) at once.
    failures = FailedRequests(len(samples), limit_percent=MAX_FAILED_CLIP_PERCENT)
    started = time.monotonic()
    deadline = started + budget
    progress(config, 'requests', completed=0, total=len(samples))
    for index, sample in enumerate(samples):
        body, headers = build_request(sample, path.parent, index)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            response = RequestFailure('timeout', f'{budget:g}s batch deadline exceeded before this clip was sent')
        else:
            # Every attempt at this clip, and the backoff between attempts, runs inside the clip's window: the
            # per-request ceiling, or what is left of the batch budget. A timeout is not retried. Verifier faults
            # (EvaluationError) propagate; every other request failure becomes a RequestFailure.
            response = attempt_request(transport, 'POST', '/actions', body=body, headers=headers,
                                       timeout=min(per_request, remaining), failures=failures)
        if isinstance(response, RequestFailure):
            failures.record(sample['id'], response)  # fails closed above MAX_FAILED_CLIP_PERCENT
        predictions.append(response)
        progress(config, 'requests', completed=index + 1, total=len(samples))
    result = score(samples, predictions, config)
    result['details'].update(elapsed_seconds=time.monotonic() - started, deadline_seconds=budget,
                             request_timeout_seconds=per_request, failed_requests=failures.summary())
    return result


def _arguments(parser):
    parser.add_argument('--request-timeout-seconds', type=float,
                        help=f'per-clip ceiling (default: the smaller of --timeout-seconds and {REQUEST_TIMEOUT_SECONDS}, as in grading)')


def _config(args):
    return {'request_timeout_seconds': args.request_timeout_seconds or min(args.timeout_seconds, REQUEST_TIMEOUT_SECONDS)}


def main(argv=None):
    return public_main(TASK, run, argv, default_data=DEFAULT_DATA, description=__doc__,
                       extra_arguments=_arguments, extra_config=_config)


if __name__ == "__main__":
    raise SystemExit(main())
