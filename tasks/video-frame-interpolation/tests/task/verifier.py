"""Review-bound inference followed by trusted, offline H100 quality metrics."""
import errno
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

from frontier_ml.scoring.architecture import verify_review
from frontier_ml.taskspec import sibling
from frontier_ml.validation.errors import EvaluationError, VerificationFailure
from frontier_ml.validation.evaluation import load_samples
from frontier_ml.validation.progress import progress
from vfi_evaluate import calibration, evaluation_identity
from vfi_gpu import score_on_h100
from vfi_protocol import (FACTORS, MAX_DIMENSION, MAX_FRAMES, MAX_SEQUENCE_BYTES, REQUEST_KEYS, build_request,
                          media_path, normalize_sample)
from vfi_runtime import CUDA_RUNTIME
from vfi_scoring import SCORING_PROTOCOL, number, require_evaluation, score
from vfi_selection import DEFAULT_SELECTION, select_intervals

public = sibling(__file__, "validate.py", require=("run",))
REVIEW_CHECKS = list(sibling(__file__, "review.py").POLICY["checks"])
# The self-check reads the public rows and the endpoint PNGs they name. Frame names depend on the data, so every
# public PNG is fetched (the pinned split has 17, about 4 MB).
SELFCHECK_PUBLIC = ("validation.jsonl", "media/*.png")
# A self-check is a smoke test and the speed gate, not a second evaluation: a pass that takes longer than this at the
# required rate is sampled, at least 6 intervals per factor (up to 3 opening rounds counted once, then 3 or more
# timed ones).
SELFCHECK_PASS_SECONDS = 600
SELFCHECK_MIN_PER_FACTOR = 6


def launch(snapshot, config, data_path):
    # Keep Harbor optional when loading the task for public validation/viewers.
    from frontier_ml.runtime.inference import serve_submission
    from frontier_ml.runtime.resources import resource_settings

    effective = config
    if region := os.environ.get("MODAL_REGION"):
        settings = resource_settings(config, "inference")
        if settings["type"] == "modal":
            # Every request transfers two images and the generated RGB frames.
            # Cross-region filesystem RPCs can dominate the whole-run deadline.
            resources = config.get("grading_resources")
            if resources is None:
                resources = json.loads(os.environ.get("FML_GRADING_RESOURCES") or "{}")
            inference = {key: settings[key] for key in ("type", "image") if key in settings}
            inference["kwargs"] = {**settings["kwargs"], "region": region}
            effective = {**config, "grading_resources": {**resources, "inference": inference}}
    return serve_submission(snapshot, effective)


def default_config():
    return {"scoring_protocol": SCORING_PROTOCOL, "evaluation_sha256": None,
            "calibration_path": "/tests/task/calibration.json", "calibration_sha256": None,
            "metric_bundle": "/opt/vfi-metrics", "metric_bundle_sha256": None,
            "minimum_generated_frames_per_second": 1.0, "concurrency": 1,
            "response_format": "rgb8",
            "evaluation_selection": dict(DEFAULT_SELECTION),
            "inference_timeout_seconds": 3600, "request_timeout_seconds": 300,
            "metric_runtime": dict(CUDA_RUNTIME),
            "metric_timeout_seconds": 3600}


def prescreen(config, data_path):
    """Reject incomplete operator configuration before paid review or candidate launch."""
    if (config.get("scoring_protocol") != SCORING_PROTOCOL or not config.get("evaluation_sha256")
            or not config.get("calibration_sha256") or not config.get("metric_bundle_sha256")
            or config.get("minimum_generated_frames_per_second") is None):
        require_evaluation()
    if type(config.get("concurrency")) is not int or config["concurrency"] != 1:
        raise VerificationFailure("Evaluation requires concurrency=1")
    for key in ("minimum_generated_frames_per_second", "inference_timeout_seconds", "request_timeout_seconds",
                "metric_timeout_seconds"):
        number(config.get(key), key, positive=True)
    samples, identity = evaluation_identity(data_path, selection=config.get("evaluation_selection"))
    if config["evaluation_sha256"] != identity["evaluation_sha256"] or config.get("expected_samples") != len(samples):
        raise VerificationFailure("Evaluation media/rows/count differ from the frozen evaluation")
    if config.get("metric_runtime") != CUDA_RUNTIME:
        raise VerificationFailure("Production metrics require the pinned H100 CUDA runtime")
    record = calibration(config, identity)
    from vfi_metrics import bundle_files
    bundle_files(config["metric_bundle"], expected_sha256=config["metric_bundle_sha256"])
    config["_vfi_ready"] = (record, samples, sum(len(s["target_indices"]) for s in samples))


def preflight(config, data_path):
    if "_vfi_ready" not in config:
        prescreen(config, data_path)
    receipt = config.get("architecture_review_path")
    if not receipt or not Path(receipt).is_file():
        raise VerificationFailure("Trusted architecture_review_path is required before provisioning")
    if config.get("submission_path"):
        verify_review(config["submission_path"], receipt, "video-frame-interpolation", REVIEW_CHECKS)


def _deadline(config, generated):
    """The whole generation pass's budget: the frame-rate floor, capped by the inference timeout."""
    return min(config["inference_timeout_seconds"], generated / config["minimum_generated_frames_per_second"])


def _generate(transport, data_path, config, output, deadline):
    # Reuse the exact public HTTP/parser/export path. Official scoring happens only in this verifier.
    return public.run(transport, data_path, dict(config, validation_mode=True, predictions_only=True,
                                                 predictions_dir=str(output), inference_timeout_seconds=deadline))


def _predict(server, data_path, config, output):
    review = verify_review(server["snapshot"], config["architecture_review_path"],
                           "video-frame-interpolation", REVIEW_CHECKS)
    _, _, generated = config["_vfi_ready"]
    deadline = _deadline(config, generated)
    diagnostic = _generate(server["transport"], data_path, config, output, deadline)
    return diagnostic, review, deadline


def _measure(data_path, config, output, diagnostic, review, deadline):
    record, samples, _ = config["_vfi_ready"]
    progress(config, "metrics", completed=0, total=len(samples), message="Staging outputs for the scoring H100")
    report = score_on_h100(data_path, output, config, samples)
    progress(config, "metrics", completed=len(samples), total=len(samples))
    result = score(report["intervals"], record["baseline"], record["target"], config)
    result["details"].update(metric_seconds=report["metric_seconds"],
                              scoring_seconds=report["scoring_seconds"],
                              scoring_gpu=report["gpu"], scoring_runtime=report["runtime"],
                              evaluation_sha256=config["evaluation_sha256"],
                              metric_bundle_sha256=config["metric_bundle_sha256"],
                              calibration_sha256=config["calibration_sha256"],
                              prediction_manifest_sha256=report["prediction_manifest_sha256"],
                              timing=diagnostic["details"]["timing"],
                              gates={"generation_deadline_seconds": deadline,
                                     "architecture": {"reviewer": review.get("reviewer"),
                                                      "checks": review.get("checks"),
                                                      "trajectory": review.get("trajectory")}})
    return result


def verify_isolated(launch, data_path, config):
    """Destroy the candidate H100 before allocating the separate scoring H100."""
    preflight(config, data_path)
    with tempfile.TemporaryDirectory(prefix="frontier-vfi-") as temporary:
        output = Path(temporary) / "predictions"
        with launch() as server:
            diagnostic, review, deadline = _predict(server, data_path, config, output)
        return _measure(data_path, config, output, diagnostic, review, deadline)


def verify(server, data_path, config):
    raise VerificationFailure("VFI grading requires verify_isolated: release the candidate before H100 scoring")


def _selfcheck_sizes(width, height):
    """Frame sizes the public examples lack: their frames turned a quarter (landscape for portrait examples), then
    that turn enlarged 1.5x to odd dimensions when the protocol's limits allow it. Requests may have any W and H up to
    4096, so a server built for one shape or stride must fail the self-check as it would fail grading."""
    sizes = [(height, width)]
    large = ((height * 3 // 2) | 1, (width * 3 // 2) | 1)
    if max(large) <= MAX_DIMENSION and large[0] * large[1] * 3 * MAX_FRAMES <= MAX_SEQUENCE_BYTES:
        sizes.append(large)
    return sizes


def _selfcheck_resized(sample, root, directory, size):
    """The interval with its endpoint PNGs turned a quarter and resized to `size`, written under directory/sizes/."""
    from PIL import Image  # the verifier image's Pillow; validate.py stays stdlib-only

    frames = []
    for frame in sample["input_frames"]:
        path = f"sizes/{size[0]}x{size[1]}/{frame['path']}"
        (directory / path).parent.mkdir(parents=True, exist_ok=True)
        with Image.open(media_path(root, frame["path"])) as image:
            turned = image.convert("RGB").transpose(Image.Transpose.ROTATE_90)
            turned.resize(size, Image.Resampling.BICUBIC).save(directory / path, format="PNG")
        frames.append({"index": frame["index"], "path": path})
    resized = {**sample, "width": size[0], "height": size[1], "input_frames": frames}
    build_request(resized, directory)  # the grader's PNG checks on what Pillow wrote
    return resized


def _selfcheck_plan(config):
    """(per, full, whole): the intervals per factor the self-check sends, then grading's count and whole-pass deadline.
    Grading's whole pass when it fits SELFCHECK_PASS_SECONDS at the required rate, otherwise the share of every factor
    that does, timed by _selfcheck_sampled."""
    full = config["evaluation_selection"]["per_factor"]
    whole = _deadline(config, full * sum(factor - 1 for factor in FACTORS))
    per = full
    if whole > SELFCHECK_PASS_SECONDS:
        per = min(full, max(SELFCHECK_MIN_PER_FACTOR, math.floor(full * SELFCHECK_PASS_SECONDS / whole)))
    return per, full, whole


def _frames(rows):
    return sum(len(row["target_indices"]) for row in rows)


def _selfcheck_pass(public_dir, directory, selection, sampled):
    """The evaluation pass's shape from the public intervals: `selection`'s count of intervals of each factor
    (grading's, or the self-check's sample of it), resampled under unique IDs and interleaved by factor, so a factor
    the server mishandles fails within the first requests. The second and third rounds change the frame size
    (_selfcheck_sizes), so a size the server mishandles fails within the first nine. Only the endpoint PNGs are
    copied; references never go on the wire. Returns the pass as parts, each (JSONL file, rows): the whole pass, or a
    sample's opening rounds (the native round and the size rounds, which bring its shapes for the first time) and the
    rest; then the public frame sizes."""
    try:
        source = Path(public_dir) / "validation.jsonl"
        samples = [normalize_sample(sample) for sample in load_samples(source)]
        groups = {factor: [] for factor in FACTORS}
        for sample in samples:
            build_request(sample, source.parent)  # the grader's checks of the endpoint PNGs, before any GPU
            groups[sample["frame_count"] - 1].append(sample)
            for frame in sample["input_frames"]:
                (directory / frame["path"]).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(media_path(source.parent, frame["path"]), directory / frame["path"])
        if not all(groups.values()):
            raise ValueError("the public intervals do not cover factors 4, 8 and 16")
        rows, opening = [], len(FACTORS)
        for copy in range(selection["per_factor"]):
            for factor in FACTORS:
                sample = groups[factor][copy % len(groups[factor])]
                sizes = _selfcheck_sizes(sample["width"], sample["height"])
                if 1 <= copy <= len(sizes):  # replaces a resampled copy: the pass keeps its intervals and frames
                    sample = _selfcheck_resized(sample, source.parent, directory, sizes[copy - 1])
                    opening = len(FACTORS) * (copy + 1)
                rows.append({**{key: sample[key] for key in REQUEST_KEYS}, "id": f"{sample['id']}-{copy:03d}"})
        parts = [rows[:opening], rows[opening:]] if sampled else [rows]
        files = []
        for number, part in enumerate(parts, 1):
            # public.run applies the evaluation selection to each part: it must keep every row, in order.
            if len(select_intervals(part, {**selection, "per_factor": len(part) // len(FACTORS)})) != len(part):
                raise ValueError("the evaluation selection does not keep the whole resampled pass")
            data = directory / f"pass-{number}.jsonl"
            data.write_text("".join(json.dumps(row) + "\n" for row in part))
            files.append((data, part))
    except (KeyError, TypeError, OSError, ValueError, ImportError, VerificationFailure) as error:
        # Public data, configuration or the controller's image: not the submission's fault.
        raise EvaluationError(f"The public intervals could not be prepared: {type(error).__name__}: {error}") from error
    return files, {(sample["width"], sample["height"]) for sample in samples}


def _selfcheck_smoke(check, output):
    """The pass's first interval as the smoke sample: the grader's request, raw RGB answer and parser."""
    with (output / "predictions.jsonl").open() as stream:
        first = json.loads(stream.readline())
    check.record("POST /interpolate", True, kind="smoke", measured=round(first["request_seconds"], 3), unit="s",
                 detail=f"the first public interval ({first['interpolation_factor']}x, {first['width']}x"
                        f"{first['height']}): {len(first['target_indices'])} raw RGB frames of the requested size, "
                        "accepted by the grader's parser")
    return first["request_seconds"]


def _selfcheck_generate(check, server, part, config, output, deadline, native, late, *, offset=0, count=None):
    """public.run over one part of the pass within `deadline`, as grading runs its pass; returns its elapsed seconds,
    transport and export included. A late, failed or rejected answer ends the self-check where grading would fail the
    pass; `late(done, elapsed, message)` records a late part. `offset` and `count` place the part in the pass."""
    data, rows = part
    selection = {**config["evaluation_selection"], "per_factor": len(rows) // len(FACTORS)}
    started = time.monotonic()
    try:
        diagnostic = _generate(server["transport"], data, dict(config, evaluation_selection=selection), output,
                               deadline)
    except EvaluationError:
        raise
    except Exception as error:  # noqa: BLE001 - a late, failed or rejected answer fails the pass, as in grading
        if isinstance(error, OSError) and error.errno in (errno.ENOSPC, errno.EDQUOT):
            # The controller's disk filled while exporting frames: a self-check fault, refunded.
            raise EvaluationError(f"The pass's frame export ran out of disk: {error}") from error
        _selfcheck_failed(check, output, rows, offset, count or len(rows), native, error,
                          time.monotonic() - started, deadline, late, config)
    return diagnostic["details"]["timing"]["elapsed_seconds"]


def _selfcheck_failed(check, output, rows, offset, count, native, error, elapsed, deadline, late, config):
    """End the self-check where grading would have failed the pass: late (`late` records it), or on a failed or
    rejected request, named by its place among the pass's `count` intervals."""
    try:
        done = json.loads((output / "run.json").read_text())["samples_completed"]
    except FileNotFoundError:
        done = 0
    if done and not offset:
        _selfcheck_smoke(check, output)
    message = f"{type(error).__name__}: {error}"
    if elapsed >= deadline or "whole-run deadline" in message:
        late(done, elapsed, message)
    row = rows[min(done, len(rows) - 1)]
    where = f"{row['frame_count'] - 1}x interval at {row['width']}x{row['height']}"
    if (row["width"], row["height"]) not in native:
        where += " (made from the public examples at a size they lack; requests may have any W and H up to 4096)"
    if "timed out" in message.lower() or "TimeoutError" in message:
        message += f" (grading allows each request {config['request_timeout_seconds']:g} s)"
    if not done and not offset:
        check.fail("POST /interpolate", f"the first public interval ({where}) failed: {message}", kind="smoke")
    check.fail(f"POST /interpolate, interval {offset + done + 1} of {count}", f"{where}: {message}", kind="smoke")


def _selfcheck_whole(check, server, part, native, whole, config, output):
    """Grading's whole pass against its deadline. Returns the measured frame rate and the timing text."""
    rows, total = part[1], _frames(part[1])

    def late(done, elapsed, message):
        frames = _frames(rows[:done])
        check.fail("generation rate", f"{done} of {len(rows)} intervals ({frames} of {total} frames) were generated "
                                      f"before the {whole:g} s whole-pass deadline ({message})",
                   measured=frames / elapsed, required=total / whole, unit="frames/s", higher_is_better=True)

    elapsed = _selfcheck_generate(check, server, part, config, output, whole, native, late)
    _selfcheck_smoke(check, output)
    return total / max(elapsed, 1e-9), (f"{total} frames in {elapsed:.1f} s of the {whole:g} s allowed, including "
                                        "transport and export")


def _selfcheck_sampled(check, server, parts, native, full, whole, config, outputs):
    """A sample of grading's pass, projected to it. Grading pays a cold start once, and any per-shape setup (a
    compile or engine per frame size and factor) once at each shape it sends: the opening rounds, which bring each of
    the sample's shapes for the first time, count once, within their frames' time at the required rate plus one
    request timeout (the self-check's own limit). The rest is timed against its share of what is left of the
    deadline and scaled to grading's other frames. Returns the projected frame rate and the timing text."""
    (_, opening), (_, rest) = parts
    total = full * sum(factor - 1 for factor in FACTORS)
    opened_frames, rest_frames = _frames(opening), _frames(rest)
    others = total - opened_frames
    count, sent = full * len(FACTORS), len(opening) + len(rest)
    sampled = f"projected from {sent} of {count} requests at concurrency 1"
    note = (f"The generation rate was projected from a sample: {sent} of the evaluation's {count} intervals "
            f"({opened_frames + rest_frames} of its {total} frames), the first {len(opening)} counted once and the "
            f"rest scaled to the evaluation, not timed over a whole pass.")
    timeout = config["request_timeout_seconds"]
    limit = min(whole, timeout + opened_frames / config["minimum_generated_frames_per_second"])

    def opening_late(done, elapsed, message):
        check.note(note)
        check.fail("generation rate", f"{done} of the first {len(opening)} intervals ({_frames(opening[:done])} of "
                                      f"{opened_frames} frames), which bring the sample's frame sizes and factors for "
                                      f"the first time, were generated before the self-check's own {limit:g} s limit "
                                      f"for them: their frames at the required rate plus one {timeout:g} s request "
                                      f"timeout for the cold start and per-shape setup ({message})",
                   required=total / whole, unit="frames/s")

    opened = _selfcheck_generate(check, server, parts[0], config, outputs[0], limit, native, opening_late, count=sent)
    first = _selfcheck_smoke(check, outputs[0])
    share = max((whole - opened) * rest_frames / others, 1e-6)  # public.run ended the opening before its limit

    def rest_late(done, elapsed, message):
        frames = _frames(rest[:done])
        check.note(note)
        check.fail("generation rate", f"{done} of the other {len(rest)} intervals ({frames} of {rest_frames} frames) "
                                      f"were generated before their {share:.1f} s share of the {whole:g} s whole-pass "
                                      f"deadline after {opened:.1f} s for the first {len(opening)}, {sampled} "
                                      f"({message})",
                   measured=total / (opened + elapsed * others / frames) if frames else 0.0, required=total / whole,
                   unit="frames/s", higher_is_better=True)

    steady = _selfcheck_generate(check, server, parts[1], config, outputs[1], share, native, rest_late,
                                 offset=len(opening), count=sent)
    projected = opened + steady * others / rest_frames
    check.note(note)
    shapes = len({(row["width"], row["height"], row["frame_count"]) for row in opening})
    return total / projected, (
        f"{opened_frames + rest_frames} frames in {opened + steady:.1f} s, {sampled}: the first {len(opening)} "
        f"intervals, which bring each of the sample's {shapes} (size, factor) shapes for the first time, took "
        f"{opened:.1f} s (the cold first one {first:.1f} s) and count once; the other {len(rest)} ({rest_frames} "
        f"frames) took {steady:.1f} s, scaled to grading's other {others} frames: {projected:.0f} s of the {whole:g} s "
        f"allowed for {total} frames, including transport and export")


def _selfcheck_by_size(outputs):
    """Request seconds per generated frame for each frame size in the pass, smallest first, request seconds per
    factor, and the pass's mean megapixels per generated frame: how the server's time grows with pixels."""
    sizes, factors = {}, {factor: 0.0 for factor in FACTORS}
    for output in outputs:
        with (output / "predictions.jsonl").open() as stream:
            for record in map(json.loads, stream):
                totals = sizes.setdefault((record["width"], record["height"]), [0, 0.0])
                totals[0] += len(record["target_indices"])
                totals[1] += record["request_seconds"]
                factors[record["interpolation_factor"]] += record["request_seconds"]
    frames = sum(count for count, _ in sizes.values())
    megapixels = sum(w * h * count for (w, h), (count, _) in sizes.items()) / frames / 1e6
    ordered = sorted(sizes.items(), key=lambda item: (item[0][0] * item[0][1], item[0]))
    return (megapixels, ", ".join(f"{w}x{h} ({w * h / 1e6:.2f} MP) {seconds / count:.3f} s"
                                  for (w, h), (count, seconds) in ordered),
            ", ".join(f"{factor}x {seconds:.1f}" for factor, seconds in factors.items()))


def _listed(sizes):
    """Frame sizes as text, smallest first."""
    return ", ".join(f"{w}x{h}" for w, h in sorted(sizes, key=lambda size: (size[0] * size[1], size)))


def selfcheck(check, public_dir, config):
    """A cold generation pass of the evaluation's shape (factors interleaved, frames per interval, the required frame
    rate) built from the public intervals, a few of them at other frame sizes, and sent through the task's launcher,
    the grader's client and its parser; the pass's first interval is the smoke sample. Grading's 2,250 s pass is
    sampled (_selfcheck_plan) and its time projected with the cold start and each shape's first request counted once
    (_selfcheck_sampled). Nothing is scored. The review's free prerequisites run before the hook; no other static gate
    applies: prescreen checks the operator's pins and preflight the review receipt."""
    from frontier_ml.selfcheck import TIGHT_MARGIN

    per, full, whole = _selfcheck_plan(config)
    sampled = per < full
    with tempfile.TemporaryDirectory(prefix="frontier-vfi-selfcheck-") as temporary:
        temporary = Path(temporary)
        parts, native = _selfcheck_pass(public_dir, temporary / "public",
                                        {**config["evaluation_selection"], "per_factor": per}, sampled)
        rows = [row for _, part in parts for row in part]
        outputs = [temporary / f"predictions-{number}" for number in range(1, len(parts) + 1)]
        with check.launch(lambda snapshot, settings: launch(snapshot, settings, str(parts[0][0]))) as server:
            # Grading starts its pass right after /health: no warm-up request comes before the clock.
            check.stage("gates")
            if sampled:
                measured, timing = _selfcheck_sampled(check, server, parts, native, full, whole, config, outputs)
            else:
                measured, timing = _selfcheck_whole(check, server, parts[0], native, whole, config, outputs[0])
            others = {(row["width"], row["height"]) for row in rows} - native
            megapixels, by_size, requests = _selfcheck_by_size(outputs)
            detail = (f"{timing}. {len(rows)} intervals ({per} per factor) were sent one at a time to a cold server, "
                      f"as grading sends its held-out pass, resampled from the public examples at {_listed(native)}"
                      + (f" with one interval per factor at {_listed(others)} early on" if others else "")
                      + f" ({megapixels:.2f} megapixels per generated frame on average). Request seconds by factor: "
                      f"{requests}; per generated frame by size: {by_size}. The requirement is per generated frame "
                      "whatever its size, and held-out frame sizes may differ and may be larger: if your time per "
                      f"frame grows with pixels, keep a margin of several times, not just {TIGHT_MARGIN:.0%}.")
            total = full * sum(factor - 1 for factor in FACTORS)
            check.gate("generation rate", measured, total / whole, higher_is_better=True, unit="frames/s",
                       detail=detail)
