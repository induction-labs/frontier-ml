"""Public, dependency-free definition of the fixed quality score."""
import math

from frontier_ml.validation.errors import VerificationFailure

PROTOCOL = "vfi-lpips-alex01-flolpips-sequence-v1"
SCORING_PROTOCOL = "vfi-lpips-flolpips-two-anchor-v2"
FACTORS = (4, 8, 16)
METRICS = ("lpips", "flolpips")
NOT_CONFIGURED = (
    "Official evaluation is not configured: freeze the data, metric weights, both measured anchors "
    "and performance gate first. Use --interface-only with --predictions-dir to export predictions."
)


def require_evaluation():
    raise VerificationFailure(NOT_CONFIGURED)


def number(value, name, *, positive=False):
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise VerificationFailure(f"{name} must be {'positive' if positive else 'nonnegative'} and finite")
    return value


def aggregate(records):
    """Frames/pairs -> interval is done by the metric; intervals -> factor here."""
    groups = {str(factor): [] for factor in FACTORS}
    seen = set()
    for row in records:
        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in seen:
            raise VerificationFailure("Metric records need unique, nonempty interval IDs")
        seen.add(row["id"])
        factor = row.get("interpolation_factor")
        if type(factor) is not int or factor not in FACTORS:
            raise VerificationFailure("Unsupported interpolation factor")
        for name in (*METRICS, "psnr"):
            number(row.get(name), name)
        groups[str(factor)].append(row)
    if any(not group for group in groups.values()):
        raise VerificationFailure("Evaluation requires intervals at every factor: 4, 8 and 16")
    return {factor: {"samples": len(group), **{
        name: math.fsum(row[name] for row in group) / len(group) for name in (*METRICS, "psnr")
    }} for factor, group in groups.items()}


def validate_baseline(baseline):
    if not isinstance(baseline, dict) or set(baseline) != {str(f) for f in FACTORS}:
        raise VerificationFailure("Baseline must contain exactly factors 4, 8 and 16")
    for factor, row in baseline.items():
        if not isinstance(row, dict) or set(row) != set(METRICS):
            raise VerificationFailure("Each baseline factor needs LPIPS and FloLPIPS")
        for name in METRICS:
            number(row[name], f"baseline {factor}/{name}", positive=True)
    return baseline


def validate_anchors(baseline, target):
    """Both anchors are measured errors, with a strictly better high-score target."""
    validate_baseline(baseline)
    validate_baseline(target)
    for factor in baseline:
        for name in METRICS:
            if target[factor][name] >= baseline[factor][name]:
                raise VerificationFailure(f"Target must be strictly better than baseline for {factor}/{name}")
    return baseline, target


def score(records, baseline, target, config=None):
    """Equal-weight progress from the measured zero anchor to the measured one anchor."""
    if config is not None and config.get("scoring_protocol") != SCORING_PROTOCOL:
        require_evaluation()
    validate_anchors(baseline, target)
    by_factor = aggregate(records)
    improvements = {factor: {
        name: max(0.0, min(1.0, (baseline[factor][name] - row[name]) /
                          (baseline[factor][name] - target[factor][name]))) for name in METRICS
    } for factor, row in by_factor.items()}
    return {"score": math.fsum(v for row in improvements.values() for v in row.values()) / 6,
            "samples": sum(row["samples"] for row in by_factor.values()),
            "details": {"scoring_protocol": SCORING_PROTOCOL, "metric_protocol": PROTOCOL,
                        "by_factor": by_factor, "baseline": baseline, "target": target,
                        "normalized_improvement": improvements,
                        "metric_weights": {name: 0.5 for name in METRICS},
                        "factor_weights": {str(f): 1 / 3 for f in FACTORS}, "psnr_cap_db": 100.0}}
