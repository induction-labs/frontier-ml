"""Ternary PTQ of pinned Qwen3.5-4B: four-benchmark mean behind architecture and weight gates."""
from pathlib import Path
from frontier_ml.taskspec import sibling

from frontier_ml.runtime.server import VerificationFailure
from frontier_ml.scoring import language
from frontier_ml.scoring.architecture import TERNARY_EXCLUSION_RULE, inspect_ternary, verify_review
from frontier_ml.validation.errors import EvaluationError

REVIEW_CHECKS = list(sibling(__file__, "review.py").POLICY["checks"])
# The self-check's public files: the GPQA CSV and one LiveCodeBench file (language.SELFCHECK_PUBLIC), not 4.2 GB.
SELFCHECK_PUBLIC = language.SELFCHECK_PUBLIC


def preflight(config, data_path):
    review = config.get('architecture_review_path')
    if not review or not Path(review).is_file():
        raise VerificationFailure('Trusted architecture_review_path is required before provisioning')
    language.validate_suite_config(config, data_path)


def verify(server, data_path, config):
    submission = server['snapshot']
    review = verify_review(submission, config['architecture_review_path'], 'ternary-ptq', REVIEW_CHECKS)
    inspection = review.get("trusted_audits", {}).get("ternary")
    # Earlier audit rules may have counted vision or MTP weights toward a passing fraction.
    if not inspection or inspection.get("exclusion_rule") != TERNARY_EXCLUSION_RULE:
        inspection = inspect_ternary(submission)
    if not inspection['passed']:
        return {'score': 0.0, 'details': {'failure': 'ternary_fraction', 'inspection': inspection}}
    result = language.evaluate_suite(server['transport'], data_path, config)
    result['details']['gates'] = {'architecture': review, 'weights': inspection}
    return result


def _weight_audit(snapshot):
    """inspect_ternary, the audit verify and the review's submission_inspector run. A manifest or tensor it rejects or
    cannot read zeroes the grade too, so it is the submission's failure here."""
    try:
        return inspect_ternary(snapshot)
    except (EvaluationError, ImportError, MemoryError):
        raise
    except Exception as error:  # noqa: BLE001 - everything it reads but the pinned inventory comes from the submission
        raise VerificationFailure(f'The weight audit rejected quantization_manifest.json or its tensors: '
                                  f'{type(error).__name__}: {error}') from error


def selfcheck(check, public, config):
    """The weight audit grading applies before scoring (inspect_ternary: the ternary fraction and the stored-parameter
    cap), then one copy, one public GPQA request read by the scorers' rules and the suite's per-request timeout at its
    concurrency. There is no speed gate. The review's checks (ternary execution, provenance) do not run here."""
    check.stage('static')
    try:
        inspection = _weight_audit(check.snapshot)
    except VerificationFailure as error:
        check.fail('weight audit', str(error), kind='static')
    fraction, threshold = inspection['ternary_fraction'], inspection['threshold_exclusive']
    audited = inspection['base_parameters'] + inspection['scale_parameters']
    stored, limit = inspection['stored_parameters'], inspection['parameter_limit']
    for name, passed, measured, required, detail in (
            ('ternary fraction', fraction > threshold, fraction, threshold,
             f"{inspection['ternary_parameters']:,} of {audited:,} audited parameters (text-model weights outside the "
             f"input embedding and output head, and their scales) are ternary codes: {fraction:.6f}, which must "
             f"exceed {threshold:g}"),
            ('stored parameters', inspection['parameter_cap_passed'], stored, limit,
             f"{stored:,} stored (codes, scales, retained floats and {inspection['unreferenced_parameters']:,} in "
             f"tensors outside the manifest, uint8/int8 counting four each) against {inspection['parameter_cap']:g} x "
             f"{inspection['base_parameters']:,} original parameters")):
        if not passed:
            check.fail(name, detail, kind='static', measured=measured, required=required)
        check.record(name, True, kind='static', measured=measured, required=required, detail=detail)
    # Grading's verdict itself, the field verify reads, so a clause the two records above miss cannot pass here.
    if not inspection['passed']:
        check.fail('weight audit', 'inspect_ternary rejected the checkpoint for a reason the records above do not '
                   'show; grading scores it zero', kind='static')
    check.record('weight audit', True, kind='static', detail='inspect_ternary passed, as verify requires before '
                 'scoring')
    requests = language.selfcheck_inputs(config, public)
    with check.launch() as server:
        language.selfcheck_requests(check, server, requests, config)
