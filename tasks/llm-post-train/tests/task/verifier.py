"""Post-training of pinned Qwen3-4B-Base: four-benchmark mean behind the review and the checkpoint-layout audit."""
from pathlib import Path
from frontier_ml.taskspec import sibling

from frontier_ml.runtime.server import VerificationFailure
from frontier_ml.scoring import language
from frontier_ml.scoring.architecture import verify_review
from frontier_ml.validation.errors import EvaluationError

REVIEW_CHECKS = list(sibling(__file__, "review.py").POLICY["checks"])
checkpoint = sibling(__file__, "checkpoint_audit.py")
# The self-check's public files: the GPQA CSV and one LiveCodeBench file (language.SELFCHECK_PUBLIC), not 4.2 GB.
SELFCHECK_PUBLIC = language.SELFCHECK_PUBLIC


def preflight(config, data_path):
    review = config.get('architecture_review_path')
    if not review or not Path(review).is_file():
        raise VerificationFailure('Trusted architecture_review_path is required before provisioning')
    language.validate_suite_config(config, data_path)


def verify(server, data_path, config):
    submission = server['snapshot']
    review = verify_review(submission, config['architecture_review_path'], 'llm-post-train', REVIEW_CHECKS)
    inspection = review.get("trusted_audits", {}).get("checkpoint")
    if not inspection or inspection.get("rule") != checkpoint.RULE:
        inspection = checkpoint.audit(submission)
    if not inspection['passed']:
        return {'score': 0.0, 'details': {'failure': inspection.get('failure', 'inventory'), 'inspection': inspection}}
    result = language.evaluate_suite(server['transport'], data_path, config)
    result['details']['gates'] = {'architecture': review, 'checkpoint': inspection}
    return result


def _checkpoint_audit(snapshot):
    """checkpoint.audit, as verify and the review's submission_inspector run it. A layout it rejects, or a checkpoint
    it cannot read, zeroes the grade."""
    try:
        inspection = checkpoint.audit(snapshot)
    except EvaluationError:
        raise
    except (ImportError, MemoryError) as error:  # the verifier image's fault; check.static charges the rest
        raise EvaluationError(f'The checkpoint audit could not run: {type(error).__name__}: {error}') from error
    except Exception as error:  # noqa: BLE001 - everything it reads but the pinned inventory comes from the submission
        raise VerificationFailure(f'The checkpoint audit could not read output/model/: '
                                  f'{type(error).__name__}: {error}') from error
    if not inspection['passed']:
        problems = inspection.get('inventory', {}).get('problems') or [inspection.get('failure', 'inventory')]
        raise VerificationFailure("output/model/ is not Qwen3-4B-Base's tensor layout: " + '; '.join(problems))
    return inspection


def selfcheck(check, public, config):
    """The checkpoint audit grading applies before scoring (output/model/ holds exactly the pinned base's tensors in a
    floating-point dtype), then one copy, one public GPQA request read by the scorers' rules and the suite's
    per-request timeout at its concurrency. There is no speed gate. Whether the weights descend from the pinned base
    is the review's call, from the training evidence and the session; it does not run here."""
    check.static('checkpoint audit', _checkpoint_audit, check.snapshot)
    requests = language.selfcheck_inputs(config, public)
    with check.launch() as server:
        language.selfcheck_requests(check, server, requests, config)
