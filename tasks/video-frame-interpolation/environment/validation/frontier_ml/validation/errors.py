class VerificationFailure(RuntimeError):
    """An invalid response or an exceeded evaluation deadline."""


class EvaluationError(VerificationFailure):
    """The verifier's own infrastructure failed; no fault of the submission was established.

    grade() records these as ``evaluation_error`` (reward still 0) so an operator can tell a broken
    grading image or transport from a submission that did not meet the contract.
    """
