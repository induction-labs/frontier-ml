"""Trusted scorer preflight and output quality gates for screenshot generation."""
from __future__ import annotations



def preflight_scorer(config, data_path):
    """Fail before any GPU is rented if the pinned OCR, its GPU runtime or the judge are missing."""
    import screenshot_generation as sg
    from frontier_ml.runtime.server import VerificationFailure
    models = config.get("scorer_models_dir")
    if not models:
        raise VerificationFailure("scorer_models_dir must point at the pinned scorer models")
    try:
        sg.verify_files(models, "ocr")
    except sg.ScorerUnavailable as error:
        raise VerificationFailure(str(error)) from error
    if not callable(config.get("judge")):
        raise VerificationFailure(f"The trusted {sg.JUDGE_MODEL} judge must be configured before provisioning")
    try:
        import rapidocr  # noqa: F401
        import onnxruntime
    except ImportError as error:
        raise VerificationFailure(f"Trusted scorer dependency missing: {error}") from error
    if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
        raise VerificationFailure("The trusted scorer needs a CUDA GPU in the verifier environment")


# ---------------------------------------------------------------- gates (enforced by the verifier)

def latency_gate(latencies):
    import screenshot_generation as sg
    result = sg.latency_summary(latencies)
    result["passed"] = result.pop("meets_requirement")
    return result
