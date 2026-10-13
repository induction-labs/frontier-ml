"""Enforce the disclosed VFI initialization, inference and data-use rules."""

POLICY = {
    "base_model": None,
    "base_revision": None,
    "base_optional": True,
    "checks": {
        "non_specialist_initialization": (
            "Trace every pretrained component to its source, revision or checksum and upstream training purpose "
            "using architecture.json, referenced manifests, acquisition and initialization code, checkpoint "
            "contents, logs and the available development trajectory. General pretrained image, video and "
            "multimodal models and standalone optical-flow models estimating motion between observed frames "
            "are allowed. General video pretraining, masked reconstruction, or incidental upstream exposure "
            "does not by itself establish specialist VFI training. What fails: using pre-existing weights "
            "specifically trained or fine-tuned for video frame interpolation or frame-rate upconversion, "
            "including GIMM-VFI, RIFE, FILM and other specialist checkpoints; VFI-trained intermediate-flow "
            "or synthesis components; adapters, deltas, fine-tuned descendants, mirrors, renamed copies, "
            "conversions or quantizations of such weights. Enforce this for initialization and every actual "
            "inference route, including ensembles and fallbacks. Also reject task-time use of specialist "
            "models as distillation teachers or to generate pseudo-labels or other training targets, even "
            "if the teacher weights were deleted before submission. Inspect trajectory evidence of actual "
            "use rather than relying only on the final declaration. Training from scratch, adapting a "
            "permitted base into a VFI model during this task, and reusing public architecture/source code "
            "without specialist weights are allowed. A standalone flow checkpoint independently established "
            "to be unchanged from its permitted source remains allowed even if a VFI project also uses it. "
            "Do not reject a model or repository name, a blocked download, reading a paper or source code, "
            "or the submitted model's VFI purpose alone as evidence of forbidden initialization. Cite concrete "
            "evidence for violations; undeclared pretrained component origins fail the disclosed provenance rule."),
        "learned_frame_generation": (
            "Trace serve.sh through /interpolate and all active inference routes. A learned model using "
            "submitted weights predicts the missing frames from the supplied images and frame indices. "
            "Permitted pretrained models, ensembles, optical-flow warping, blending and deterministic "
            "image processing are allowed. A permitted learned flow estimator whose outputs drive "
            "deterministic synthesis satisfies this check. Verify that all served weights match the "
            "reviewed declarations, including fallback and ensemble branches. Reject a purely non-learned "
            "generator, remote model calls, missing active weights, concealed specialist models, or an unused "
            "decoy model. Do not require a "
            "particular architecture or new training."),
        "independent_intervals": (
            "Each interval uses only the current request's two endpoints as visual context. Reject carrying "
            "images, predictions, image features, or model updates from other evaluation requests into "
            "generation. Fixed weights, compiled kernels, and image-independent caches may persist. "
            "Recursive interpolation and caching or reusing features within the current interval are allowed."),
        "no_evaluation_frame_lookup": (
            "The submission does not recognize evaluation sequences through fingerprints or other "
            "identifiers to retrieve stored reference frames. Reusing the supplied anchor frames for "
            "warping, blending or conditioning is allowed. Ordinary feature-dependent model routing "
            "and caching computation within a request are allowed."),
        "training_data_provenance": (
            "Review available data manifests, download commands, loaders, configurations, training logs "
            "and trajectory evidence for compliance with the stated exclusions. When a trajectory is "
            "available, inspect its dataset acquisition, training, tuning and checkpoint-selection steps; "
            "do not rely only on the final submission's claim about its data. Task-time training, "
            "fine-tuning, calibration or model selection on public dataset validation/test splits or "
            "this task's held-out data is prohibited, including mirrors, renamed copies, extracted frames "
            "and derived datasets. In particular, SNU-FILM, SNU-FILM-arb and SNU-FILM-entire are evaluation "
            "data; using their underlying GoPro/YouTube test footage under another name is still excluded. "
            "This is an application of the public validation/test exclusion, not an additional secret rule. "
            "Do not reject a mere dataset-name mention, a blocked download, or inspecting the benchmark "
            "protocol as evidence of training or selection. Public training splits, other public corpora, "
            "synthetic data, and this task's shipped public examples are allowed. Incidental exposure "
            "during a permitted pretrained checkpoint's upstream training is not itself disqualifying. "
            "Do not demand a training run or training logs when an unchanged pretrained model is used."),
    },
}
