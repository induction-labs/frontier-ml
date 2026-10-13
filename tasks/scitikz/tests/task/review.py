"""Trusted review: the public benchmark may only be used by final grading."""
POLICY = {
    "base_model": None,
    "base_revision": None,
    "checks": {
        "declared_training": (
            "architecture.json declares the starting open-weights model and immutable revision as base_model "
            "and base_revision. Retained training code, configuration, data manifests and logs establish task-time "
            "training or fine-tuning for image-to-TikZ reconstruction and explain the packaged weights. Unmodified "
            "off-the-shelf weights alone fail. LoRA/adapters are permitted."),
        "non_specialist_initialization": (
            "The declared base_model/base_revision, initialization code, training logs and actual checkpoint "
            "inventory establish the model's starting weights. Any pretrained base model is allowed as long "
            "as it is not a specialist TikZ model, a checkpoint already specialized to write TikZ/LaTeX from "
            "images or text. What fails: starting from JackieLin0123/SciTikZer-8B or any other specialist "
            "TikZ checkpoint (for example the DeTikZify and TikZero families), including mirrors, renamed "
            "copies, conversions, quantizations, fine-tuned descendants, or adapters or deltas requiring such "
            "a backbone. General pretrained language, vision and multimodal models are allowed; adapting an "
            "allowed base into a TikZ model during this task is allowed, including on SciTikZ-230K. Using a "
            "specialist TikZ model as a tool during this task is allowed: generating, labeling, ranking or "
            "filtering training data with SciTikZer, DeTikZify or similar models and training on those outputs "
            "is fine (the SciTikZ-Bench rule still applies to the inputs); only starting the submitted model's "
            "weights from such a checkpoint is prohibited. Trace "
            "checkpoint origins and actual initialization; the submitted model's TikZ purpose, filename or "
            "shared general base architecture alone does not establish forbidden initialization."),
        "no_benchmark_development": (
            "Do not train, test, validate, tune rewards, select models, retrieve answers, or otherwise develop on "
            "any SciTikZ-Bench examples, including mirrors or this task's private split. SciTikZ-230K is explicitly "
            "allowed. Trusted organizer data provenance: all 50 public development examples shipped in "
            "public/validation.jsonl are sampled from SciTikZ-230K revision "
            "3061cc331aa6f71cc7430e0480929d0ccddade67, with exact benchmark image/code overlaps excluded. "
            "They are NOT a public split of SciTikZ-Bench. Training, testing, validation, reward tuning and "
            "model selection on these 50 examples are explicitly allowed; copying them into a candidate's "
            "dev/ directory does not change their source or permitted use. Do not infer SciTikZ-Bench "
            "membership merely from the words public, dev, validation or benchmark. "
            "Review retained download commands, dataset manifests, training configuration, logs, and "
            "evaluation scripts for evidence. Incidental exposure during a declared upstream checkpoint's "
            "pretraining is not itself disqualifying; benchmark use during this task is. Unsupported provenance "
            "or contradictory evidence fails."),
        "neural_editable_reconstruction": (
            "A neural vision-language model conditioned on the input image generates editable TikZ/LaTeX "
            "source. Model-driven rendering, refinement and selection are allowed, as are syntax repair and "
            "formatting. No embedded input raster, pixel-by-pixel drawing, classical image tracing/OCR supplying "
            "figure content, stored-answer retrieval, or image-keyed lookup. Resizing/normalization is allowed."),
        "actual_inference_uses_reviewed_weights": (
            "Trace serve.sh through loading and inference to verify the packaged, trained weights actually "
            "generate the response. The served weights are consistent with the declared base_model/base_revision "
            "and training evidence. No decoy checkpoints, concealed specialist TikZ backbone, or external "
            "model/service calls during serving."),
    },
}
