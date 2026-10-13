"""Trusted review policy for trained causal end-of-turn probability streams."""
POLICY = {
    "base_model": None,
    "base_revision": None,
    "checks": {
        "learned_causal_model": (
            "End-of-turn probabilities come from a trained model applied to the supplied audio. "
            "The submitted training code and logs support the claimed model or fine-tuning. "
            "Pretrained encoders, VAD, ASR, and arbitrary inference libraries may be used."),
        "causal_frame_probabilities": (
            "The probability reported for frame i may depend on either speaker channel only through the end of "
            "that frame, (i + 1) x hop_seconds. Although complete files are supplied for efficient evaluation, "
            "future samples must not affect earlier frames. Lookahead, ASR context, bidirectional encoding, "
            "end-of-file padding and retrospective smoothing must be folded into which later frame a probability "
            "is reported at; probabilities must not be backdated. All model state and preprocessing are causal "
            "at each frame. One tolerance: the fixed-length anti-aliasing filter of a standard sample-rate "
            "conversion (a windowed-sinc, polyphase or FIR resampler such as torchaudio.functional.resample, "
            "scipy.signal.resample_poly or soxr) may look at most 5 ms past the frame end, compensated or not; "
            "that is not a violation. Whole-signal resampling (FFT-based, e.g. scipy.signal.resample over the "
            "file) and every other kind of lookahead are not covered by this tolerance."),
        "no_lookup_of_evaluation_audio": (
            "No audio fingerprints, memorized transcripts, or stored event timelines recognize evaluation "
            "recordings and retrieve answers. Predictions generalize from the received audio."),
        "training_provenance": (
            ('Training code, configuration, data manifests, download commands and retained logs establish where the '
             "model's training, fine-tuning, calibration and model-selection data came from, consistent with the shipped "
             'weights and the contract: other public conversational audio and the training splits of public datasets are '
             'fine; the 8 public conversations shipped in /task/public (validation.jsonl and audio/) may be used for '
             'training, fine-tuning, calibration and checkpoint selection as well as evaluation. What fails: any use of '
             "this benchmark's own data, namely any use at all of the TurnBench dev release beyond the shipped 8 "
             '(mundo-ai/turn-benchmark-dev, any revision, mirror, copy or derivative, audio or labels; this includes the '
             'ungated derivative JSALT2026-Conv-AI-Simulator/turnbench-dev-no-backchannel, which carries all 38 '
             'conversations with both channels, the annotator tracks and conversation_id), whether for training, '
             'fine-tuning, calibration, selection or retrieval. The release holds exactly 38 labelled conversations and '
             'the shipped 8 plus the held-out 30 are that whole release: the shipped 8 are TurnBench conversation ids 20, '
             '21, 22, 94, 95, 122, 124, 125; the held-out 30 are 38, 40, 41, 42, 43, 44, 68, 71, 105, 106, 107, 112, 113, '
             '114, 126, 127, 128, 129, 138, 139, 140, 141, 157, 158, 159, 160, 169, 170, 171, 172. A download of the '
             'release or of a derivative, a training, calibration or selection script that reads TurnBench conversations '
             'beyond the shipped 8, or any of the 30 held-out ids used as TurnBench conversation identifiers (file or '
             'directory names, id lists, manifests, caches, download commands, logs) or their labels appearing in the '
             'submission fails. Also failing: any other copy of the held-out conversations or their labels; training, '
             'fine-tuning, calibrating or selecting on the validation or test split of any public dataset or benchmark; '
             'evaluation labels used as training data; or provenance the evidence does not support (weights whose training'
             ' data cannot be traced). Look for validation/test file names, benchmark loaders, split arguments and '
             'download commands in the training, calibration and checkpoint-selection code. Incidental exposure during a '
             "declared pretrained encoder's upstream pretraining is not itself disqualifying. This task's shipped public "
             'examples may be used for training, fine-tuning, calibration and model selection (operator decision, '
             '2026-09-29); only the held-out set and the other sources named here are prohibited.')),
    },
}
