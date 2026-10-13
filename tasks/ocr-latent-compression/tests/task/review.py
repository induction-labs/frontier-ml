"""Trusted architecture policy for this task. Never shipped to the candidate."""

POLICY = {
    "base_model": None,
    "base_revision": None,
    "checks": {
        "learned_discrete_visual_codes":
            "Trace POST /encode through the supplied inference code. It must run a neural network on the image "
            "pixels and quantize the visual features into a sequence of codes. FSQ, LFQ, learned codebooks, residual "
            "or product quantization, and equivalent discretizations are valid; quantization levels may be fixed "
            "or learned, and multiple codebooks are allowed. The declared codebook_size describes the serialized "
            "integer alphabet used to count the codes against the 16,384-bit (2 KB) budget. The codes are learned "
            "visual codes: no text recognition or OCR engine, no language-model text generation, no transcription "
            "to characters, words or text-tokenizer ids, and no DOM/HTML/markup extraction runs anywhere in /encode, "
            "and the code list is not a text-token sequence or a compressed "
            "transcript. Deterministic image preprocessing and learned entropy models over the codes are allowed. "
            "A visual quantizer whose codes end up correlating with glyphs is allowed; an explicit transcription "
            "step, even if followed by a learned code, is not. Judge operations actually run on the image, not "
            "imported libraries or unused methods; loading a complete vision-language model is allowed when /encode "
            "only executes its visual path.",
        "decoder_generates_text_from_codes_only":
            "POST /decode maps the code list through embeddings, scalar reconstruction or another representation "
            "into a neural decoder (any language decoder, fine-tuned or frozen with adapters) that generates the "
            "returned text. The code list is the only per-image information reaching /decode: no files, caches, "
            "databases, sockets, environment variables, clocks, request order or process state carry information "
            "between requests or between the encode and decode instances, "
            "nothing is precomputed from evaluation images, and no dictionary of stored transcripts or retrieval over "
            "a text corpus produces or completes the output. Decoding is deterministic (greedy or fixed-seed) for a "
            "given code list.",
        "served_weights_are_the_trained_model":
            "Provenance and the actual checkpoint inventory establish that the served encoder, decoder and any "
            "learned quantizer parameters are the trained weights described by the training code, configurations "
            "and logs, that fixed quantization rules match the supplied code and configuration, that the declared "
            "codebook_size matches the serialized integer alphabet, and that any starting checkpoint is declared. "
            "Frozen backbones with trained adapters, merged or fully trained checkpoints, and inference exports "
            "are all valid. Trace serve.sh through both endpoints to the supplied weights and any runtime conversion; there is no decoy "
            "checkpoint or alternate model.",
    },
}
