"""Trusted code-review policy for babylm-replay (shared review engine). Never shipped to the candidate.

The submission is source code only: there is no checkpoint (require_weights False) and no starting
model (candidate-declared base; the only acceptable declaration is from scratch, enforced by a check).
"""

POLICY = {
    "base_model": None,
    "base_revision": None,
    "require_weights": False,
    "checks": {
        "trained_from_scratch": (
            "architecture.json declares base_model \"none\" and base_revision \"from-scratch\", and the code agrees: "
            "the model and tokenizer are randomly initialized and learned inside train.sh; nothing is initialized "
            "from, loaded from, or distilled from any pretrained model, tokenizer or library-provided weights."),
        "no_embedded_external_data": (
            "No file embeds or reconstructs data, statistics or learned parameters that did not come from the corpus "
            "during the replay: no weights, embeddings, vocabularies/merge lists, word or n-gram tables, frequency "
            "lists, lookup tables, text passages, or encodings of such data (base64, hex, numbers, identifier names, "
            "prose/word-choice steganography, procedurally regenerated tables). Establish the provenance and use of "
            "such data from the code. File size, entropy, language, literal length and constant counts alone do not "
            "establish external data use. Hyperparameters, algorithm constants and format strings are fine."),
        "training_uses_only_corpus": (
            "train.sh and the code it runs learn only from files under $CORPUS_DIR (plus randomness and the submitted "
            "code) and write outputs only to $OUTPUT_DIR. Reading any other path for data (image or package files, "
            "other mounts, caches), network use, or loading pretrained models/tokenizers from any library fails."),
        "serving_uses_only_replay_outputs": (
            "serve.sh and the server load model state only from $WEIGHTS_DIR (files written by train.sh) and the "
            "submitted code; reading any other model, data file, cache, the corpus, or the network fails."),
        "causal_normalized_scoring": (
            "/score and /dist are computed by the same trained causal model(s): each log-prob depends only on the "
            "document prefix, /dist is a normalized distribution over the full vocabulary, and there is no "
            "request-history caching, lookup of previously seen text, or special-casing of the audit endpoints."),
        "no_slowrun_lookup": (
            "The participant did not look up NanoGPT Slowrun (https://github.com/qlabs-eng/slowrun), which inspired "
            "this task; the contract forbids it. Judge this from the trajectory: the participant's own tool calls and "
            "commands, including those run on `spin` machines. What fails, whether or not anything found was used: a "
            "web search that names or targets Slowrun; opening, fetching, cloning or downloading the qlabs-eng/slowrun "
            "repository or any file from it, its forks, mirrors or other derivative repositories, or pages about it "
            "(its qlabs.sh pages, DeepWiki, articles), by any means (WebSearch, WebFetch, curl, wget, git, pip, the "
            "GitHub API, a browser or a script). What is fine: reading /task/CONTRACT.md, which names it; results of a "
            "search that did not ask for Slowrun listing such a page that is then not opened; code that shares "
            "techniques with Slowrun (Muon, ensembles, snapshot or weight averaging, U-Net skips, value embeddings, "
            "logit soft-capping and the like); and other public code and papers. Cite the trajectory steps that "
            "establish a violation; with no such step the check passes."),
    },
}
