"""Trusted review policy for lean-simplification (review: true since 2026-09-30). Never shipped to the candidate.

No starting checkpoint is prescribed and none is required (base_optional): a search procedure over Lean tactics,
a trained model, or both are all admissible. The review exists to catch what the kernel checker cannot: proofs
looked up rather than produced, and training or tuning on the held-out theorems.
"""

POLICY = {
    "base_model": None,
    "base_revision": None,
    "base_optional": True,
    "checks": {
        'training_data_provenance': (
            'Training code, data manifests, download commands and retained logs establish where the data used to '
            'train, fine-tune, calibrate or select the prover, its search heuristics or any ranking model came from, '
            'consistent with the contract: training splits of public datasets, other public corpora, proofs the '
            'candidate generated itself with Lean and the shipped public examples in /task/public are fine. What '
            'fails: any use of the held-out set or of any revision of the dataset repository pinned in '
            '/task/setup.json (inductionlabs/frontier-ml-tasks, folder lean-simplification, whose private folder is '
            'publicly readable) beyond those shipped examples; training, fine-tuning, calibrating or selecting on the '
            'validation or test split of any public dataset or benchmark; or unsupported provenance. Look for '
            'validation/test file names, benchmark loaders, split arguments, dataset-repository downloads and proof '
            'corpora keyed by theorem statement in the training, calibration and checkpoint-selection code, and in '
            "any data the submission bundles. Incidental exposure during a declared base model's upstream pretraining "
            'is not itself disqualifying. A submission with no trained component states so; absent provenance for a '
            'trained component fails.'
        ),
        'no_proof_lookup': (
            "Trace the POST /simplify handler: each returned proof is produced at request time from the request's "
            'statement and proof by the submitted system (model inference, tactic search, calls to Lean, or any '
            'combination). What fails: a table, index, retrieval store, cache or embedding lookup that maps theorem '
            'statements (or their hashes, normalized forms or embeddings) to prepared proofs; the held-out theorems '
            'or their proofs in any form (the pinned dataset repository, any of its revisions, or copies such as '
            'extracted JSONL, pickles or SQLite files) in the submission or fetched at run time; or any network '
            'access. Retrieving related lemmas, premises or worked examples to condition a model is fine; retrieving '
            'a proof of the requested theorem itself is not. Inspect every bundled data file, not only the code.'
        ),
        'actual_inference_path': (
            'Trace serve.sh through model loading, Lean invocation and the /simplify handler to establish that the '
            'reviewed code and weights (if any) actually produce the proofs, with no unused decoy checkpoint, hidden '
            'alternate path, remote service or state carried across requests that changes answers.'
        ),
    },
}
