"""Trusted review policy for speculative-drafting (review: true since 2026-09-30). Never shipped to the candidate.

No drafter checkpoint is prescribed and none is required (base_optional): a trained draft model, an n-gram or
prompt-lookup scheme over the request's own tokens, or any combination are all admissible. The review exists to
catch what the replay cannot: drafts copied from the target's held-out continuations, and training or tuning on
the held-out conversations.
"""

POLICY = {
    "base_model": None,
    "base_revision": None,
    "base_optional": True,
    "checks": {
        'training_data_provenance': (
            'Training code, data manifests, download commands and retained logs establish where the data used to '
            'train, fine-tune, calibrate or select the drafter came from, consistent with the contract: training '
            'splits of public datasets, other public corpora, continuations the candidate generated itself with the '
            'target model on other prompts and the shipped public examples in /task/public are fine. What fails: any '
            'use of the held-out set or of any revision of the dataset repository pinned in /task/setup.json '
            '(inductionlabs/frontier-ml-tasks, folder speculative-drafting, whose private folder is publicly '
            'readable) beyond those shipped examples; training, fine-tuning, calibrating or selecting on the '
            'validation or test split of any public dataset or benchmark; or unsupported provenance. Look for '
            'validation/test file names, benchmark loaders, split arguments, dataset-repository downloads and '
            'token-id dumps or JSONL matching the held-out conversations in the training, calibration and '
            'checkpoint-selection code, and in any data the submission bundles. Incidental exposure during a declared '
            "base model's upstream pretraining is not itself disqualifying. A submission with no trained component "
            'states so; absent provenance for a trained component fails.'
        ),
        'non_specialist_initialization': (
            'The declared base_model/base_revision (or the statement that the drafter has no trained component), '
            "initialization code, training logs and the actual checkpoint inventory establish the drafter's starting weights. "
            'Any pretrained base model is allowed as long as it is not a specialist speculative-drafting model: a checkpoint '
            'already trained to draft tokens for a target model. What fails: starting from a released draft head or draft '
            'model, for example EAGLE, EAGLE-2 or EAGLE-3 heads (SafeAILab/EAGLE and the Qwen3-8B EAGLE-3 heads published by '
            "SpecForge, AngelSlim and others), Medusa, Hydra, HASS, GLIDE or FastDraft drafters, a target model's own "
            'multi-token-prediction or draft heads, or any other checkpoint trained as a speculative drafter for any target, '
            'including mirrors, renamed copies, conversions, quantizations, fine-tuned descendants, or adapters or deltas '
            'requiring such a backbone. General pretrained language models of any size (for example the Qwen3 family) are '
            'allowed, as is training from scratch; adapting an allowed base into a drafter during this task is allowed, '
            'including on continuations the candidate generated with the target. n-gram and prompt-lookup schemes have no '
            'starting weights and pass this check. Using a specialist drafter as a tool during this task (generating or '
            "filtering training data) is allowed; only initializing the submitted drafter's weights from one is prohibited. "
            "Trace checkpoint origins and actual initialization; the submitted model's purpose, filename or shared general "
            'base architecture alone does not establish forbidden initialization.'
        ),
        'no_continuation_lookup': (
            "Trace the POST /draft handler: each draft is computed from the request's tokens and the submitted files "
            "by the drafter (model inference, n-gram or prompt lookup over the request's own tokens, or any "
            "combination; a prefix cache across requests is fine). What fails: a stored copy of the target's held-out "
            'continuations or anything keyed on them (token-prefix tables, hashes, ids or request order mapping to '
            'continuation tokens); the held-out conversations or continuations in any form (the pinned dataset '
            'repository, any of its revisions, or copies) in the submission or fetched at run time; serving the '
            'target model itself; or any network access. Inspect every bundled data file, not only the code.'
        ),
        'actual_inference_path': (
            'Trace serve.sh through model loading (if any) and the /draft handler to establish that the reviewed code '
            'and weights actually produce the drafts, with no unused decoy checkpoint, hidden alternate path or '
            'remote service.'
        ),
    },
}
