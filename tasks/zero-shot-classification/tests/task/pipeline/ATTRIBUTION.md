# Public development data

The 500 validation examples are a deterministic selection from a single task
bucket: topic classification. They are separate from the final evaluation sources
and are for local validation only: `DATA_POLICY.md` forbids training, fine-tuning,
calibration or model/prompt/hyperparameter selection on them.

- [DBpedia-14](https://huggingface.co/datasets/fancyzhx/dbpedia_14), revision
  `9abd46cf7fc8b4c64290f26993c540b92aa145ac`, test split; dataset card license:
  [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
  The title and content are concatenated; label names are expanded into descriptions.

Duplicate inputs and exact normalized overlaps with evaluation inputs are
excluded. IDs are locally assigned. The license describes the pinned upstream
dataset card, not an assertion that the final evaluation sources share its terms.
