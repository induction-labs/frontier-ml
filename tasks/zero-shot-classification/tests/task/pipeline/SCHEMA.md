# Classifier contract

Each validation row has `id`, `dataset`, `text`, `labels` and `target`.
`target` indexes the ordered label descriptions. The validator sends only
`{"text": "...", "labels": ["description", "..."]}` to `POST /classify` after
permuting the labels. Return exactly `{"probabilities": [...]}` in request order.
The full distribution must be finite, nonnegative and sum to one within 1e-6.
JSON serialization size does not affect validity. Argmax breaks ties by the first supplied label.

The score is the equal-dataset mean of `macro_F1 * (1 - ECE)`. Macro-F1 includes
the union of gold and predicted classes per dataset, with undefined class F1 set to zero. Top-label ECE uses
15 fixed bins; bin index is `min(14, floor(15 * max(probabilities)))`.
Each bin contributes `count / N * abs(accuracy - mean_confidence)`.

Official evaluation requires **at least 50 documents/second** on one warm H100
with up to **32 concurrent requests**. One request classifies one text against
all supplied labels. The client replenishes requests as they finish; the server
may batch independent requests. The complete 4,400-document batch must finish
within **88 seconds**, including scheduling, serialization, tokenization, all
label processing, complete HTTP responses and response validation. Model loading,
warm-up, final metric calculation and cloud control-plane overhead are excluded.
There is no separate per-request latency gate.
Warm-up shares the 600-second evaluation allowance, reserving 88 seconds for the
complete scored batch. There is no five-second warm-up request limit.

Sixteen unrelated synthetic warm-up requests precede measurement; no scored text
or label inventory is warmed. A request that fails (non-200 status, connection
error or a non-JSON body) is retried up to two more times on the same payload,
after 1 s and then 3 s, while the batch deadline allows; a timeout or a body that
violates this contract is not retried. Retries and backoff count against the
88-second batch. A document that still fails scores as a confidently wrong answer
(probability one on a wrong label); if more than 5% of the 4,400 requests fail,
the run fails (see CONTRACT.md). Public development contains 500 topic-classification
examples only, for local validation: do not train, calibrate or select on them (see
DATA_POLICY.md). It does not represent the entire final evaluation distribution.
The public validator reports throughput and request latencies, but cannot certify
the final hardware or throughput gate.

# Required training evidence

Include `training_manifest.json` with this structure, alongside the shared
`architecture.json` review evidence. Replace all placeholders with real values:

```json
{
  "schema_version": 1,
  "base_models": [{"source": "organization/model", "revision": "<40-character commit>"}],
  "training_datasets": [{"source": "independent corpus or synthetic generator", "revision": "sha256:<64-character content digest>"}],
  "teacher_models": [],
  "reused_artifacts": [],
  "training": {"script": "train.py", "config": "training_config.json", "log": "training.log", "steps": 100},
  "trained_weights": [{"path": "model.safetensors", "sha256": "<64-character file digest>"}]
}
```

Every source entry needs `source` and an immutable `revision`: a 40-character
commit or `sha256:` followed by a 64-character content digest. For a hosted
teacher, record its fixed model/deployment and hash the retained request/output
corpus; explain the mapping in the training configuration. List all synthetic
generators and their teacher dependencies. `teacher_models` and `reused_artifacts`
are required lists even when empty. List all task-trained weight shards or adapters
with exact SHA-256 hashes and document how they are loaded. Serialization format
alone is not a compliance requirement. Evidence paths must stay inside the
submission. Training must actually update classifier parameters; a manifest alone
does not establish compliance.
For training from scratch, use an empty `base_models` list and document the
initialization in the training configuration; all training and provenance checks
still apply.

Source identities, code and logs are reviewed along with the actual checkpoint
inventory and serving path. Do not package training or benchmark corpora in the
inference artifact; retain their provenance and generation/loading code instead.
