# Vendored scoring provenance

`redaction_rscore.py` is copied without semantic modifications from this repository's
`frontier_ml/budgeted/redaction_rscore.py`, as frozen for PII v8. This is an independent
reproduction of RedactionBench's paper metric, **not** the official implementation.
Source: https://arxiv.org/html/2606.18782v1#S3.SS3. Its `ASSUMPTIONS` constant records
edge-case conventions. The RedactionBench data is CC-BY-4.0 at
https://huggingface.co/datasets/RedactionBench/RedactionBench, revision
`d45e9cec89bc49c69355e252fec29cc0229982f6`.

The public validation client uses a task-scoped adaptation in `spans.py`. The span
algorithm is unchanged, with source-discovery metadata omitted from its public report.
