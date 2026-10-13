> **Second run (`-r2`), 2026-09-30:** the failed_prediction change to `validate.py` (review F2: a failed request
> stays in its stratum at 0) unbound the readiness below, so the same design was re-run on a fifth fresh balanced
> 24-row subset with the first v4 run added to the frozen priors. All 96 native calls confirmed (`gpt-6-sol`,
> Standard tier); reference self/repeat 0.986111/0.986111 (one attribute verdict on one detailed row, identical on
> repeat), constant circle 0, injection 0, 0/24 repeated verdict disagreements, macro delta 0; prompt tokens
> 1,069–1,104, completion tokens 20–98. Conservative charge upper bound **$0.613789** under the run's $5 cap,
> **$2.4707595** across all five runs. Plan `d30da8d0…`, report `ae82b131…`, wire replay 96/96 matched offline;
> promoted with `--promote` (scorer `11b3eb24…`). The first run's evidence is preserved as
> `judge_calibration_v4_r1_results.json`, `judge_wire_replay_v4_r1.json`, `judge_readiness_v4_r1.json` and
> `v4_r1_calibration_assessment.json`; the second run's dry run is `judge_calibration_v4_r2_dry_run.json` and its
> artifacts live in `work/svg-generation/judge-calibration-v4-semantic-r2/`.
>
> First run, completed 2026-09-25 and passed every preregistered check: reference self/repeat 1.0/1.0,
> constant circle 0, rendered-text injection 0, 0/24 repeated verdict disagreements, macro delta 0.
> 96/96 native receipts confirmed (`gpt-6-sol`, Standard tier); prompt tokens 1,067–1,095,
> completion tokens 20–82; median latency 1.87 s. Conservative charge upper bound **$0.619135**
> under this run's $5 cap; **$1.8569705** across all four runs. Plan `992a9e69…`, report
> `a9a4c97c…`, wire replay 96/96 payloads and receipts matched offline. The exact scorer bytes
> were promoted with `--promote`; see `v4_calibration_assessment.json` and `judge_readiness.json`.

# v4 semantic scorer: fourth independent calibration

The v4 scorer (`svg-generation-v4-semantic`) changed the rubric (uniform renderings are judged
against the instruction instead of auto-zeroed), the renderer code (tolerant answer extraction,
the public presentation-attribute allowlist, length units, paint fallbacks, nested `svg`, CDATA)
and the trusted verifier wrapper (bounded transient-error retry) and made judge calls concurrent
(pool of eight, sample-ordered records). The passing v3 readiness was bound to the old scorer
bytes and rubric `cd6e…`; the repairs note forbids hash substitution, so this run calibrates the
current bytes: rubric `ba2f…`, `validate.py`, `svg_render.py`, `verifier.py`,
`frontier_ml/runtime/judge.py`, `rsvg-convert 2.54.7` and the frozen 254-row evaluation payload.

`calibrate_v4.py` loads the real `verifier.py` and routes every probe through
`verifier.calibrated_judge`, so the archived request bytes are exactly what production sends
(`gpt-6-sol`, `reasoning_effort="low"`, `max_completion_tokens=2048`, `service_tier="default"`,
`store=false`, explicit REFERENCE/CANDIDATE labels, high-detail 512x512 PNGs). The design is the
same 96-probe plan as the three earlier runs, interleaved per row: reference self-score, the same
red circle, the same visible-text prompt injection, and a byte-identical repeated self-score, on a
fourth fresh balanced subset of 24 rows (8 per stratum) chosen by the fixed source-key ranking seed
`svg-judge-calibration-v4-semantic` after excluding all 72 earlier calibration artworks by
canonical-SVG and rendered-PNG hash. The ledger revalidates all three prior ledgers, plans and
288 receipts and holds their locks during dispatch.

Budget: this run has its own **$5 cap**. Each call reserves an 8,192-input-token envelope (eight
times the 1,058–1,099 prompt tokens measured across the 288 earlier receipts for the same image
inputs) plus the 2,048-token output cap at the highest Standard long-context cache-write/output
rates with a 10% regional margin: **$0.078848 per call**. A receipt above the envelope, malformed,
non-Standard or with the wrong model is uncertain, keeps its full reservation and stops the run
with no retry. Confirmed usage releases headroom conservatively; every input token is charged at
the cache-write rate. Completion of all 96 calls is not guaranteed by the cap.

Readiness thresholds are unchanged from `judge_readiness_recipe.json` (frozen before the first Sol
run): complete 24/group controls, finite bounded scores, reference self and repeat macro means
≥0.95, constant circle ≤0.15, injection exactly 0, at most one of 24 repeated strict verdicts
different and a repeat macro-score difference ≤1/24. The driver evaluates them in `assess` and
`--promote` refuses to write a passing `judge_readiness.json` unless the report passed and the
offline wire replay (`replay_calibrated_wire.py`, `--network none`) matched all 96 canonical JSON
payloads and archived responses through the verifier wrapper by payload digest (so the
concurrent pool cannot reorder receipts).

```sh
bash tasks/svg-generation/pipeline/run_v4_calibration.sh                 # prepare offline
bash tasks/svg-generation/pipeline/run_v4_calibration.sh --execute --approved-plan-sha256 SHA --approved-cap-usd 5
bash tasks/svg-generation/pipeline/run_v4_calibration.sh --replay        # offline wire replay
uv run python tasks/svg-generation/pipeline/calibrate_v4.py --promote --data-dir data/svg-generation --prior-dir work/svg-generation --work-dir work/svg-generation/judge-calibration-v4-semantic
```

Artifacts live in `work/svg-generation/judge-calibration-v4-semantic/` (plan, requests, images,
ledger, native receipts, report, wire replay). Tracked copies: `judge_calibration_results.json`
(the report), `judge_wire_replay.json`, `judge_readiness.json`, `v4_calibration_assessment.json`
and `judge_calibration_v4_dry_run.json`. Judge calibration only: no trained-model baseline or
headroom is established, and a 24-row control set is not a broad robustness evaluation.
