# Operator preparation and calibration

**Current status:** the v4 semantic scorer is calibrated and promoted. `judge_readiness.json`
is bound to the current `validate.py` (2026-09-30: failed requests stay in their stratum), `svg_render.py`,
`verifier.py`, shared judge client, renderer 2.54.7 and frozen data by the second v4 run (`-r2`, 2026-09-30);
see [V4_CALIBRATION.md](V4_CALIBRATION.md), `judge_calibration_results.json` (the current report),
`judge_wire_replay.json` and `v4_calibration_assessment.json`. The first v4 run's evidence is kept as
`*_v4_r1_*` files. The passing v3 readiness is preserved in
`judge_readiness_v3.json` with its report in `judge_calibration_sol_low_results.json`. Any change to
a bound file requires a new run: `run_v4_calibration.sh` (prepare, `--execute`, `--replay`) and
`calibrate_v4.py --promote`. The v4 scorer also tolerates harmless answer formatting (any fence,
declaration or prose around the outermost `<svg>` document), accepts the public presentation
attribute allowlist, length units, paint fallbacks, nested `svg` and CDATA, judges concurrently in a
pool of eight, and retries transient judge transport errors; all 284 prepared references
re-canonicalize byte-identically and render to their audited hashes under that code.

`source_lock.json` pins the official repository revision, three caption/reference
files and Apache-2.0 license. The reviewed version 2 payload has 30 simple-icon
examples for development and 254 evaluation examples: 67 simple, 92 detailed, 95
complex. A complete visual audit of all 299 official renders retained 245 captions,
corrected 40 objective contradictions and excluded 14 ambiguous rows; one duplicate
instruction was then removed. `caption_audit.json` records each original/revised
caption and hash, canonical SVG/render hashes, decision, reason and reviewers.
`caption_audit_lock.json` freezes that audit. Original-caption fields/hashes refer
to the whitespace-normalized source instruction; raw source-file hashes preserve
the original bytes. Corrections/exclusions have a second
visual review; 18 hash-selected retained rows were cross-reviewed too. These are
model visual reviews, not human annotations. No judge result selects a correction
or exclusion. The failed original data remain archived locally.

Free reproduction, in a CPU environment with the exact official renderer:

```sh
python tasks/svg-generation/svg_data.py fetch --source-dir work/svg-generation/research
python tasks/svg-generation/svg_data.py prepare --source-dir work/svg-generation/research --output-dir EMPTY_OUTPUT_DIRECTORY
python tasks/svg-generation/svg_data.py verify --output-dir EMPTY_OUTPUT_DIRECTORY
python .agents/skills/frontier-task/scripts/write_manifest.py EMPTY_OUTPUT_DIRECTORY --check
```

Preparation requires `rsvg-convert version 2.54.7` and DejaVu fonts. It verifies all
source hashes, complete audit coverage, original/corrected captions, and exact
reviewed render hashes before applying the version 2 deterministic grouped split.
Every reviewed source must still render; drift fails rather than silently dropping
rows. Official source renders use the pinned local CPU image recorded in the
audit. `prepared_lock.json` fixes the current private bytes/count/hash. Public
examples have neutral IDs and no source metadata; attribution remains in the
manifest and full license. Public-source prompts remain searchable.

`official_render_results.json` is the historical version 1 check: 298 references and
blank/solid floors passed under 2.54.7. Version2 re-rendered all 299 source images
under the same bounds before auditing, and its prepared references are hash-bound
to those images. `sanity_v2_results.json` records historical public blank/solid floors;
these probes perform no model inference or paid judging.

The [original Luna calibration](JUDGE_CALIBRATION.md) completed 96 authorized calls
and failed: reference self/repeat 0.861111, constant-circle 0.236111, text-only
injection 0.361111. The confirmed conservative charge is $0.0289905. All 96 native
receipts and the durable ledger remain in
`work/svg-generation/judge-calibration-v1/`; tracked
`judge_calibration_luna_results.json` preserves the failed results. Original scorer
sources are frozen under `luna_v1/`. It was not promoted.

The [replacement Sol plan](SOL_CALIBRATION.md) prepares 96 fresh balanced,
interleaved probes against the audited corpus, with explicit image labels and the
same substantive rubric. The user approved its aggregate $8 lifetime cap including original spend.
All 96 calls completed: self/repeat 0.944444, two repeated verdict disagreements,
circle/injection 0. New conservative charge $0.596706; total $0.6256965. The replacement
failed its preregistered checks. No scorer was promoted and no data/rubric was
retuned. See `sol_calibration_assessment.json` and `sol_reference_diagnosis.json`. Its full-context
reservation avoids undocumented image-token assumptions. Do not promote a
replacement scorer before complete 24/group controls pass and evidence is reviewed.

Before release, measure the strongest permitted open autoregressive checkpoint
and a bounded $50 improvement experiment. The task card states admission criteria;
judge controls establish neither model headroom nor attainable gains. Underlying
artwork rights still need a publication audit. Data remain local and unpublished;
no trained-model GPU baseline has been performed.

The final [Sol low-reasoning calibration](SOL_LOW_CALIBRATION.md) completed all 96
calls on a third fresh balanced subset, preserving the reviewed corpus, substantive
rubric and readiness thresholds. Reference self/repeat scored 1.0; circle/injection
scored 0; all 24 repeated verdicts agreed. New conservative charge was $0.612139,
for $1.2378355 across the three runs under the approved aggregate $8 cap.

The historical v3 public scorer was a byte-for-byte copy of `sol_low_validate.py`.
The current v4 scorer intentionally differs in its uniform-rendering treatment, rubric, answer
extraction, allowlist and concurrency. The trusted verifier adds only Standard service and
`store=false` plus a bounded transient-error retry of the identical payload. `replay_calibrated_wire.py`
runs without networking and proves all 96 complete canonical JSON payloads match
the paid request bytes, then replays archived responses to reproduce every score.
The shared HTTP client may serialize equivalent JSON with different whitespace;
no message or parameter may differ. `judge_wire_replay.json` binds this evidence
to the public scorer, verifier, shared judge client, renderer, data and native
report. Official scoring requires both this proof and the complete passing quality report
for the current scorer. `judge_calibration_results.json` is the v4 report; uniquely named Luna, Sol-none and Sol-low reports preserve history.
Changes to any bound component require new reviewed evidence. Public judging is
unofficial; rendering-only diagnostics require no judge.
