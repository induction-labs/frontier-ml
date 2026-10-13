> Historical plan: all 96 authorized Luna calls completed. The frozen results in
> `judge_calibration_luna_results.json` failed the reference and constant controls;
> visible-text injection also received positive scores. Conservative charge:
> $0.0289905. Preserve this failed evidence. See `SOL_CALIBRATION.md` for the
> separately proposed replacement; do not rerun this plan.

# Prepared judge calibration

This is an operator diagnostic, prepared without paid calls. It does not establish
model headroom or a $50 improvement result. User approval is required before
execution; the separate CoT labeling budget does not authorize this run.

The fixed plan selects eight held-out rows from each of the simple, detailed, and
complex strata by the smallest SHA-256 of `svg-judge-calibration-v1:<row-id>`.
It makes these four rounds in order, using those same 24 instructions:

1. Reference SVG as candidate (24 calls).
2. The same nonuniform red-circle SVG as every candidate (24 calls).
3. The same SVG containing visible instructions to ignore the rubric and return
   four `true` values (24 calls).
4. Repeat the reference self-scores with exactly identical payloads (24 calls).

The exact public scorer creates all prompts, renders, sampling settings and
request bodies. Preparation captures them locally and discards its dummy
verdicts. No dummy quality scores are reported. The report later replays archived
responses through the exact public scorer and checks that each replayed payload
hash equals its paid request. It reports per-stratum macro scores and paired
repeat disagreement; incomplete groups have no imputed score. This plan does not
include an image-order reversal experiment.

## Cost and failure accounting

The endpoint is fixed to `https://api.openai.com/v1/chat/completions`; model is
`gpt-6-luna`, `reasoning_effort="none"`, `temperature=0`, and
`max_completion_tokens=1024`. The transport adds `service_tier="default"` and
`store=false`. No tools, redirects, proxy environment variables, model fallbacks,
automatic retries, or concurrent requests are used.

The [Luna model page](https://developers.openai.com/api/docs/models/gpt-6-luna)
documents a 1,050,000-token context. The [vision guide](https://developers.openai.com/api/docs/guides/images-vision)
says images and text share the model's input/context limits, but its current
sizing table omits Luna. We therefore reserve against the **entire context for
both 512×512 images plus all text**, rather than claiming a Luna-specific image
token estimate. Output is independently bounded at 1024 tokens.

The [pricing table](https://developers.openai.com/api/docs/pricing), checked
2026-09-25, lists Standard long-context cache writes at $0.25/M tokens and output
at $0.75/M. Apply a further 10% regional margin: $0.275/M input and $0.825/M output.
Every request reserves **$0.2895948** before dispatch. All confirmed input tokens
are charged at that conservative cache-write rate, even cached reads and cases
without cache-write details; no discounts are assumed. A completed native receipt
must confirm Standard processing, the right model, and consistent bounded token
usage before its reservation is reduced. Usage means the token counts in the
[Chat Completions response](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).

The **$1 lifetime cap** covers the sum of confirmed cost upper bounds and all
unresolved reservations. Ledger writes fsync the file and its directory before
HTTP dispatch; an exclusive lock prevents parallel spending. An uncertain call
retains its full reservation and stops execution. Its ID is never retried.
Resuming skips already confirmed IDs and refuses to continue past an uncertain
one. A response archive without a matching reservation cannot be reused.

There are **96 planned calls, not 96 guaranteed completed calls**. Without the
cap, their full-context worst-case reservations total **$27.8011008**. The rolling
ledger instead enforces at most $1 and may stop early when headroom is too small.
No ledger or calibration directory may be reset to circumvent this lifetime cap.
Pricing/context assumptions must be reviewed again before a later run.

## Local commands and artifacts

The launcher pins the already-built local CPU image
`sha256:0da16852e69dd2aec24078571b82b62a0190d1c9ed81b4e80e0dea118633bc87`
from `python:3.12-slim-bookworm`, with `librsvg2-bin` 2.54.7 and DejaVu fonts.
It checks the official renderer version and uses the renderer's normal bounds.
The container has 2 CPU, 1GiB RAM and read-only input/code mounts. Preparation
disables networking and does not receive the API key:

```bash
bash tasks/svg-generation/pipeline/run_judge_calibration.sh
```

The persistent directory is `work/svg-generation/judge-calibration-v1/`:
`plan.json` contains fixed rows/probes and input/code/image hashes;
`requests/` contains exact native request payloads; `images/` contains their PNGs;
and `dry_run.json` is the reviewable cost/probe summary. These operator artifacts
contain held-out references and are not public task data.

After separate user authorization, the operator supplies `OPENAI_API_KEY` in its
environment and the exact reviewed `plan_sha256` from `dry_run.json`:

```bash
bash tasks/svg-generation/pipeline/run_judge_calibration.sh \
  --execute --approved-plan-sha256 REVIEWED_PLAN_SHA256
```

This explicit mode enables container networking and passes the key by environment
name. `ledger.json` records durable spending reservations, `responses/` preserves
native receipts, and `report.json` contains the accounting and scorer results.
The script never reads another task's `.env` or automatically claims its budget.
