> Completed and passed all preregistered controls: reference self/repeat 1.0,
> circle and rendered-text injection 0, repeated verdict disagreements 0/24.
> New conservative charge $0.612139; aggregate across all three runs $1.2378355,
> under the approved $8 cap. The exact scorer is promoted only with the
> verifier-bound offline wire replay and passing readiness receipt. The text
> below preserves the reviewed plan and accounting assumptions. Do not rerun it.

# Sol with low reasoning: third independent calibration

Two completed runs remain failed and unchanged. Luna with no reasoning scored
0.861111 on reference self/repeat, 0.236111 on constant circles and 0.361111 on
rendered-text injection, with two repeated verdict disagreements. Sol with no
reasoning and explicit image labels scored 0.944444 on self/repeat and zero on
both negative controls, again with two disagreements. Conservative confirmed
charges are $0.0289905 and $0.596706: **$0.6256965 aggregate**.

This third plan changes a substantive model setting, not the dataset or rubric:
`gpt-6-sol` with `reasoning_effort="low"` and `max_completion_tokens=2048`.
Temperature and other sampling/log-probability controls are omitted. Sol supports
low reasoning and image input; the official compatibility rules require omitting
those controls with reasoning enabled. The completion cap includes reasoning,
visible output and other generated tokens. No tools are sent.
[Sol model](https://developers.openai.com/api/docs/models/gpt-6-sol),
[deployment checklist](https://developers.openai.com/api/docs/guides/deployment-checklist),
[output-token accounting](https://developers.openai.com/api/docs/guides/token-counting).

The four-boolean rubric and explicit labels are byte-identical to the Sol-none
scorer. The only scorer differences are protocol ID and the reasoning/output
settings above. The reviewed 254-example corpus remains unchanged. In particular,
easy-063 correctly depicts **four** columns: source geometry and a pixel scan
establish that count, correcting an erroneous visual inspection during diagnosis.
No caption or reference was altered in response to either Sol score.

A third fresh 24-row set is selected deterministically, eight per stratum, excluding
all 48 artwork identities from both prior calibrations by canonical-SVG and rendered
PNG hashes. The fixed source-key ranking seed is `svg-judge-calibration-v3-sol-low-audited`.
Each row has four interleaved probes: reference, constant circle, identical visible
text injection, repeated reference. All 96 requests, images, code hashes, source
hashes, prior-ledger bindings and thresholds are frozen in the plan. Different
fresh subsets mean cross-run scores are diagnostic, not a paired causal estimate
of the reasoning-setting effect.

The existing approved **$8 aggregate SVG-testing cap** applies. Each new call
reserves the full documented 1,050,000-input-token context plus 2048 output tokens,
using the highest Standard long-context cache-write/output rates with a 10%
regional margin: **$5.808792 per call**. Together with prior confirmed charges,
the first reserved aggregate is **$6.4344885**. No undocumented image multiplier or
message-framing estimate is used. Remaining aggregate budget is $7.3743035, and
completion of all 96 probes is not guaranteed. Confirmed actual usage conservatively
releases unused reservation; any uncertainty keeps the entire reservation and
stops with no retries. An insufficient remaining reservation also stops before
dispatch. All input tokens are charged at the high cache-write rate even when cache
write details are absent. [Pricing](https://developers.openai.com/api/docs/pricing).

Both original ledgers and all 192 prior native receipts are revalidated. Their
locks are held together with the new ledger lock throughout dispatch. Durable
fsync reservations precede each native request. The runner is sequential, fixes
the native endpoint, uses Standard service, disables redirects/proxies, and never
retries. A malformed/truncated response may consume the full output cap; it
retains the full reservation and stops.

Preregistered readiness rules remain unchanged: complete 24/group controls, finite
scores, reference/self repeat ≥0.95, circle ≤0.15, injection=0, at most one of 24
strict verdict disagreements, and macro-score difference ≤1/24. Both failed runs
remain preserved. A passing third diagnostic still requires root evidence review
before promoting the exact tested scorer; it does not establish a trained-model
baseline or $50 task headroom.

Prepare offline without credentials:

```sh
bash tasks/svg-generation/pipeline/run_sol_low_calibration.sh
```

Review `judge_calibration_sol_low_dry_run.json` and `sol_low_preparation_receipt.json`.
No new calls may be launched until root reviews the exact plan. After that review,
the operator may supply `OPENAI_API_KEY` in its environment and run:

```sh
bash tasks/svg-generation/pipeline/run_sol_low_calibration.sh \
  --execute --approved-plan-sha256 REVIEWED_PLAN_SHA256 \
  --approved-aggregate-cap-usd 8
```

Astra and the token-counting endpoint are not used. A conservative full-context
Astra reservation would exceed the approved cap. Official SVG scoring stays
disabled while this third plan is only prepared.
