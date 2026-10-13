> Completed under the explicitly approved $8 aggregate cap. This replacement
> failed: self/reference repeat 0.944444, two repeated verdict disagreements;
> circle and injection controls both zero. New conservative charge $0.596706;
> aggregate across Luna and Sol $0.6256965. See `sol_calibration_assessment.json`
> and `sol_reference_diagnosis.json`. The text below preserves the original
> preparation and preregistration. Do not rerun or promote this failed scorer.

# Reviewed-data replacement judge calibration

The completed Luna calibration failed. Its 96 native responses, ledger, original
plan, failed results and original data remain preserved. Official scoring is
blocked. The replacement below is prepared only: current SVG authorization is
$1, and the proposed aggregate **$8 cap requires explicit user approval** before
any new native call.

The complete source audit precedes the new split: 299 official reference PNGs,
245 retained captions, 40 minimal objective corrections, 14 exclusions for
ambiguity, then one duplicate removed. Development remains 30 simple icons;
evaluation has 254 examples. `caption_audit.json` and its lock retain original and
revised caption hashes, SVG/render hashes, decisions, reasons and reviewers.
Original-caption fields are whitespace-normalized; raw source files remain pinned. The
three source-stratum reviewers were blinded to previous judge scores. A second
reviewer checked all proposed changes/exclusions, flagged retained cases and 18
seeded retained samples. These are model visual reviews, not human annotations.

`sol_validate.py` is an operator-only candidate scorer. Its substantive four-boolean
rubric is byte-identical to the failed Luna scorer. Only the judge family, protocol
version and explicit text labels immediately before each image differ. It uses
native `gpt-6-sol`, `temperature=0`, `reasoning_effort="none"`,
`max_completion_tokens=1024`, Standard `service_tier="default"`, and `store=false`.
The [Sol model documentation](https://developers.openai.com/api/docs/models/gpt-6-sol)
confirms image input and the reasoning setting; the
[deployment checklist](https://developers.openai.com/api/docs/guides/deployment-checklist)
allows the sampling setting with effort `none`. Strict model identity, one stopped
choice and four exact JSON booleans are required; no fallback is allowed.

The new plan contains 24 fresh examples, eight per stratum, selected by a fixed
source-key hash after excluding every original calibration artwork by canonical
SVG **and rendered-PNG hash**. Old neutral IDs are not used for freshness. Strata
cycle simple/detailed/complex. Each example gets four consecutive probes:
reference self-score, the same red circle, the same visible-text injection, and a
repeated self-score. The fixed 96 probes therefore remain informative if spending
stops early. No judge scores affect sample selection.

The billing reservation uses the entire documented 1,050,000-token context bound,
including both images and message framing. No undocumented vision multiplier or
estimated tokenizer overhead is assumed. The highest Standard long-context
cache-write and output rates are $5 and $15 per million tokens; a 10% regional
margin makes the conservative rates $5.50 and $16.50. Charging every input token at
the cache-write rate is conservative even without cache-write usage details.
[Pricing](https://developers.openai.com/api/docs/pricing),
[cache-write accounting](https://developers.openai.com/api/docs/guides/prompt-caching).

Each call reserves **$5.791896** durably before dispatch. The original 96-call
ledger contributes **$0.0289905**, so the first reservation brings the aggregate
commitment to **$5.8208865**. The proposed $8 lifetime ceiling covers both runs.
All 96 worst-case calls would cost $556.022016 before the old charge, so completion
is **not guaranteed**: confirmed usage releases unused reservation conservatively;
unknown usage, malformed responses, timeouts and crashes retain the full
reservation and stop execution without retries. Any insufficient remaining
headroom also stops before dispatch. Native HTTP is sequential, has no proxy or
redirect, and sends no tools. Both the original and new ledger locks are held;
the original frozen ledger and every original receipt are revalidated. A changed
plan, code, ledger, response or recorded charge fails closed.

Prepare with the existing pinned local CPU renderer image, offline and without
credentials:

```sh
bash tasks/svg-generation/pipeline/run_sol_calibration.sh
```

This creates `work/svg-generation/judge-calibration-v2-sol/{plan.json,dry_run.json,
requests/,images/}` and makes zero API calls. The exact image ID, code hashes,
data/audit hashes, sample IDs, payload hashes, token bounds and proposed budget
are in the dry run. `judge_calibration_sol_dry_run.json` is its tracked summary.
Do not change these frozen inputs after review.

Only after the user approves the raised aggregate cap and the operator reviews
the exact plan hash, the operator may supply `OPENAI_API_KEY` through its own
environment and execute:

```sh
bash tasks/svg-generation/pipeline/run_sol_calibration.sh \
  --execute --approved-plan-sha256 REVIEWED_PLAN_SHA256 \
  --approved-aggregate-cap-usd 8
```

The existing failed official scorer remains disabled during this experiment.
Promotion requires complete 24-example groups, reference and repeat macro scores
at least 0.95, constant-circle at most 0.15, injection exactly zero, at most one of 24 repeated strict verdicts different, and
an absolute repeat macro-score difference no greater than 1/24. All scores must be
finite. These thresholds were frozen before replacement calls in
`judge_readiness_recipe.json`. A smaller pilot or incomplete run cannot enable official
scoring. The exact tested scorer must be promoted with evidence bound to its
code/rubric/render/data hashes. This diagnostic does not establish a strong model
baseline, $50 improvement, or task admission.

The unused Astra draft is archived under
`work/svg-generation/audit-v2/astra-unused/`; it made no calls. Frozen original
Luna scorer sources are under `pipeline/luna_v1/`, with byte-identical original
calibration code, preserving original receipt replay independently of future
scorer promotion.
