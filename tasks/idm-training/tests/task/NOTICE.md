# Vendored scoring provenance

`idm_upstream_actions.py` and `idm_upstream_keylog.py` are the pinned upstream scorer from
`p-doom/inverse-dynamics-model` at `a22f421b575ae74c56ed25db02a30ba150a2864b`, under the
Apache-2.0 license (text in `frontier_ml/licenses/Apache-2.0.txt`).

The public validation client uses task-scoped adaptations in `actions.py` and `keylog.py`.
They retain scoring and keylog conversion; free-form model-response parsing is removed
because the task API requires canonical JSON. They ship to the agent with the
license text as `LICENSE-apache-2.0.txt`.

The metric follows the released `actions.py:score_actions` and the count-based
micro aggregation in `infer.py`, rather than independently reimplementing the
Hugging Face dataset card's leaderboard description. Both pins still match the
upstream repository heads checked on 2026-09-24.

Upstream behavior intentionally retained:

- Key presses and clicks require exact details within -2/+3 frames; mouse moves
  require only the same frame; scrolling requires the same frame and direction.
- The released evaluator does not consume the dataset's visibility annotations,
  manual overrides, or precomputed gesture labels, despite the dataset card
  describing visibility-filtered leaderboard results.
- Mouse displacement and scroll magnitude are separate diagnostics, not terms
  in the headline F1. The validation result retains those diagnostics per clip.
- Empty/empty clips receive F1 one locally, but the aggregate is computed from
  summed counts and is zero if there are no actions anywhere.

Deliberate deviation from upstream (2026-09-30):

- `actions.py:_extract_key_click_gt` drops the MouseScroll events that the keylog
  parser (`keylog.py:parse_keylog_events`, unchanged) emits at the first frame of each
  run of scrolling frames. Upstream scores them alongside `extract_mouse_gt`'s per-frame
  scroll labels, so every scroll run is labelled twice: on the 43 held-out clips that is
  51 extra labels on 970, a model reporting each scroll once tops out at F1 0.973, and
  reporting the first scroll of each run twice scored 0.996. The per-frame labels are the
  ones in SCHEMA.md's format and the ones the mouse diagnostics use, so they are kept and
  each scroll now counts once; reporting the true labels scores 1.0 and a duplicate is a
  false positive. `tests/test_idm_training.py` pins this as the only function that may
  differ from the vendored reference.

The task's server returns canonical full-clip JSON directly. Upstream's optional
windowing, model-response text parsing, and window-output deduplication belong
to its inference adapter and are not imposed on this server interface.
