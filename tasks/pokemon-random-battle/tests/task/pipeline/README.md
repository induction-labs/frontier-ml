# Gen 9 Random Battle against Foul Play

The task builds a bot and evaluates full `gen9randombattle` singles against
**Foul Play only**. Existing bots, including unchanged Foul Play, are allowed;
new training is optional. Raw quality is `p = (wins + 0.5 * draws) / 3000` and
final reward is `max(0, min(1, (p - 0.5) / 0.5))`. There is no
heuristic opponent pool, opponent weighting or Elo conversion. The task card is
[here](../../../docs/tasks/pokemon-random-battle.md); [SCHEMA.md](SCHEMA.md) describes
observations and the HTTP endpoint.

## Folder layout

The task root holds the task definition, validator, verifier, review policy,
battle runtime and opponent installation files. `verifier.Dockerfile` builds the
trusted evaluation environment.

This `pipeline/` directory contains payload preparation, set-prior generation,
mechanics export, calibration scripts and their documentation. `example_submission/`
contains the small training/server example. The payload builder still copies the
schema, notices and example to their existing public paths, including `public/starter/`.
Commands below run from the repository root.

The current 3,000-game payload is published at
`inductionlabs/frontier-ml-tasks@5ca7a86e0cda2dd9a4630b2b5f4c2aeb63769f20` under
`pokemon-random-battle/`. `task.yaml` pins this revision and the final matchup-file
SHA-256. Hosted grading downloads these stored teams; no fresh final teams are
generated. Use the preparation instructions only when changing the payload.

## Pinned opponent

- Pokémon Showdown: `ea822f8863407f7a16995a9db3519f8e250237c1`.
- Foul Play: `6c467c081e862fb321adb405355beb41aba8e226`, unmodified upstream source.
- poke-engine: `0.0.48`, built with the upstream Terastallization feature.
- Upstream default search: 100 ms per sampled state, one process and one thread.
  Random Battle samples four states at 50 ms early, or two at 100 ms later.
- Sixteen independent trusted battle workers; final verifier allocation is 16 CPUs.
- Set priors: 100,000 teams from the pinned public generator, seed
  `[179,421,991,2027]`. No evaluation teams are used, and no live set data is fetched.

The adapter uses upstream battle parsing, hidden-state sampling and search. It
receives only Foul Play's own request and player-visible history. It never sees
the candidate's unrevealed team, private request, pending action or simulator RNG.
Both policies get exact own HP and rounded opponent HP. No legal-choice fallback
or replacement heuristic is permitted if Foul Play fails. Singleton menus (such
as recharge or the only possible forced switch) are executed directly because
there is no policy choice; this also avoids an upstream recharge-tera suggestion
that Showdown does not permit.

Revival Blessing also arrives as a switch-shaped request, but only fainted targets
are legal. Upstream Foul Play records `reviving` without passing it to poke-engine,
so ordinary forced-switch search can select a living reserve and abort evaluation.
The adapter compares the legal revived-team afterstates using the pinned native
MCTS and its public hidden-set samples. It retains the active Pokémon and restores
one target to half its previously observed own maximum HP. The normal search budget
is divided among targets, not multiplied. This is an explicit adapter extension;
normal decisions still use upstream `find_best_move`, with no heuristic fallback.

Runtime protocol 3 adds two narrow parser/conversion corrections: truncate to
upstream's first four moves before computing native indices, and skip move-order
speed inference after selecting a switch (as upstream already does for Choice
Scarf inference). These prevent native fifth-slot panics and switch-as-move
lookups. Ordinary inference and search remain unchanged; crashes fail evaluation.

Runtime protocol 4 additionally fixes Eiscue's repeated form changes within one
request interval. An intermediate own form can be absent from the final request;
the adapter defers that stats refresh only for a unique matching player/nickname
and base species. Team positions are not stable across switches. Public form
events still apply, and exact own stats refresh before search. Other mismatches
still fail; the upstream archive and opponent search policy remain unchanged.

Public validation generates 32 fresh random team/chance matchups on each invocation
and records its diagnostic root seed. Runtime protocol 5 freezes only the final
3,000 games as 1,500 paired matchups with opposite candidate team assignments/seats.
`private/test.jsonl` stores both complete packed Showdown teams, their order,
battle chance seed and opponent sampling seed for each game. The verifier loads
these exact teams on every run, checks the pinned file digest, and rejects the
old seed-only schedule. No final team generation or fresh root seed remains.
The Rust search is time-based and stochastic: fixed inputs reproduce matchups, not
necessarily identical decisions or scores. Historical ladder peaks do not measure
this particular pinned version, set prior, hardware and search budget.
The final evaluation budgets the candidate's own server time: the trusted
H100-local move timings summed over all `/act` requests must stay within 18,000
seconds (Modal transport, opponent search and simulation excluded); a 12-hour
wall-clock limit only stops a stalled evaluation. The endpoint still has one second
per complete response. Public validation retains its one-hour overall allowance.

## Evaluation scheduling

The evaluator keeps up to 32 games active and fills each finished slot from the
shuffled schedule immediately, instead of waiting for all games in a group to end.
Each slot keeps its Foul Play worker affinity; completing a game clears that slot's
parser and random state without disturbing another game on the same worker.
All scheduled team/seat pairs still contribute exactly once.

For each batch, Foul Play search and the candidate HTTP request run concurrently
from the same pre-turn player views. Both must succeed before the simulator applies
either action. There is still only one candidate request in flight, with the same
one-second H100-local response deadline, a 60-second per-request cap including
transport (one retry for transport faults only), the 18,000-second server-time
budget and the 12-hour wall limit. Failures stop the run and tear down the
opponent processes.

Result timing now reports setup, opponent search, candidate request (including
transport), simulator calls, opponent reset, and combined decision wall time, plus
mean/max batch size and total observations. Opponent and candidate times overlap;
do not add them to estimate wall time. These are trusted driver measurements, not
candidate-reported timings. The separate H100-local receipt remains authoritative
for the one-second gate.

Scheduling originally required protocol 2; the current payload requires protocol
6 (the opponent adapters above, stored final teams, and an illegal Foul Play
choice forfeiting the game to the candidate). Both splits include the simulator
refill operation and selective opponent reset. The validator rejects an
older payload explicitly before starting battles. Rebuild with the preparation
command below and publish the reviewed payload with `frontier upload
pokemon-random-battle` before launching a hosted run; the official dataset does not
auto-publish local changes. The 54.61-minute pilot predates these optimizations;
no full H100 speedup has been measured yet. CPU allocation increases from four to
16; search is time-based, so hardware contention can affect exact MCTS decisions.

Local verification of this change: 132 task, packaging and agent-surface tests
passed, with one unrelated missing-media skip. Coverage includes 640 scheduled
games scored exactly once in the scheduler fixture, concurrent decisions and
failure cleanup, real battles through reused slots on the same Foul Play worker,
32-game public HTTP validation, hidden-information boundaries and existing move
deadline tests. Payload hashes, source/adapter agreement, task build, `frontier
check`, and the stdlib-only validator smoke check also passed. Upstream source
archives, set priors and matchup schedules are byte-identical to the prior payload.
The full repository run reported 2,157 passed, 73 skipped and 18 failures; every
failure was already present in the saved pre-change merge test run, and none was
in the Pokémon task. No paid infrastructure or full H100 evaluation was launched.

## Policy eligibility and reward

The task now permits unchanged or modified Foul Play, public pretrained policies,
heuristic search, learned policies, ensembles and local fallback policies. The
review checks `implementation_provenance`, `actual_served_policy` and
`permitted_information`; it does not require training or a checkpoint. This
replaces the original pilot's from-scratch/learned-decision requirements. Serving
source and reused code/model provenance remain necessary, with training evidence
when applicable. No private-state access, cached evaluation answers or remote
decision services are allowed.

Final grading applies the same shared normalization used by other tasks, with
`floor: 0.5` and `ceiling: 1.0`. The raw mean battle points remain in `raw_score`;
the final `score` is improvement over 50%. Public validation retains its raw
diagnostic score. The 50% floor is user-selected rather than a measured calibration;
there is no special-case reward for recognizing a Foul Play submission. Draws keep
half credit and capped games remain losses. The original Astra pilot's 237/640
raw result would earn zero on this scale. Historical run files are not rewritten.

Verification of the eligibility/normalization change: 242 focused checks passed
with one unrelated missing-media skip. Tests cover weightless-bot review plumbing
with injected reviewer verdicts, continued rejection of information-policy
violations, reward boundaries, draws and preservation of raw outcomes. The full
suite reported 2,167 passed, 73 skipped and the same 18 previously recorded
failures. Task build/check, payload manifest and bare-Python validation also pass.
The updated public schema is rebuilt locally; the published data pin still needs
updating together with the scheduling runtime before a hosted run. No paid review
or GPU evaluation was performed for this change.

## Prepare the payload

Requires Git, Node.js >=18, npm, Python >=3.12 and the repository's uv environment.
Run from the repository root, using fresh output directories:

```bash
git clone https://github.com/smogon/pokemon-showdown.git /tmp/pokemon-showdown-random-battle
git -C /tmp/pokemon-showdown-random-battle checkout ea822f8863407f7a16995a9db3519f8e250237c1
(
  cd /tmp/pokemon-showdown-random-battle
  npm ci --omit=dev --ignore-scripts
  node build
)
git clone https://github.com/pmariglia/foul-play.git /tmp/foul-play-random-battle
git -C /tmp/foul-play-random-battle checkout 6c467c081e862fb321adb405355beb41aba8e226
node tasks/pokemon-random-battle/pipeline/foul_play_priors.js \
  /tmp/pokemon-showdown-random-battle > /tmp/foul-play-priors.json
# Reuse the previously audited engine artifacts from the current public payload.
# To create a new organizer build, see build_poke_engine.py; inspect its receipt
# and explicitly update the digest audit before admitting different build bytes.
uv run python tasks/pokemon-random-battle/pipeline/pokemon_random_battle_data.py \
  --showdown /tmp/pokemon-showdown-random-battle \
  --foul-play /tmp/foul-play-random-battle --priors /tmp/foul-play-priors.json \
  --engine-build data/pokemon-random-battle/public/engine \
  --test-seed-file data/pokemon-random-battle.private-seed \
  --output /tmp/pokemon-random-battle-payload
python .agents/skills/frontier-task/scripts/write_manifest.py /tmp/pokemon-random-battle-payload --check
```

Preparation refuses to overwrite an existing directory. Generate into staging
when updating, inspect its manifest and then replace the local payload. The runtime
lock covers the simulator, complete Foul Play source archive, prior and adapters.
The payload includes the audited native engine wheel, corresponding source and
build receipt, with 32 public scheduling rows and 3,000 fixed private game rows.
`--test-seed-file` is created once with restricted permissions and reused on every
rebuild. Keep this file outside the payload and retain it to reproduce the frozen
teams. Do not delete it or use a different seed file for routine adapter updates.
The build audits distinct held-out teams. Validation draws independently at runtime;
there is no fixed validation team set. After inspecting a new payload, set `verifier.config.evaluation_sha256`
in `task.yaml` to the manifest's `evaluation.schedule_sha256`. A deliberate new
matchup set requires a new dataset revision and hash, not an evaluation-time flag.

Only the verifier receives `private/test.jsonl` through the harness. The HF dataset
is public, so `private/` is not an access restriction; compliance review prohibits
acquiring or using held-out teams. Fixed teams are not precomputed answers, and
timed Foul Play search still varies. See [NOTICE.md](NOTICE.md) for licenses.

## Install local opponent dependencies

Install Python 3.12 and Rust >=1.89 first. Then:

```bash
sh tasks/pokemon-random-battle/setup_foul_play.sh /tmp/frontier-foul-play
export FOUL_PLAY_PYTHON=/tmp/frontier-foul-play/bin/python
```

The validator remains stdlib-only and can run with `python -S`. It launches the
separate dependency environment for the opponent. The custom trusted verifier
Dockerfile installs this environment at `/opt/frontier-foul-play`; candidate
serving does not need Foul Play. Missing dependencies fail before provisioning.
Neither the simulator nor opponent needs a Showdown account, server or network.

## Local calibration and example endpoint

```bash
uv run python tasks/pokemon-random-battle/pipeline/baselines.py \
  --data data/pokemon-random-battle/public/validation.jsonl \
  --output runs/pokemon-random-battle-foul-play/validation-baselines.json
```

This CLI tests candidate heuristics against Foul Play. They are development
teachers/probes, never evaluation opponents. A heuristic or existing bot may also
be submitted through the required HTTP endpoint and scored under the same rules.
The default command runs five policies; it now takes minutes instead of seconds.
Use `--policy type` for a single calibration. An operator calibration on the fixed 3,000 games:

```bash
uv run python tasks/pokemon-random-battle/pipeline/baselines.py --policy type \
  --data data/pokemon-random-battle/private/test.jsonl \
  --output runs/pokemon-random-battle-foul-play/calibration-3000.json
```

The diagnostic baseline argument is unavailable on the candidate endpoint. The
final verifier ignores diagnostic seed, baseline and sample-count overrides.

The first public calibration on 2026-09-27 completed 32 games: the type-aware
candidate won 1 and lost 31 (3.125% mean points), in about 46 seconds locally,
with no invalid actions or capped games. This small sample is not a ladder rating.
Old reports in `runs/pokemon-random-battle-local-baselines/` and the old starter's
56.25% result were against the superseded heuristic pool, not Foul Play. Its payload
is preserved at `runs/pokemon-random-battle-heuristic-archive/data/`. Older VGC
reports and `runs/pokemon-vgc-archive/data/` belong to the superseded doubles task.

The small example still demonstrates imitation, safetensors export and serving:

```bash
uv sync --extra data
uv run python tasks/pokemon-random-battle/pipeline/example_submission/train.py \
  --task-code tasks/pokemon-random-battle \
  --data data/pokemon-random-battle/public/validation.jsonl \
  --output runs/pokemon-random-battle-starter-foul-play --epochs 10
cp tasks/pokemon-random-battle/pipeline/example_submission/server.py runs/pokemon-random-battle-starter-foul-play/server.py
HOST=127.0.0.1 PORT=8123 uv run python runs/pokemon-random-battle-starter-foul-play/server.py
```

In another terminal with `FOUL_PLAY_PYTHON` set:

```bash
uv run python tasks/pokemon-random-battle/validate.py \
  --server-url http://127.0.0.1:8123 \
  --data data/pokemon-random-battle/public/validation.jsonl \
  --output runs/pokemon-random-battle-starter-foul-play/http-validation.json
```

A final submission needs offline `serve.sh`, serving source/dependencies,
code/model provenance and `architecture.json`. Training source and logs are needed
only if training is performed. A weightless bot may declare `base_model="none"`
and `base_revision="none"`, with reused code repositories/revisions listed
separately. Shared review checks the actual served implementation and that only
permitted information enters serving.

## Task checks and running

```bash
uv run frontier check pokemon-random-battle
uv run pytest -q tests/test_pokemon_random_battle.py tests/test_agent_surface.py tests/test_build.py
```

The dataset is published at `443ae3bd1f9ed833d2d8c539124aff13e77d8ccd`.
`frontier check` verifies the local payload against that pin, renders the task,
and checks the remote public/private files and credentials. Both the verifier
and CUDA inference images have also built successfully on Modal.

To run Astra with a $50 development wallet and Modal environments:

```bash
uv run --env-file .env frontier run pokemon-random-battle \
  -a codex -m openai/gpt-6-astra --env modal --budget 50 \
  -- --ak reasoning_effort=high
```

`frontier run` uses this task's pinned official dataset and starts the task using
the normal viewer. Publishing local payload changes requires an explicit
`frontier upload pokemon-random-battle`. Supply the OpenAI, Spin, Modal and review credentials through
`.env` or the inherited environment. The $50 cap applies to the Spin development
wallet; agent API and final Modal grading charges are separate. Final inference
uses a Modal H100 by default.

The first Astra pilot, `pokemon-random-battle-astra-20260927`, completed training
and all 640 held-out games: 237 wins, zero draws or caps, **37.03%**. The candidate
ran on a verified Modal H100; review passed all three original policy checks.
Its original data pin was `206e1e1ba403a9831cc5dd2026662e7f604e48aa`. A development
match exposed unsupported Revival Blessing choices in the opponent adapter;
the corrected evaluation pin above uses native-search afterstates within the same
search budget. The original payload and the change record are preserved in the
run directory. Final startup took 66.92 seconds and the 640-game evaluation took
3,276.36 seconds (54.61 minutes), with 1,595 requests and a 4.81-second maximum
complete request. The Spin wallet closed at $29.72 of $50; agent API usage was
$69.32 separately, with Modal and review API charges additional.

The first grader needed exact-content audits for oversized public mechanics
data. The second regrade completed but Harbor lost the final artifact download;
results were recovered directly from the verifier and the real transport error
was retained. See `runs/pokemon-random-battle-astra-20260927/REPORT.md` and
`runs/pokemon-random-battle-astra-20260927-regrade2/artifact-recovery.json`.
The immutable candidate was unchanged across repairs. All 103 focused
review/grading/Pokémon tests passed. All resources were stopped and the complete
submission and individual outcomes were preserved.
Results in `runs/<job>/<trial>/verifier/result.json` include individual outcomes,
counts, paired uncertainty, timing, matchup seed and pinned opponent settings.

### Move response deadline (added after the first pilot)

Each complete `/act` response must arrive within **1 second**, including batches of
up to 32 moves. `pokemon_move_bridge.py` is installed as a root-owned client on
the H100 after server readiness. It enforces an absolute deadline over sending
the observation and receiving the entire body, including trickled responses.
Modal upload/exec/download overhead is subject to the separate 60-second
per-request ceiling (one retry for transport faults) and the 12-hour wall limit;
it does not count toward the 18,000-second server-time budget. The final result reports
trusted median, p95 and maximum HTTP latency; missing or inconsistent timing
records fail closed. Public validation uses the same response limit from the
validation client's location, so run it next to the development endpoint.

The first pilot's saved endpoint log covers 1,591 of 1,595 requests. Its reported
single-move median was 5.65 ms (356 single-move calls, maximum 7.27 ms), batch p95
was 20.03 ms, and maximum was 400.48 ms. These are candidate-reported diagnostic
timings, not trusted certification. Its trusted maximum including Modal transport
was 4.81 seconds. That result used the older rule and has not been regraded under
the new one-second limit.

Verification: 44 Pokémon task tests and three build/agent-surface checks passed,
including real delayed and trickled HTTP responses, a missing-timer rejection,
invalid timing receipts and public validation without site packages. `frontier
check` renders and passes credential/source checks. Updated public documentation
and its manifest are local; `frontier upload pokemon-random-battle` publishes them and updates the data pin.
No paid evaluation was launched for this deadline change.

### Late moves forfeit their games (2026-10-02)

Until 2026-10-02 one response over 1 second scored the whole run zero. In the
2026-10-02 sweep that zeroed five of eight grades, three of them after passing
self-checks whose slowest of 124 responses was 0.72-0.83 s: a one-strike rule
over about 3,600 requests is decided by a tail that 124 samples cannot see. Per
the task owner, a late response now forfeits every game in its batch instead:
the trusted move client records it (marked late, with its time, which still
counts toward the 18,000 s), the evaluator discards the answer, sends Foul
Play's -1 for those games so the pinned simulator ends them (no runtime change),
records each as a loss with reason `move_timeout`, and plays on. Results report
`move_timeout_forfeits`, `timing.late_requests` and the first 100 `late_moves`.
Public validation forfeits the same way at `--move-seconds`. The self-check plays
its slice the same way and reports late responses as a warning, not a failure.

## Verification record

The revised focused selection passed **95 tests**, with one unrelated missing-clip
skip. It includes real Foul Play games, a complete 32-game HTTP validation with
`python -S`, forced recharge and Revival Blessing handling, hidden-state boundaries, paired schedules,
mean scoring, missing-dependency failure, deadlines, runtime hashes and fixture
image rendering. The dependency setup script built poke-engine successfully in a
fresh Python 3.12 environment. The manifest verified all 22 files and both splits.
Logs are in `runs/pokemon-random-battle-foul-play/`.

The full suite reported **936 passed, 54 skipped and 19 existing failures**:
missing language/PII/Maple fixtures (8), macOS address-space limits (1), OCR
malformed-response tests (8), reasoning sampling (1) and screenshot batch budget
(1). No Pokémon task test failed. See `full-suite.log` in the run directory.
