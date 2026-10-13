# Gen 9 Random Battle decision protocol

The simulator uses pinned `gen9randombattle`: Generation 9 singles with a randomly
generated team of six per player. The upstream generator supplies species, sets,
team order and balancing levels. There is no team preview, team selection or open
team sheet. One Pokémon is active at a time, and each side can Terastallize once.
The simulator and pinned Foul Play opponent run locally; no Showdown server or account is required.

Start your offline server with `serve.sh`; use the standard `/health` contract.
Final serving uses one offline H100 80GB, 8 CPU cores and 80 GiB RAM, with
300 seconds to become healthy. The opponent runs on a separate 16-CPU verifier.
First install the opponent dependencies in your development environment using
`sh /task/public/setup_foul_play.sh /opt/frontier-foul-play` (Python 3.12;
Linux x86-64 uses the supplied wheel, other platforms need Rust >=1.89). Then evaluate with
`python /task/validate.py --server-url http://127.0.0.1:8000`.
For a different install location, set `FOUL_PLAY_PYTHON` to its `bin/python`.
The official trusted verifier already contains these dependencies.

If you reuse poke-engine in your submission, copy the exact wheel from
`/task/public/engine/` into `/workspace/output/wheels/` and install it at startup
with `python -m pip install --no-index --target /tmp/bot-deps wheels/*.whl` (also
vendor all other required wheels). Add `/tmp/bot-deps` to `PYTHONPATH` before
starting the server. This CPython 3.12 Linux x86-64 wheel has Terastallization
enabled and is audited by its exact SHA-256. The source archive and build receipt
are alongside it. PyPI distributes this engine version only as source; a wheel
you compile yourself will fail the binary audit, even if its filename matches.
This organizer artifact is an explicit task-specific addition to the shared
contract's published-wheel policy. Modified Python bot source remains permitted
and reviewed normally.

`POST /act` receives exactly `{"observations": [...]}` with 1–32 observations
and a maximum JSON request size of 8 MiB. Reply exactly `{"actions": [0, ...]}`
with HTTP 200 and a body of at most 16 MiB. Each entry is an integer index into
the corresponding action menu. Booleans, out-of-range indexes, missing entries,
extra keys, other status codes and non-JSON bodies are invalid and end the
evaluation; the grader never re-sends a request because of your response (see
the time limits below).

Each observation includes:

- `format`: `gen9randombattle`.
- `player`: `p1` or `p2`, your seat in this battle.
- `turn`: simulator turn number. The first decision is turn one, after the leads
  enter battle; there is no preview decision.
- `request`: the actual private Showdown request delivered to your player.
  `active` lists the active Pokémon's move information; `forceSwitch` marks a
  required replacement. `side.pokemon` includes your six Pokémon's stats, items,
  abilities, moves, exact HP, Tera types and active flags. The opponent's request
  and unrevealed team members or sets are never included.
- `public_log`: complete player-visible Showdown protocol history (the field name
  is retained for compatibility). Own HP is exact; opponent HP is rounded. Only revealed species, moves, items, abilities and Tera types appear;
  Illusion preserves the simulator's public disguise. No omniscient log, input
  log, seed, timestamp, pending action or opponent-policy label is sent.
- `legal_actions`: single orders such as `move 1`, `move 2 terastallize` or
  `switch 3`. No joint commands or explicit target locations are needed in singles.
  Revival Blessing replacement requests select a fainted teammate. Both players
  choose before the turn resolves.
- `action_features`: one 20-value numeric row per offered action, derived solely
  from this observation and public mechanics. These convenience features are not
  a value oracle. They omit abilities, weather, stat boosts, Tera and many move
  interactions. A policy may ignore them and learn its own representation.

The menu follows the player's current knowledge. An unrevealed trap or move
restriction may make an offered choice unavailable. As in Showdown, the simulator
then delivers an updated request to that player. The opponent's accepted choice
stays locked, and the endpoint chooses from the updated menu. This is a new game
request, not a retried HTTP call. Other invalid choices fail evaluation.

Feature order: `bias, damage, base_power, accuracy, priority, physical, special,
status, heal, setup, switch, tera, own_hp, foe_hp, super_effective, resisted,
switch_hp, switch_speed, switch_bulk, hazard`. Damage is a rough base-power ×
accuracy × STAB × type-effectiveness estimate based on visible species. Healing
scales with missing HP; setup/hazard features only flag the first two battle turns.
These crude features leave substantial room for a richer encoder.

Keep the endpoint stateless. Batches mix unrelated battles and decision types.
Use the complete visible history and current private request to reconstruct state;
never assume batch order is a game ID.

Public `validation.jsonl` contains paired schedules (`id`, `pair`, `seat`,
`opponent`), not stored teams. Every public validation invocation draws fresh
random teams and battle seeds for its 32 games; its report records the seed for
reproducing a diagnostic run. Every validation and final
game is against Foul Play, pinned at `6c467c081e862fb321adb405355beb41aba8e226`,
with poke-engine 0.0.48 and upstream default search settings: 100 ms per sampled
state, one process and one thread per search. Upstream chooses two or four sampled
states per decision. Sixteen independent battle workers run on the trusted verifier,
which reserves 16 CPUs. Up to 32 games stay active, with finished games replaced
from the schedule. Both policies decide concurrently from their pre-turn views;
the simulator applies their actions only after both complete. Requests remain
stateless and contain at most 32 observations.
The pinned engine lacks a Revival Blessing target-choice interface. For this
request only, the adapter compares legal revived-team afterstates with the same
native search, sharing the normal search budget across targets. The active Pokémon
stays in place; maximum HP comes from that player's previously observed requests.
Normal move/switch decisions retain upstream Foul Play's selection policy.
The adapter truncates native move lists to upstream's first four entries before
computing the last-used index. Upstream otherwise performs those operations in
the opposite order and can pass an invalid fifth-slot index to the native engine
during inference. It also skips speed inference when the player's last selection
was a switch, matching upstream's Choice Scarf inference guard. Switches provide
no comparable move-speed evidence; ordinary move-order inference is unchanged.
Runtime protocol 4 also handles intermediate own forms absent from the final
request snapshot, such as Eiscue regaining and losing Ice Face in one turn. It
defers that request refresh only when a unique player/nickname identifies the
same species in another form. Public form events still apply, and the normal
request refresh restores exact own stats before search. Unrelated or ambiguous
identities still fail; opponent stats never come from the own-team request.
No alternative action or opponent replaces a failed search;
an organizer worker failure is reported as an evaluation infrastructure error.
Search is stochastic and time-based, so outcomes can vary on repeated matchups.
Final grading uses the same 3,000 games for every submission: 1,500 stored matchups,
each played with the candidate on both sides. Full teams, their ordering, battle
chance seeds and opponent sampling seeds are frozen during data preparation.
Runtime protocol 5 loads those teams directly without calling the team generator.
Runtime protocol 6, the current `protocol` in `public/runtime.json` (the verifier
requires exactly this protocol), changes how an illegal Foul Play choice is scored:
if the opponent's search selects a command that is not in its legal menu (an
unknown move or switch, or a rejected combination such as Terastallizing during
Struggle), Foul Play forfeits that game at once and it counts as a candidate win
(`reason: opponent_illegal` in the result, tallied as `opponent_forfeits`).
Nothing substitutes for the illegal choice; a crash of the organizer's worker
itself remains an evaluation infrastructure error, not a candidate result.
The verifier checks the evaluation file's pinned SHA-256 before serving and scoring.
Schedule metadata, seeds and unrevealed opponent sets are never sent to the endpoint.
Foul Play's timed search can still produce different trajectories and outcomes.
Full outcomes determine the score: win 1, natural draw 0.5, loss 0; a game that
Foul Play forfeits through an illegal choice is a candidate win. Raw mean battle
points are `p = (wins + 0.5 * draws) / 3000`; every game has equal weight. Final
reward is `max(0, min(1, (p - 0.5) / 0.5))`: 50% or below earns 0, 75% earns 0.5
and 100% earns 1. The 50% floor is fixed rather than recalibrated per run. Public
validation reports raw battle points; final grading preserves those as
`raw_score` and applies the reward normalization. Games capped at 200 turns or
512 decision rounds score zero for the candidate.

Time limits and retries, as the trusted verifier enforces them:

- **1 second per `/act` response.** A root-owned HTTP client inside the H100
  sandbox (`pokemon_move_bridge.py`, installed after `/health` succeeds) times
  each request from opening the connection to `127.0.0.1:8000` and sending the
  observations through reading the complete response body, with an alarm at
  1.0 s and a check of the measured interval. That interval includes your
  feature processing, search and inference for the whole batch of up to 32
  decisions. The candidate cannot supply its own latency measurement. A
  response over 1 second forfeits every game in its batch: each counts as a
  loss, the late answer is discarded, it is not retried, and the evaluation
  continues with the other games. The grader does not wait for the late
  answer, so a server still working on it when the next batch arrives can make
  that batch late too.
- **18,000 seconds of server time for the whole evaluation.** After the last
  game, the verifier sums those trusted per-request timings over every `/act`
  request; a sum above 18,000 s scores zero. Only your server's own response
  time counts: Modal control-plane transport, Foul Play's search and the
  simulator do not. A late response counts with its time, about 1 second,
  since the client stops waiting there. Since every counted response is at
  most about 1 second, this budget binds only in a run with more than 18,000
  requests.
- **60 seconds per request including transport.** Each request, including the
  grader's Modal file upload, the remote exec of the trusted client and any
  retry, must finish within 60 s of wall time; otherwise the evaluation scores
  zero.
- **One retry for grading-transport faults only.** If a request fails in the
  transport itself, meaning a grading-sandbox command failure or timeout, or the
  local connection to your server refused, reset or dropped before a complete
  response (`ConnectionRefusedError`, `ConnectionResetError`,
  `ConnectionAbortedError`, `BrokenPipeError`, `RemoteDisconnected`,
  `IncompleteRead`), the grader sends the identical payload once more within the
  remaining part of those 60 s and reports the retry in the result. A response
  over the 1-second deadline, a non-200 status, a non-JSON body or an invalid
  action list is never retried.
- **12-hour wall-clock limit.** The whole evaluation, including setup, the
  opponent's search and simulation, must finish within 43,200 s of wall time.
  This limit only stops a stalled evaluation; reaching it also scores zero.

Except for a late response, which forfeits only the games in its batch, any of
these failures, an invalid response, a failed review or a startup failure gives
score zero for the whole run; no per-game partial credit survives.
Public validation defaults to two opponent workers and finishes their searches
before calling the candidate. This prevents opponent CPU contention during the
endpoint timing on a small shared agent VM. It still sends batches of up to 32.
Use `--opponent-workers 16 --parallel-decisions` when the opponent and candidate
have separate CPU allocations. Fewer workers can make validation take longer.
Public validation retains a 3,600-second overall allowance for its 32 games.

The default public response limit is also 1 second, measured from the validation
client; a late response forfeits its batch's games there too. A two-CPU development VM cannot certify the final eight-CPU H100 latency.
Use `--move-seconds 8` (at most 30) for quality debugging on slower hardware; the
report records this diagnostic limit and never certifies final timing. Validate
latency separately on representative eight-CPU hardware with full batches. Final
grading always uses the trusted H100-local one-second timer, separate opponent
CPUs, 16 workers and concurrent policy decisions; public flags cannot change it.

For development, `pokemon_random_battle.evaluate(..., baseline="tactical",
observer=...)` generates heuristic candidate teacher decisions against Foul Play without an HTTP
model. These heuristics are never evaluation opponents. The runtime
archive includes the pinned upstream generator and mechanics.
`pokemon_random_battle_teams.py` creates paired schedules and seeds; it does not
replace the simulator's random-team distribution. Use independent development
seeds for training. The `starter/` example trains an action scorer and exports
safetensors; it is a small recipe, not a competitive solved submission.

Existing public bots and pretrained policies, including unchanged or modified
Foul Play, are allowed. Heuristics, search, learned models and combinations are
permitted; new training and learned weights are optional. Every submitted bot must
meet the same offline, stateless and timing requirements and use only the supplied
player view and public mechanics.

Retain serving source, configuration and provenance for reused code/models, plus
training source, data provenance and readable logs if training is performed.
Shared `architecture.json` declares the actual serving path and any model's
repository/revision. A bot without a neural base may use `base_model="none"`,
`base_revision="none"` and no weight files; list reused code repositories and
revisions separately. A newly trained model may declare `base_model="none"` and
`base_revision="from-scratch"`. Export any weights as safetensors. Keep raw
development caches outside `/workspace/output/`.
