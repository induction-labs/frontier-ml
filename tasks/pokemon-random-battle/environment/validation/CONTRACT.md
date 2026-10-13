# Build a Pokémon Gen 9 Random Battle bot

Existing public bots and pretrained policies, including unchanged Foul Play, are permitted submissions.  The pinned simulator uses gen9randombattle and its upstream random-team generator, including its species pool, sets and balancing levels. Use all six team members with one active at a time and one Terastallization per battle. There is no team preview or choice of opening lead. Opponent team members and sets remain hidden until revealed by gameplay; exact HP, pending actions and RNG state remain private. Use only the supplied observation and public mechanics. Do not access private evaluation files, recognize fixed test cases, or cache their answers. Include readable serving source, configuration and provenance for reused code and models. If you train, include training code, data provenance and readable logs. Export any model weights as safetensors; a bot without learned weights does not need a checkpoint.

## Server interface

POST /act with JSON {"observations": [observation, ...]}, at most 32 observations and 8 MiB per request. Each observation contains format, player, turn, request (own private Showdown request), public_log, legal_actions, and action_features. Legal actions are single-Pokémon Showdown command strings; the parallel numeric feature rows are optional convenience inputs derived solely from visible information. Return exactly {"actions": [index, ...]} with one integer index into each observation's legal_actions, in request order. No extra fields or omitted actions. Treat requests as stateless; public_log contains the complete visible history so far. Normal turns, forced switches and simulator-updated choices use this same endpoint. An unavailable choice that reveals new information produces an updated observation; accepted opponent choices remain locked.

## Evaluation

32 public games with fresh random teams and battle seeds on every validation run; 3000 final games exclusively against Foul Play using the same stored teams and battle seeds for every submission. The 1500 matchups are generated once, each played from both candidate sides. The pinned opponent uses its upstream default search settings: 100 ms per sampled state, one search process and one thread. Sixteen trusted CPU workers run up to 32 active games, replacing completed games immediately. The two policies decide concurrently from the same pre-turn views. For Revival Blessing, the adapter compares legal revived-team afterstates using the same native search, dividing that search budget across revival targets. The adapter applies upstream's four-move truncation before computing native move indices, preventing invalid fifth-slot indices during inference. Speed inference skips selected switches, matching upstream's Choice Scarf guard. Intermediate form changes absent from the final request defer own-stat refresh; the same Pokémon is identified by its player/nickname, and exact final stats are refreshed before search. Opponent information remains player-visible only. Foul Play's search is stochastic, so fixed matchups do not imply identical outcomes. Pair matchups with team assignments swapped and balance player seats. Raw battle points are win 1, natural tie 0.5, loss 0, averaged over all 3000 games. If the Foul Play opponent makes an illegal choice, it forfeits that game, which counts as a candidate win. Final reward is max(0, min(1, (mean_battle_points - 0.5) / 0.5)): 50% or below earns 0, 75% earns 0.5, and 100% earns 1. The 50% reference is fixed, not remeasured per submission; an unchanged Foul Play submission receives its measured result on this same scale. A game reaching 200 turns or 512 decision rounds scores 0 for the candidate. Your server's total response time, summed over all requests on the trusted in-sandbox move timer, must stay within 18,000 seconds; the opponent's search and the simulation do not count. A 12-hour wall-clock limit only stops a stalled evaluation. Every complete POST /act response must arrive within 1 second on the H100, including batches of up to 32 moves. A trusted local HTTP client times the entire response, including feature processing, search and inference; Modal control-plane transport is excluded from this move deadline. A response over 1 second forfeits every game in that batch: each counts as a loss, the late answer is discarded and the evaluation goes on with the other games. Its time still counts toward the 18,000-second total. The grader does not wait for a late answer, so a server still working on it when the next batch arrives can make that batch late too. Each request including that transport is separately capped at 60 seconds within the full evaluation budget. A request that fails in transport (a grading-sandbox command failure or timeout, or a dropped connection) is retried once within those 60 seconds; a response over the 1-second move deadline or an invalid response is never retried. Invalid/missing actions, failed review, startup failure or any other timeout give score zero for the whole run. Public validation reports raw diagnostic battle points; final grading applies the reward normalization and compliance review. Public validation defaults to two opponent workers and sequential policy decisions to avoid contention on a small shared VM. Its --move-seconds option is for quality debugging only and cannot certify or relax the final 1-second deadline. No live ladder or external account is involved.

## Artifacts

No starting checkpoint is prescribed. Existing public code and weights are allowed. /task/public/ contains the pinned simulator runtime, Foul Play source/runtime data, SCHEMA.md and a small training/server example. Install the validation opponent's dependencies with public/setup_foul_play.sh using Python 3.12; point FOUL_PLAY_PYTHON at that environment's Python. Linux x86-64 uses the supplied engine wheel; other platforms require Rust >=1.89. The trusted verifier preinstalls these dependencies. If your submission uses poke-engine, copy the exact organizer-built CPython 3.12 Linux x86-64 wheel from /task/public/engine/ and install it offline at startup. This exact wheel is a task-specific audited dependency, in addition to unmodified published wheels permitted by the shared contract. Candidate-built native wheels are not covered by this audit. The corresponding upstream source and build receipt are alongside it. A bot without a learned checkpoint may declare base_model="none" and base_revision="none" in architecture.json, with an empty training_evidence_files list if no training was performed. List reused code repositories and revisions separately and include the actual serving source in inference_evidence_files. /task/pokemon_random_battle_teams.py supplies development seeds for the upstream random-team generator. The same runtime can generate additional development battles for imitation learning or self-play. Read the public schema before implementing an observation encoder.

Include architecture.json with base_model, base_revision, training_evidence_files and inference_evidence_files. The file lists use paths relative to output/ and include training source, serve.sh and model-loading code. Keep initialization/training logs with the submission as readable text or JSON. Name the source of every training, calibration and selection corpus (in architecture.json or a data manifest) and keep its download commands and logs with the submission. I will check for training-data provenance. Ship compiled dependencies as unmodified PyPI or PyTorch wheels, installed into /tmp at startup. Prewarmed JIT kernel caches (Triton, TorchInductor) may be included.

## Submission

Put serve.sh, code, weights and dependencies in /workspace/output/ as regular files. Relative
symlinks that resolve inside output/ are copied as the regular files they point to. However, absolute, escaping,
dangling or cyclic symlinks fail the submission (for example, use `python -m venv --copies`, or vendor wheels).
The /workspace/output folder is copied to /submission when being evaluated. serve.sh runs as an unprivileged 
user that can write to /submission, /tmp and the preinstalled Python environment (/opt/venv) and nowhere else; 
install vendored wheels there or under /submission. You should package dependencies for offline installation,
since the evaluation server will not have internet access.
The command `bash /submission/serve.sh` receives HOST=0.0.0.0, PORT=8000 and
MODEL_NAME=submission. GET /health must return 200 when ready, within 300 seconds.
Startup and inference run offline on one H100 SXM 80GB, with 8 CPU cores and 80 GiB RAM.
The serving image is Ubuntu 24.04 with the CUDA 12.8 runtime libraries (no CUDA compiler), Python 3.12 in /opt/venv first on PATH (pip, numpy, pillow, safetensors, httpx preinstalled), gcc and Python headers, plus apt nodejs. PyTorch is not preinstalled: vendor cp312 manylinux wheels (and their dependencies).
Every request must answer HTTP 200 with the documented JSON within the task's deadline, and requests 
are never retried. A failed, malformed or late response scores zero for the whole run.

## Development

The development machine is CPU-only. Use `spin --help` to rent compute and access
model APIs, and `spin budget` to check how much allowance you have left.
The API key your coding harness was started with is for the harness alone. Do not use the API key
in your own code, scripts, curl or Spin machines, copy it anywhere, or start extra harness sessions as workers;
AI model calls you make yourself must go through `spin` and spend your allowance. I will check your session transcript for this.

To check your solution locally, run `python /task/validate.py --server-url URL`. See /task/public for examples 
and supporting files.

You can check your solution on the same infrastructure I will evaluate it on using `frontier selfcheck`,
which copies /workspace/output/, starts it as the evaluation does, sends public examples through the same 
response checks and measures the speed requirements on public inputs. Its report gives the stage reached, 
the digest of the files it checked, the GPU, the startup time and each requirement's measured value and 
margin, and the end of the server log. You can check your solution four times.

A check usually takes 10-40 minutes (`frontier selfcheck status` shows this task's limit) and runs while you keep
working. See `frontier selfcheck --help` for more details.

When your solution is final, hand it in with `frontier submit`: it shows what /workspace/output/ holds and asks
you to confirm with `frontier submit --confirm`, which submits the folder as it is and ends your session. Without
a submission, /workspace/output/ is graded as it is when your session ends. See `frontier submit --help`.

Your session ends the moment you finish a turn without a tool call: background commands, scheduled
wake-ups and reminders will not bring you back, and anything not yet in /workspace/output/ is lost.
Wait for long jobs with foreground commands under the tool time limit (for example
`sleep 600; tail -n 3 train.log`, repeated) and stop only when the final files are in place.
