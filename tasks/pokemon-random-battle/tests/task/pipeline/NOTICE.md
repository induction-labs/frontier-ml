# Sources and licenses

Pokémon Showdown: <https://github.com/smogon/pokemon-showdown>, MIT license,
pinned commit `ea822f8863407f7a16995a9db3519f8e250237c1` (2026-01-13).
`showdown.tar.gz` contains its compiled `dist/sim`, `dist/data`, `dist/lib`,
`dist/config` JavaScript/JSON, package metadata and upstream `LICENSE`.
The runtime dependency `ts-chacha20` 1.2.0 is MIT licensed; its package and license
are included under `node_modules/ts-chacha20`. No sprites, sound files, ROMs or pretrained weights are distributed here.

`simulator.js`, the Python driver, schedules, and starter policy example
are original task code. The upstream simulator is not patched. The game format is
`gen9randombattle`: Generation 9 Random Battle singles. Teams and balancing levels
come from the upstream random-team generator and its set data at the pinned commit.

The task is an independent research benchmark and is not affiliated with Nintendo,
Game Freak, The Pokémon Company or Smogon.

Foul Play: <https://github.com/pmariglia/foul-play>, GPL-3.0, pinned commit
`6c467c081e862fb321adb405355beb41aba8e226` (2026-09-06). `foul-play.tar.gz` retains
all tracked upstream files, including complete source, tests, README and LICENSE.
The upstream source is unmodified. `foul_play_worker.py` adapts the offline
protocol and loads a local set prior rather than downloading live set statistics.
It calls the upstream battle parser, hidden-state sampler and `find_best_move`.
The search dependency poke-engine 0.0.48 is MIT licensed and installed separately
with the upstream Terastallization build feature. Its source and license are at
<https://github.com/pmariglia/poke-engine> and in the versioned PyPI distribution.
`engine/` includes that complete, unmodified source distribution (SHA-256
`070010686f2aedff11e25137e696e301ccd80fd57c805d255464067fc905ca12`), its MIT license,
the organizer-built CPython 3.12 Linux x86-64 wheel and `engine-build.json` receipt.
The receipt identifies the compiler, build frontend, feature flags and Cargo lock
digest. The operator recipe is `pipeline/build_poke_engine.py` in this repository.

`foul_play_priors.js` samples 100,000 teams from the pinned public Showdown generator
with seed `[179,421,991,2027]`. `foul-play-priors.json` stores aggregate species/set
frequencies, independently of all evaluation matchups. This prevents live set-data
drift and uses the correct pinned format's levels and sets. Normal decisions use
the upstream search policy. The adapter extends the unsupported Revival Blessing
target request by comparing legal revival afterstates with the same native search,
sharing the existing search budget across targets. It never substitutes a heuristic.
The adapter also applies the upstream four-move truncation before, rather than
after, computing native move indices. This prevents a native panic when inference
temporarily contains a fifth move; the retained moves and search policy are unchanged.
It also skips speed inference following a selected switch, matching upstream's
Choice Scarf guard. This prevents a switch command from being looked up as a move
when a request splits the turn's protocol messages. Normal move-order inference
and the upstream source archive remain unchanged.
The form adapter defers own-stat refresh for intermediate forms absent from the
final request, using a unique player/nickname and matching base species to
establish identity. Public form events still apply and exact own stats refresh
before search. It preserves normal refresh behavior, including upstream's
Terapagos Stellar/Terastal handling; unrelated or ambiguous mismatches still fail.

Public mechanics tables and set priors receive normal source review. The
architecture reviewer accepts the exact task-provided simulator archive by SHA-256;
changed archives are rejected. Executable source, training evidence, weight
inventories and upstream wheel provenance retain their normal review.
The exact organizer-built engine wheel and corresponding
source archive have content-bound task audits; a candidate-produced wheel is not
accepted merely because it has the same filename, version or claimed provenance.
The audits do not execute candidate code or alter the submitted model.
