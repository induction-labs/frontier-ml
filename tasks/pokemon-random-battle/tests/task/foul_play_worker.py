"""Organizer adapter to unmodified Foul Play; only a player's allowed view enters here.

Run in the separate dependency environment. The public validator itself is stdlib.
The packaged upstream source retains its GPL-3.0 license and corresponding source.
"""
import copy
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import math
import multiprocessing
from pathlib import Path
import random
import socket
import sys
import traceback


FORFEIT = -1  # opponent action meaning: Foul Play made an illegal choice and forfeits this game

def install_engine_adapter():
    """Apply upstream's four-move truncation before computing native indices.

    Upstream computes last_used_move against the full list, then truncates that
    list while converting the Pokémon. Inference can temporarily contain more
    than four moves; a fifth last-used move becomes an invalid native enum and
    panics. Keep the same first four moves and upstream's existing missing-move
    behavior, but derive the index from the actual list sent to the engine.
    """
    from fp.search import poke_engine_helpers as helpers

    original = helpers.battler_to_poke_engine_side
    if getattr(original, "frontier_move_index_fix", False):
        return

    def convert(battler, *args, **kwargs):
        if len(battler.active.moves) > 4:
            battler.active.moves = battler.active.moves[:4]
        return original(battler, *args, **kwargs)

    convert.frontier_move_index_fix = True
    helpers.battler_to_poke_engine_side = convert


def install_protocol_adapter():
    """Do not infer move-order speed from a selected switch.

    A request can split the turn's public messages after the switch line. When
    only the opponent's move remains, upstream looks up our last selection as a
    move, even if it was a switch. Match its existing Choice Scarf guard: a
    switch supplies no comparable move priority or speed evidence.
    """
    from fp.battle import protocol

    original = protocol.check_speed_ranges
    if getattr(original, "frontier_switch_speed_fix", False):
        return

    def check_speed_ranges(battle, msg_lines):
        if battle.user.last_selected_move.move.startswith("switch "):
            return
        return original(battle, msg_lines)

    check_speed_ranges.frontier_switch_speed_fix = True
    protocol.check_speed_ranges = check_speed_ranges


def install_form_adapter():
    """Defer own-stat refresh for intermediate forms absent from the request.

    Showdown's request describes the end of the entire message batch. Eiscue can
    regain and lose its Ice Face within that batch; upstream refreshes its own
    stats after each form change and cannot find the intermediate form. Only
    defer when the same own Pokémon identifier has another form of that species.
    Public events still update the form, and the normal pre-search request
    refresh supplies exact final stats. Other mismatches must still fail.
    Terapagos-Stellar keeps upstream's Terastal request mapping; only when the
    request has no Terastal form either (a fainted Terapagos reverts to its base
    form in the request) does it defer like any other intermediate form.
    """
    from fp.battle.helpers import normalize_name
    from fp.battle.state import Battler
    from fp.data import pokedex

    original = Battler.re_initialize_active_pokemon_from_request_json
    if getattr(original, "frontier_intermediate_form_fix", False):
        return

    def refresh(side, request):
        active = side.active
        species = pokedex[active.name]
        rows = request["side"]["pokemon"]
        names = {normalize_name(species["name"]), active.base_name}
        matching = [row for row in rows
                    if normalize_name(row["details"].split(",")[0]) in names]
        # Preserve upstream's explicit Stellar/Terastal request compatibility.
        terastal = active.name == "terapagosstellar" and any(
            normalize_name(row["details"].split(",")[0]) == "terapagosterastal" for row in rows)
        if not matching and not terastal:
            # Showdown reorders team entries on switches; a stored index is not
            # a stable identity. Require exactly one matching player/nickname.
            identified = [row for row in rows
                          if active.nickname is not None
                          and row["ident"] == f"{side.name}: {active.nickname}"]
            if len(identified) == 1:
                row = identified[0]
                requested = pokedex.get(normalize_name(row["details"].split(",")[0]))
                if (requested is not None
                        and normalize_name(species.get("baseSpecies", species["name"]))
                        == normalize_name(requested.get("baseSpecies", requested["name"]))):
                    return
        return original(side, request)

    refresh.frontier_intermediate_form_fix = True
    Battler.re_initialize_active_pokemon_from_request_json = refresh


def forced_decision(obs, battle):
    """Protocol bookkeeping for a singleton menu, e.g. a recharge turn."""
    if len(obs["legal_actions"]) != 1:
        return None
    command = obs["legal_actions"][0]
    slot = int(command.split()[1]) - 1
    if command.startswith("move "):
        return obs["request"]["active"][0]["moves"][slot]["id"]
    target = next(p for p in battle.user.reserve if p.index == slot + 1)
    return "switch " + target.name


def revival_decision(obs, battle, known_max_hp):
    """Compare legal Revival Blessing afterstates with the pinned native search.

    Showdown encodes revival targets as switches, but upstream Foul Play does not
    pass its `reviving` flag to poke-engine. Ordinary forced-switch search would
    therefore select a living reserve. Keep the active Pokémon in place, revive
    each legal target in a separate copy, and compare native search values. Split
    the usual search milliseconds among targets; normal decisions are unchanged.
    """
    if not any(p.get("active") and p.get("reviving")
               for p in obs["request"]["side"]["pokemon"]):
        return None
    from fp.config import FoulPlayConfig
    from fp.search.main import get_result_from_mcts
    from fp.search.poke_engine_helpers import battle_to_poke_engine_state

    slots = [int(command.split()[1]) for command in obs["legal_actions"]]
    targets = [next(p for p in battle.user.reserve if p.index == slot) for slot in slots]
    if any(p.hp != 0 for p in targets):
        raise ValueError("A Revival Blessing target must be fainted")
    count, milliseconds = battle.mode.search_params(battle)
    samples = battle.mode.prepare_battles(battle, count)
    values = [0.0] * len(targets)
    with ProcessPoolExecutor(max_workers=FoulPlayConfig.parallelism) as executor:
        pending = []
        for sample, chance in samples:
            for index, target in enumerate(targets):
                after = copy.deepcopy(sample)
                after.force_switch = False
                after.user.active.reviving = False
                revived = next(p for p in after.user.reserve if p.index == target.index)
                # The request spells a fainted condition as "0 fnt", without a
                # maximum. Upstream's request refresh sets max_hp to zero there;
                # retain the maximum already observed in our own earlier request.
                revived.max_hp = known_max_hp[target.name]
                if revived.max_hp <= 0:
                    raise ValueError("Missing previously observed maximum HP for revival")
                revived.hp = max(1, int(revived.max_hp) // 2)
                revived.status = None
                duration = milliseconds // len(targets) + (index < milliseconds % len(targets))
                state = battle_to_poke_engine_state(after).to_string()
                future = executor.submit(get_result_from_mcts, state, duration, index,
                                         FoulPlayConfig.search_threads)
                pending.append((index, chance, future))
        for index, chance, future in pending:
            result = future.result()
            visits = sum(option.visits for option in result.side_one)
            if visits <= 0:
                raise ValueError("Revival search returned no visits")
            value = sum(option.total_score for option in result.side_one) / visits
            if not math.isfinite(value):
                raise ValueError("Revival search returned a nonfinite value")
            values[index] += chance * value
    best = max(values)
    index = random.choice([i for i, value in enumerate(values) if value == best])
    return "switch " + targets[index].name


def main():
    # All sets and mechanics are local. A network access is an integration error.
    def offline(*args, **kwargs):
        raise RuntimeError("Foul Play evaluation cannot access the network")
    socket.socket.connect = offline
    socket.create_connection = offline
    # Upstream starts its search process pool per decision. These are single-threaded
    # workers, so fork avoids re-importing the dex on every decision on macOS too.
    multiprocessing.set_start_method("fork", force=True)
    sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
    from fp.battle.state import Battle, LastUsedMove
    from fp.battle.protocol import process_battle_updates, update_battle
    from fp.config import FoulPlayConfig
    from fp.data.mods.apply_mods import apply_mods
    from fp.modes.random_battle import RandomBattleMode
    from fp.search.main import find_best_move

    install_engine_adapter()
    install_protocol_adapter()
    install_form_adapter()
    logging.basicConfig(stream=sys.stderr, level=logging.ERROR)
    FoulPlayConfig.pokemon_format = "gen9randombattle"
    FoulPlayConfig.search_time_ms = 100
    FoulPlayConfig.parallelism = 1
    FoulPlayConfig.search_threads = 1
    apply_mods(FoulPlayConfig.format_spec)
    mode = RandomBattleMode()
    mode.datasets.pkmn_mode = "gen9randombattle"
    mode.datasets.raw_pkmn_sets = json.loads(Path(sys.argv[2]).read_text())
    mode.datasets._initialize_pkmn_sets()
    games = {}

    def decide(item):
        key, obs = item["game"], item["observation"]
        req = copy.deepcopy(obs["request"])
        req["rqid"] = item["decision"]
        lines = obs["public_log"]
        if key not in games:
            battle = Battle("offline")
            battle.pokemon_format = "gen9randombattle"
            battle.generation = battle.format_spec.generation
            battle.battle_type = battle.format_spec.battle_type
            battle.mode = mode
            battle.user.name = obs["player"]
            battle.opponent.name = "p2" if obs["player"] == "p1" else "p1"
            battle.started = True
            battle.request_json = req
            battle.rqid = req["rqid"]
            battle.user.initialize_first_turn_user_from_json(req)
            # Same initialization as RandomBattleMode.start_battle: request sets
            # our lead; apply the remaining first-turn protocol messages once.
            begin = lines.index("|start") + 1
            battle.msg_list = [line for line in lines[begin:]
                              if not line.startswith("|switch|" + battle.user.name)]
            process_battle_updates(battle)
            games[key] = (battle, len(lines), random.Random(item["policy_seed"]).getstate())
        battle, offset, rng_state = games[key]
        random.setstate(rng_state)
        update_battle(battle, "\n".join(lines[offset:] + ["|request|" + json.dumps(req)]))
        battle_copy = copy.deepcopy(battle)
        known_max_hp = {p.name: p.max_hp for p in [battle.user.active, *battle.user.reserve]}
        battle_copy.user.update_from_request_json(req)
        # A singleton menu has no strategic choice. In particular, poke-engine
        # can suggest recharge-tera although Showdown forbids tera while recharging.
        choice = (forced_decision(obs, battle_copy) or revival_decision(obs, battle_copy, known_max_hp)
                  or find_best_move(battle_copy))
        battle.user.last_selected_move = LastUsedMove(
            battle.user.active.name, choice.removesuffix("-tera").removesuffix("-mega"), battle.turn)
        games[key] = (battle, len(lines), random.getstate())
        if choice.startswith("switch "):
            name = choice.removeprefix("switch ")
            targets = [p.index for p in battle_copy.user.reserve if p.name == name]
            if len(targets) != 1:
                return FORFEIT  # an unknown switch is an illegal choice: Foul Play loses this game
            command = "switch " + str(targets[0])
        else:
            move = choice.removesuffix("-tera")
            moves = req.get("active", [{}])[0].get("moves", [])
            targets = [i+1 for i, m in enumerate(moves) if m["id"] == move]
            if len(targets) != 1:
                return FORFEIT  # an unknown move is an illegal choice: Foul Play loses this game
            command = "move " + str(targets[0]) + (" terastallize" if choice.endswith("-tera") else "")
        # Never substitute a heuristic, drop tera, or silently take the first action. An illegal choice (e.g. poke-engine
        # suggesting Struggle-tera on a multi-option menu, 2026-09-29) forfeits this game to the candidate: the
        # simulator records it as a candidate win (reason opponent_illegal). Worker crashes remain infrastructure errors.
        if command not in obs["legal_actions"]:
            print(f"Illegal Foul Play choice {command}; menu={obs['legal_actions']}; turn={obs['turn']}; forfeit",
                  file=sys.stderr, flush=True)
            return FORFEIT
        return obs["legal_actions"].index(command)

    for line in sys.stdin:
        try:
            message = json.loads(line)
            if message["op"] == "reset":
                if "games" in message:
                    for key in message["games"]:
                        games.pop(key, None)
                else:
                    games.clear()
                result = {"ready": True}
            else:
                result = {"actions": [decide(item) for item in message["items"]]}
            print(json.dumps(result), flush=True)
        except BaseException:
            # PyO3 PanicException inherits directly from BaseException. Report it
            # as an opponent infrastructure failure; never substitute an action.
            print(json.dumps({"error": traceback.format_exc()}), flush=True)
            return


if __name__ == "__main__":
    main()
