"""Shared review, hash-pinned fixed matchups, then actual offline battle outcomes."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from functools import partial
from itertools import chain, count
import json
import hashlib
import math
import statistics
import tempfile
import time

from frontier_ml.runtime.server import VerificationFailure
from frontier_ml.scoring.architecture import verify_review
from frontier_ml.taskspec import sibling
from frontier_ml.validation.errors import EvaluationError
from pokemon_random_battle import BATCH_SECONDS, EXECUTION_SECONDS, FINAL_GAMES, check_runtime, check_foul_play_dependencies, checked_rows, evaluate
from pokemon_random_battle import (DEVELOPMENT_SEED, LATE, MAX_BATCH, MOVE_TIMEOUTS, REQUEST_SECONDS, RUNTIME_FILES,
                                   FoulPlay, candidate_batch, engine, parse_actions, step_message, transport_fault)
from pokemon_random_battle_teams import scenarios
from pokemon_move_bridge import MOVE_SECONDS, TIMINGS

REVIEW_CHECKS = list(sibling(__file__, "review.py").POLICY["checks"])
# Self-check (frontier_ml/selfcheck.py): rounds of the public games, played and timed as evaluate() plays the final
# ones. It reads the public schedule and the pinned simulator and opponent runtime, about 5 MB.
SELFCHECK_PUBLIC = ("validation.jsonl", "runtime.json", *sorted(RUNTIME_FILES))
GRADE_REQUESTS = 3722  # POST /act batches of a complete 3,000-game grade (2026-09-29), to project the execution budget
# Decision rounds, a sample of a complete grade: as many requests as fill 600 s of server time at the execution
# budget's rate (3,722 in 18,000 s: 124), one in flight as in grading. The first is every game's opening move, a cold
# batch of 32; public logs grow every turn, and by round 60 requests are about as large as a complete grade's largest
# (about 400 KB of JSON).
SELFCHECK_ROUNDS = max(8, GRADE_REQUESTS * 600 // EXECUTION_SECONDS)
SELFCHECK_SLICE_SECONDS = 420  # no round starts later (a round through the Modal relay takes about 2.6 s)
# The organizer's simulator and opponent: their setup, the slice and its last round. With the move gates and the
# closing probe this stays within selfcheck.py's default work budget.
SELFCHECK_RUNTIME_SECONDS = 660
MOVE_GATE = f"move response within {MOVE_SECONDS:g} s"


@contextmanager
def launch(snapshot, config, data_path):
    from frontier_ml.runtime.inference import serve_submission  # Harbor is optional (viewer build)
    with serve_submission(snapshot, config) as server:
        resource = server["transport"].resource
        # Readiness uses the ordinary health client. All scored /act requests
        # then use this task's root-owned, bounded HTTP client.
        resource.upload(Path(__file__).with_name("pokemon_move_bridge.py"),
                        "/opt/frontier-grading/http_bridge.py")
        yield {**server, "move_timer_installed": True}


def move_timings(server, expected, late=0):
    """The trusted move client's record: one timing per request, `late` of them over the move deadline (the bridge
    marks those; their games were forfeited). Every timing counts toward the execution budget."""
    with tempfile.TemporaryDirectory(prefix="pokemon-move-times-") as directory:
        path = Path(directory) / "timings.jsonl"
        server["transport"].resource.download(TIMINGS, path)
        entries = [json.loads(line) for line in path.read_text().splitlines()]
    values = [entry["elapsed_seconds"] for entry in entries]
    over = [entry.get("late") is True for entry in entries]
    if (not values or len(values) != expected or sum(over) != late
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 or (v > MOVE_SECONDS and not was_late)
                   for v, was_late in zip(values, over))):
        raise VerificationFailure("Missing, invalid or over-limit trusted move timings")
    execution = sum(values)
    if execution > EXECUTION_SECONDS:
        raise VerificationFailure(f"Execution budget exceeded: {execution:.0f} s of server time > {EXECUTION_SECONDS} s")
    values.sort()
    return {"source": "trusted_h100_local_http_client", "limit_seconds": MOVE_SECONDS,
            "execution_seconds": execution, "execution_budget_seconds": EXECUTION_SECONDS,
            "requests": len(values), "late_requests": late, "median_seconds": statistics.median(values),
            "p95_seconds": values[math.ceil(.95 * len(values)) - 1], "max_seconds": max(values)}


def check_schedule(config, data_path):
    expected = config.get("evaluation_sha256")
    if (not isinstance(expected, str) or len(expected) != 64
            or hashlib.sha256(Path(data_path).read_bytes()).hexdigest() != expected):
        raise VerificationFailure("Evaluation schedule differs from the pinned evaluation_sha256")
    checked_rows(data_path, expected=FINAL_GAMES, require_fixed=True)


def prescreen(config, data_path):
    check_schedule(config, data_path)
    check_runtime(data_path)
    check_foul_play_dependencies()


def preflight(config, data_path):
    prescreen(config, data_path)
    review = config.get("architecture_review_path")
    if not review or not Path(review).is_file():
        raise VerificationFailure("Trusted architecture_review_path is required before provisioning")
    verify_review(config["submission_path"], review, "pokemon-random-battle", REVIEW_CHECKS)


def verify(server, data_path, config):
    if server.get("move_timer_installed") is not True:
        raise VerificationFailure("Trusted H100-local move timer is required")
    review = verify_review(server["snapshot"], config["architecture_review_path"], "pokemon-random-battle", REVIEW_CHECKS)
    # Recheck the immutable matchup file. No diagnostic seed can regenerate teams.
    check_schedule(config, data_path)
    result = evaluate(server["transport"], data_path, seed=None, expected=FINAL_GAMES,
        seconds=min(float(config.get("inference_timeout_seconds", BATCH_SECONDS)), BATCH_SECONDS),
        local_move_timer=True)
    timing = result["details"]["timing"]
    result["details"]["move_timing"] = move_timings(server, timing["requests"], late=timing["late_requests"])
    result["details"]["gates"] = {"architecture":{"reviewer":review.get("reviewer"),"checks":review["checks"]}}
    return result


def selfcheck(check, public, config):
    """One instance behind the trusted move client (this task's launch), then rounds of the public games against the
    pinned Foul Play, timed as verify() times the final ones. prescreen and preflight check only the organizer's
    schedule and runtime and the review receipt, so beyond the review's prerequisites (selfcheck.py checks those)
    nothing about the submission is checked statically."""
    data_path = Path(public) / "validation.jsonl"
    try:  # before the GPU: the organizer's simulator and opponent, whose faults are never the submission's
        checked_rows(data_path)
        check_runtime(data_path)
        check_foul_play_dependencies()
    except VerificationFailure as error:
        raise EvaluationError(f"The battle runtime is unavailable: {error}") from error
    with check.launch(launcher=partial(selfcheck_launch, data_path=str(data_path))) as server:
        played = play_slice(check, server, data_path)
        check.stage("gates")
        move_gates(check, server, played)


@contextmanager
def selfcheck_launch(snapshot, config, data_path):
    """launch(), plus the health() that selfcheck.py's closing probe calls in place of GET /health through the
    transport: the move client accepts only POST /act."""
    with launch(snapshot, config, data_path) as server:
        yield {**server, "health": partial(stock_health, server)}


def play_slice(check, server, data_path):
    """Up to SELFCHECK_ROUNDS decision rounds of the public games, played as evaluate() plays the final ones: 32 games
    at once on a rolling schedule (a finished game's slot gets a further development-seed matchup, so batches keep
    grading's size and mix of game ages), the candidate's batch and Foul Play's choices decided concurrently from the
    same pre-turn state, one batch in flight. A response over the move deadline forfeits every game in its batch, as in
    grading, and the slice goes on. Nothing is scored. Returns each request's round trip, batch size and body size, how
    many requests were sent again after a grading-transport fault, how many were late and how many games they forfeited."""
    from frontier_ml.selfcheck import CheckFailed
    rows = checked_rows(data_path)
    # The public schedule's matchups, then its pairings again with fresh development-seed teams.
    matchups = chain.from_iterable(scenarios(rows, DEVELOPMENT_SEED + n) for n in count())
    active = {slot: next(matchups) for slot in range(MAX_BATCH)}
    played = {"seconds": [], "sizes": [], "bytes": [], "retries": 0, "late": 0, "forfeited": 0}
    forfeited = set()
    rounds = replaced = 0
    deadline = time.monotonic() + SELFCHECK_RUNTIME_SECONDS
    try:
        with engine(data_path, deadline) as simulator, ExitStack() as cleanup:
            decisions = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pokemon-opponent")
            cleanup.callback(decisions.shutdown, wait=True, cancel_futures=True)
            opponent = FoulPlay(Path(data_path).parent.resolve(), Path(simulator.process.args[-1]), deadline)
            cleanup.callback(opponent.close)  # after a failed request, stop its search before joining the thread
            opponent.reset()
            state = simulator.call({"op": "init", "games": list(active.values())})
            started = time.monotonic()
            while rounds < SELFCHECK_ROUNDS and (not rounds or time.monotonic() - started < SELFCHECK_SLICE_SECONDS):
                finished = [slot for slot in active if state["results"][slot] is not None]
                if finished:  # as evaluate() refills: Foul Play forgets the finished games, the simulator starts new ones
                    forfeited.difference_update(finished)
                    opponent.reset(finished)
                    active.update((slot, next(matchups)) for slot in finished)
                    state = simulator.call({"op": "refill",
                                            "games": [{"slot": slot, "config": active[slot]} for slot in finished]})
                    replaced += len(finished)
                rounds += 1
                pending = decisions.submit(opponent.choose, state["opponents"], dict(active))
                actions = []  # a forced switch can require only the opponent
                slots, observations = candidate_batch(state, forfeited)
                if observations:
                    actions, elapsed, retried = act(check, server["transport"], observations, rounds,
                                                    first=not played["seconds"],
                                                    timed_here=not server.get("move_timer_installed"))
                    played["seconds"].append(elapsed)
                    played["sizes"].append(len(observations))
                    played["bytes"].append(len(json.dumps({"observations": observations}).encode()))
                    played["retries"] += retried
                    if actions is LATE:
                        forfeited.update(slots)
                        played["late"] += 1
                        played["forfeited"] += len(slots)
                        slots, actions = [], []
                opponent_actions = pending.result(timeout=max(0, deadline - time.monotonic()))
                state = simulator.call(step_message(state, slots, actions, opponent_actions, forfeited))
    except (CheckFailed, EvaluationError):
        raise
    except (TimeoutError, VerificationFailure) as error:  # the organizer's simulator or opponent stalled or failed
        raise EvaluationError(f"The battle simulation stopped: {type(error).__name__}: "
                              f"{str(error) or 'the opponent did not decide in time'}") from error
    if not played["seconds"]:
        raise EvaluationError("The battle simulator asked the candidate for no decision")
    sizes = played["sizes"]
    check.record("battle slice", True, kind="smoke", measured=len(sizes), unit="requests",
                 detail=f"{len(sizes)} POST /act batches over {rounds} decision rounds, {MAX_BATCH} games at a time: the "
                        f"first {min(len(rows), MAX_BATCH)} public games (teams from development seed "
                        f"{DEVELOPMENT_SEED}), {replaced} finished games replaced by further development-seed matchups, "
                        f"against the pinned Foul Play: {sum(sizes)} observations, {min(sizes)}-{max(sizes)} per batch, "
                        f"up to {max(played['bytes']) / 1000:.0f} KB per request, every answered action legal; not scored")
    return played


def act(check, transport, observations, number, *, first=False, timed_here=False):
    """One round's POST /act as evaluate() sends it (a grading-transport fault is retried once within
    REQUEST_SECONDS, a missed move deadline or an invalid answer never), checked with grading's parser. A missed move
    deadline returns LATE in place of the actions: grading forfeits that batch's games and goes on. Any other failure
    is the submission's and ends the self-check, as it ends grading; the first answer is recorded. Returns the actions,
    the round trip and whether the request was sent again. Without the trusted move client (timed_here: a local
    stand-in server), a round trip over the deadline counts as late."""
    from frontier_ml.selfcheck import excerpt
    name = f"POST /act, round {number}"
    payload = {"observations": observations}
    started = time.monotonic()
    retried = False
    try:
        try:
            response = transport.request("POST", "/act", payload, timeout=REQUEST_SECONDS)
        except Exception as error:
            left = REQUEST_SECONDS - (time.monotonic() - started)
            if not transport_fault(error) or left <= 0:
                raise
            retried = True
            response = transport.request("POST", "/act", payload, timeout=left)
    except EvaluationError:
        raise
    except Exception as error:  # noqa: BLE001 - a missed move deadline, a crash, a non-200 or non-JSON answer
        if str(error).startswith(MOVE_TIMEOUTS):
            return LATE, time.monotonic() - started, retried
        check.fail(name, f"{type(error).__name__}: {error}", kind="smoke")
    elapsed = time.monotonic() - started
    if timed_here and elapsed > MOVE_SECONDS:
        return LATE, elapsed, retried
    try:
        actions = parse_actions(response, observations)
    except Exception as error:  # noqa: BLE001 - grading rejects this answer and scores the run zero
        check.fail(name, f"the response to {len(observations)} observations was rejected: {error}; "
                         f"got {excerpt(response)}", kind="smoke")
    if first:  # the round trip includes the Modal transport, so it is no measure of the 1-second move deadline
        check.record(name, True, kind="smoke",
                     detail=f"{len(observations)} opening decisions answered with legal actions, {elapsed:.2f} s round "
                            f"trip on this side with the transport: {excerpt(actions)}")
    return actions, elapsed, retried


def move_gates(check, server, played):
    """The 1-second move deadline on the slice's responses, read from the trusted in-sandbox timer as verify() reads
    it, and the execution budget projected to a complete grade. A local stand-in server without the move client
    (tests, local runs) is timed round trip on this side instead."""
    seconds, sizes, late = played["seconds"], played["sizes"], played["late"]
    if server.get("move_timer_installed"):
        try:
            timing = move_timings(server, len(seconds), late=late)
        except VerificationFailure as error:
            # The root-owned move client records every answer, late ones marked, in a file the candidate cannot
            # write, so a mismatch is a request the grading transport ran twice (a retry after a fault that struck
            # once the bridge had answered, or a relay fallback) or a timer fault, never the submission's.
            raise EvaluationError(f"The trusted move timer's record does not match the self-check's {len(seconds)} "
                                  f"requests ({late} late, {played['retries']} sent again after a grading-transport "
                                  f"fault), so the move deadline was not measured: {error}") from error
        source = "on the trusted in-sandbox move timer"
    else:
        values = sorted(seconds)
        timing = {"execution_seconds": sum(values), "requests": len(values), "median_seconds": statistics.median(values),
                  "p95_seconds": values[math.ceil(.95 * len(values)) - 1], "max_seconds": values[-1]}
        source = "round trip on this side, without the trusted move timer"
    batches = (f"{min(sizes)}-{max(sizes)} observations, up to {max(played['bytes']) / 1000:.0f} KB; the first a cold "
               f"batch of {sizes[0]}")
    if late:
        check.warn(MOVE_GATE, measured=late, unit="late responses",
                   detail=f"{late} of {timing['requests']} POST /act responses ({batches}) took longer than "
                          f"{MOVE_SECONDS:g} s {source}; median {timing['median_seconds']:.3f} s, p95 "
                          f"{timing['p95_seconds']:.3f} s. Grading forfeits every game in a late batch (each counts as "
                          f"a loss) and plays on, so this slice lost {played['forfeited']} games that way. A grade "
                          f"sends {GRADE_REQUESTS:,} requests and plays whole games, whose late turns carry longer "
                          "public logs")
    else:
        check.gate(MOVE_GATE, timing["max_seconds"], MOVE_SECONDS, higher_is_better=False, unit="s",
                   detail=f"slowest of {timing['requests']} POST /act responses ({batches}) {source}; median "
                          f"{timing['median_seconds']:.3f} s, p95 {timing['p95_seconds']:.3f} s. A grade plays whole "
                          f"games, whose late turns carry longer public logs, and sends {GRADE_REQUESTS:,} requests; "
                          "every game in a response over the deadline is forfeited (a loss)")
    requests, mean = timing["requests"], timing["execution_seconds"] / timing["requests"]
    check.gate("execution budget (projected)", round(mean * GRADE_REQUESTS, 1), EXECUTION_SECONDS,
               higher_is_better=False, unit="s",
               detail=f"projected from {requests} of {GRADE_REQUESTS:,} requests at concurrency 1: mean {mean:.3f} s "
                      f"per request x {GRADE_REQUESTS:,}, as many as one complete 3,000-game grade sent; grading sums "
                      "every request's in-sandbox time")
    check.note(f"The speed gates were measured on a sample and the execution budget projected from it: {requests} of a "
               f"complete grade's {GRADE_REQUESTS:,} POST /act requests, from the first decision rounds of the public "
               "games rather than whole games.")


def stock_health(server):
    """GET /health for selfcheck.py's closing probe, through the shared HTTP bridge put back in the move client's
    place; grading sends nothing but POST /act once launch() installs the move client."""
    from frontier_ml.runtime import inference
    try:
        server["transport"].resource.upload(Path(inference.__file__).with_name("http_bridge.py"),
                                            "/opt/frontier-grading/http_bridge.py")
    except Exception as error:  # noqa: BLE001 - the grading sandbox's fault, never the submission's
        raise EvaluationError(f"Could not reinstall the shared HTTP bridge: {type(error).__name__}: {error}") from error
    return server["transport"].raw_request("GET", "/health", timeout=10)
