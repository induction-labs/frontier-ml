"""Stdlib-only trusted battle driver and public development evaluation."""
from contextlib import contextmanager, ExitStack
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import random
import select
import shutil
import signal
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time

from frontier_ml.validation.errors import EvaluationError, VerificationFailure
from frontier_ml.validation.evaluation import load_samples, timed_batch
from pokemon_random_battle_teams import OPPONENT, scenarios

MAX_BATCH = 32
MAX_REQUEST_BYTES = 8 * 1024 * 1024
FINAL_GAMES = 3000
# The scoring budget is the candidate's own execution time: the trusted in-sandbox move timer summed over every /act
# request (checked by the verifier from its timings). Grader-side transport (Modal uploads/execs), the opponent's search
# and the simulator do not count, since the candidate cannot control them. WALL_SECONDS only stops a stalled
# evaluation; it is far above any observed grade (73-243 min).
EXECUTION_SECONDS = 18000
WALL_SECONDS = 43200
BATCH_SECONDS = WALL_SECONDS  # the evaluation's wall-clock safety limit (kept name for callers)
REQUEST_SECONDS = 60  # per /act request including Modal transport, and including its one transport retry
# Failures of the grading transport itself, not of the candidate's move: a Modal command that failed or timed out, or
# the connection to the server dropping. Such a request is retried once within its REQUEST_SECONDS. The move bridge's
# own 1-second deadline (reported as "Submission HTTP request failed: TimeoutError") is the contract's move rule: it is
# never retried and forfeits every game in that batch. An invalid response is never retried either.
TRANSPORT_FAULTS = ("Grading resource operation timed out", "Grading sandbox command failed",
                    "Submission HTTP request failed: ConnectionRefusedError", "Submission HTTP request failed: ConnectionResetError",
                    "Submission HTTP request failed: ConnectionAbortedError", "Submission HTTP request failed: BrokenPipeError",
                    "Submission HTTP request failed: RemoteDisconnected", "Submission HTTP request failed: IncompleteRead")


def transport_fault(error):
    return isinstance(error, (RuntimeError, OSError)) and any(str(error).startswith(fault) for fault in TRANSPORT_FAULTS)
# The move client's deadline as the transport reports it (socket.timeout is named "timeout" before Python 3.10).
MOVE_TIMEOUTS = ("Submission HTTP request failed: TimeoutError", "Submission HTTP request failed: timeout")
LATE_MOVE = "move_timeout"  # record reason of a game the candidate forfeited with a response over the move deadline
LATE = object()  # an /act request whose response missed the move deadline


def late_move(error, *, local_move_timer):
    """Whether a failed /act request missed the move deadline, which forfeits its batch's games instead of ending the
    evaluation: the trusted move client reports its own deadline as TimeoutError, and public validation's HTTP client
    times out at --move-seconds."""
    return (str(error).startswith(MOVE_TIMEOUTS)
            or (not local_move_timer and isinstance(error, TimeoutError) and not transport_fault(error)))
MOVE_SECONDS = 1.0
DEVELOPMENT_SEED = 91827461  # Reproducible operator probes; validation draws fresh seeds.
SHOWDOWN_REVISION = "ea822f8863407f7a16995a9db3519f8e250237c1"
FORMAT = "gen9randombattle"
FOUL_PLAY_REVISION = "6c467c081e862fb321adb405355beb41aba8e226"
FOUL_PLAY_ENGINE = "0.0.48"
FOUL_PLAY_WORKERS = 16
RUNTIME_PROTOCOL = 6  # 6: an illegal Foul Play choice (action -1) forfeits the game to the candidate
RUNTIME_FILES = {"showdown.tar.gz", "simulator.js", "foul-play.tar.gz", "foul_play_worker.py", "foul-play-priors.json"}


def foul_play_python():
    default = Path("/opt/frontier-foul-play/bin/python")
    return os.environ.get("FOUL_PLAY_PYTHON", str(default) if default.is_file() else sys.executable)


def check_foul_play_dependencies():
    try:
        subprocess.run([foul_play_python(), "-c",
            "import requests, websockets, dateutil, poke_engine; "
            "from importlib.metadata import version; "
            f"assert version('poke-engine') == {FOUL_PLAY_ENGINE!r}"],
            check=True, timeout=15, capture_output=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise VerificationFailure("Install the pinned Foul Play dependencies using public/setup_foul_play.sh; "
                                  "set FOUL_PLAY_PYTHON to its venv/bin/python") from exc


def checked_rows(data_path, expected=None, *, require_fixed=False):
    rows = load_samples(data_path)
    if not rows or (expected is not None and len(rows) != expected):
        raise VerificationFailure("Wrong battle schedule length")
    pairs = {}
    fixed = "scenario" in rows[0]
    if require_fixed and not fixed:
        raise VerificationFailure("Final evaluation requires stored teams and battle seeds")
    for row in rows:
        fields = {"id", "pair", "seat", "opponent"} | ({"scenario"} if fixed else set())
        if (set(row) != fields or type(row["pair"]) is not int or row["pair"] < 0
                or type(row["seat"]) is not int or row["seat"] not in (0,1) or row["opponent"] != OPPONENT):
            raise VerificationFailure("Invalid battle schedule row")
        if fixed:
            c = row["scenario"]
            if (not isinstance(c, dict) or set(c) != {"teams", "seed", "policy_seed"}
                    or not isinstance(c["teams"], list) or len(c["teams"]) != 2
                    or any(not isinstance(t, str) or len(t.split("]")) != 6
                           or not 0 < len(t) <= 16384 for t in c["teams"])
                    or not isinstance(c["seed"], list) or len(c["seed"]) != 4
                    or any(type(v) is not int or not 0 <= v < 65536 for v in c["seed"])
                    or type(c["policy_seed"]) is not int or not 0 <= c["policy_seed"] < 2**32):
                raise VerificationFailure("Invalid stored teams or battle seeds")
        pairs.setdefault(row["pair"], []).append(row)
    for pair in pairs.values():
        if len(pair) != 2 or {r["seat"] for r in pair} != {0,1} or len({r["opponent"] for r in pair}) != 1:
            raise VerificationFailure("Schedule needs paired seats against Foul Play")
        if fixed and pair[0]["scenario"] != pair[1]["scenario"]:
            raise VerificationFailure("Paired games must reuse identical teams and battle seeds")
    return rows


def check_runtime(data_path):
    root = Path(data_path).parent
    if not shutil.which("node"):
        raise VerificationFailure("Install Node.js >=18 to run the pinned battle simulator")
    version = subprocess.check_output(["node", "--version"], text=True, timeout=5).strip()
    if int(version.removeprefix("v").split(".")[0]) < 18:
        raise VerificationFailure("Install Node.js >=18 to run the pinned battle simulator")
    lock = json.loads((root / "runtime.json").read_text())
    if (lock.get("showdown_revision") != SHOWDOWN_REVISION or lock.get("format") != FORMAT
            or lock.get("foul_play_revision") != FOUL_PLAY_REVISION
            or lock.get("poke_engine_version") != FOUL_PLAY_ENGINE
            or set(lock.get("files",{})) != RUNTIME_FILES):
        raise VerificationFailure("Wrong simulator runtime lock")
    if lock.get("protocol", 1) != RUNTIME_PROTOCOL:
        raise VerificationFailure(f"Battle runtime needs protocol {RUNTIME_PROTOCOL} (fixed evaluation teams); "
                                  "prepare and publish the updated payload")
    for name, expected in lock["files"].items():
        file = root / name
        if file.is_symlink() or not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise VerificationFailure("Simulator runtime hash mismatch: " + name)
    return root


class Engine:
    """Only pinned organizer code is executed here, never submission code."""
    def __init__(self, root, work, deadline, command=None):
        self.deadline = deadline
        self.buffer = b""
        self.stderr = tempfile.TemporaryFile()
        self.process = subprocess.Popen(command or ["node", str(root / "simulator.js"), str(work)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, bufsize=0, start_new_session=True)

    def call(self, message):
        raw = json.dumps(message, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        if len(raw) > 16 * 1024 * 1024:
            raise VerificationFailure("Simulator command exceeds its size cap")
        # Writes and reads both have a hard deadline, including a stalled subprocess.
        fd = self.process.stdin.fileno()
        while raw:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0 or not select.select([], [fd], [], remaining)[1]:
                raise VerificationFailure("Battle simulation deadline exceeded")
            count = os.write(fd, raw[:4096])
            raw = raw[count:]
        while b"\n" not in self.buffer:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise VerificationFailure("Battle simulation deadline exceeded")
            part = os.read(self.process.stdout.fileno(), 65536)
            if not part:
                self.stderr.seek(0)
                raise EvaluationError("Organizer simulator/worker stopped: " + self.stderr.read(8000).decode(errors="replace"))
            self.buffer += part
            if len(self.buffer) > 16 * 1024 * 1024:
                raise VerificationFailure("Simulator response exceeds its size cap")
        line, self.buffer = self.buffer.split(b"\n", 1)
        response = json.loads(line)
        if "error" in response:
            raise EvaluationError("Organizer simulator/worker error: " + response["error"][:8000])
        return response

    def close(self):
        # Kill the whole worker group, including upstream search subprocesses.
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=2)
        self.process.stdin.close()
        self.process.stdout.close()
        self.stderr.close()


@contextmanager
def engine(data_path, deadline):
    root = check_runtime(data_path)
    check_foul_play_dependencies()
    with tempfile.TemporaryDirectory(prefix="frontier-random-battle-") as directory:
        work = Path(directory)
        for name, target in [("showdown.tar.gz",work), ("foul-play.tar.gz",work / "foul-play")]:
            with tarfile.open(root / name, "r:gz") as archive:
                members = archive.getmembers()
                if any(not m.isfile() or Path(m.name).is_absolute() or ".." in Path(m.name).parts for m in members):
                    raise VerificationFailure("Runtime archive must contain only relative regular files")
                if sum(m.size for m in members) > 128 * 1024 * 1024:
                    raise VerificationFailure("Runtime archive exceeds extracted size cap")
                archive.extractall(target, filter="data")
        process = Engine(root, work, deadline)
        try:
            yield process
        finally:
            process.close()


class FoulPlay:
    def __init__(self, root, work, deadline, *, workers=None):
        workers = FOUL_PLAY_WORKERS if workers is None else workers
        if type(workers) is not int or not 1 <= workers <= FOUL_PLAY_WORKERS:
            raise ValueError(f"Opponent workers must be between 1 and {FOUL_PLAY_WORKERS}")
        self.workers = []
        try:
            for _ in range(workers):
                self.workers.append(Engine(root, work, deadline, [foul_play_python(),
                    str(root / "foul_play_worker.py"), str(work / "foul-play"), str(root / "foul-play-priors.json")]))
        except BaseException:
            self.close()
            raise
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def reset(self, games=None):
        # Slots keep worker affinity until completion. Clear only completed games
        # so a replacement never inherits an earlier game's parser or RNG state.
        groups = [[] for _ in self.workers]
        if games is not None:
            for game in games:
                groups[game % len(groups)].append(game)
        pending = [self.pool.submit(worker.call, {"op":"reset", **({"games":group} if games is not None else {})})
                   for worker, group in zip(self.workers, groups, strict=True) if games is None or group]
        for future in pending:
            future.result()

    def choose(self, items, configs):
        groups = [[] for _ in self.workers]
        for position, item in enumerate(items):
            groups[item["game"] % len(groups)].append((position, {
                **item, "policy_seed":configs[item["game"]]["policy_seed"]}))
        pending = [(group, self.pool.submit(worker.call, {"op":"act", "items":[item for _,item in group]}))
                   for worker, group in zip(self.workers, groups, strict=True) if group]
        actions = [None] * len(items)
        for group, future in pending:
            response = future.result()
            parsed = parse_actions(response, [item["observation"] for _,item in group], allow_forfeit=True)
            for (position,_), action in zip(group, parsed, strict=True):
                actions[position] = action
        return actions

    def close(self):
        for worker in self.workers:
            worker.close()
        if hasattr(self, "pool"):
            self.pool.shutdown(wait=True, cancel_futures=True)


OPPONENT_FORFEIT = -1  # foul_play_worker.FORFEIT: the opponent made an illegal choice and loses the game


def parse_actions(response, observations, *, allow_forfeit=False):
    """Candidate and opponent action indices into each legal menu; only the organizer opponent may forfeit (-1)."""
    if not isinstance(response, dict) or set(response) != {"actions"}:
        raise VerificationFailure("Expected exactly an actions array")
    actions = response["actions"]
    if not isinstance(actions, list) or len(actions) != len(observations):
        raise VerificationFailure("Expected one action per observation, in order")
    for index, observation in zip(actions, observations, strict=True):
        if allow_forfeit and type(index) is int and index == OPPONENT_FORFEIT:
            continue
        if type(index) is not int or not 0 <= index < len(observation["legal_actions"]):
            raise VerificationFailure("Action index is outside the supplied legal menu")
    return actions


def candidate_batch(state, forfeited):
    """The slots and observations to send: every game asking the candidate for a decision except those it already
    forfeited (a forfeited game can need one more round before the simulator ends it)."""
    pairs = [(slot, observation) for slot, observation in zip(state["indices"], state["observations"], strict=True)
             if slot not in forfeited]
    return [slot for slot, _ in pairs], [observation for _, observation in pairs]


def step_message(state, slots, actions, opponent_actions, forfeited):
    """The simulator step for the candidate's answers to `slots`. A forfeited game gets Foul Play's -1, which ends it in
    the pinned simulator (the caller records it as a candidate loss), and a placeholder candidate action 0, applied only
    in a round where Foul Play has no decision pending in that game."""
    chosen = dict(zip(slots, actions, strict=True))
    return {"op": "step", "actions": [chosen.get(slot, 0) for slot in state["indices"]],
            "opponent_actions": [OPPONENT_FORFEIT if item["game"] in forfeited else action
                                 for item, action in zip(state["opponents"], opponent_actions, strict=True)]}


def summarize(records, elapsed, request_times, seed):
    if not records:
        raise VerificationFailure("Cannot score an empty battle schedule")
    pairs = {}
    for record in records:
        point = record["points"]
        if type(point) not in (float,int) or point not in (0,0.5,1):
            raise VerificationFailure("Invalid simulator outcome")
        pairs.setdefault(record["pair"], []).append(point)
    pair_points = [statistics.mean(points) for points in pairs.values()]
    return {"score": statistics.mean(r["points"] for r in records), "details": {
        "metric":"mean_battle_points", "samples":len(records), "records":records,
        "wins":sum(r["points"] == 1 for r in records), "draws":sum(r["points"] == 0.5 for r in records),
        "losses":sum(r["points"] == 0 for r in records), "capped_games":sum(r["reason"] == "cap" for r in records),
        "opponent_forfeits":sum(r["reason"] == "opponent_illegal" for r in records),
        "move_timeout_forfeits":sum(r["reason"] == LATE_MOVE for r in records),
        "opponent":{"name":OPPONENT,"revision":FOUL_PLAY_REVISION,"poke_engine":FOUL_PLAY_ENGINE,
                    "search_time_ms":100,"search_parallelism":1,"search_threads":1,"workers":FOUL_PLAY_WORKERS,
                    "revival_policy":"native-search-afterstates-shared-budget"},
        "paired_standard_error":statistics.stdev(pair_points)/math.sqrt(len(pair_points)) if len(pair_points)>1 else 0,
        "evaluation_seed":seed, "timing":{"elapsed_seconds":elapsed,"requests":len(request_times),
            "max_request_seconds":max(request_times,default=0),"deadline_seconds":BATCH_SECONDS,
            "execution_budget_seconds":EXECUTION_SECONDS}}}


def evaluate(transport, data_path, *, seed, expected=None, seconds=BATCH_SECONDS, baseline=None, observer=None,
             local_move_timer=False, opponent_workers=None, parallel_decisions=True, move_seconds=MOVE_SECONDS):
    seconds = float(seconds)
    if not math.isfinite(seconds) or seconds <= 0:
        raise VerificationFailure("Evaluation timeout must be positive and finite")
    if (not math.isfinite(move_seconds) or not 0 < move_seconds <= 30
            or (local_move_timer and move_seconds != MOVE_SECONDS)):
        raise VerificationFailure("Development move timeout must be in (0, 30]; final timing is fixed at 1 second")
    seconds = min(BATCH_SECONDS, seconds)
    started = time.monotonic()
    deadline = started + seconds
    rows = checked_rows(data_path, expected, require_fixed=seed is None)
    configs = list(scenarios(rows, seed))
    # Mix matchups and player seats; internal game IDs/seeds never reach the endpoint.
    jobs = list(zip(rows, configs, strict=True))
    random.Random(0 if seed is None else seed + 1).shuffle(jobs)
    records, request_times, batch_sizes, transport_retries, late_moves = [], [], [], [], []
    forfeited = set()  # slots whose game the candidate forfeited with a late response, until the simulator ends it
    phases = {"setup_seconds":0.0, "opponent_seconds":0.0, "opponent_reset_seconds":0.0,
              "candidate_seconds":0.0, "simulator_seconds":0.0, "decision_wall_seconds":0.0}
    with engine(data_path, deadline) as simulator, ExitStack() as cleanup:
        root = Path(data_path).parent.resolve()
        work = Path(simulator.process.args[-1])
        decisions = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pokemon-opponent")
        cleanup.callback(decisions.shutdown, wait=True, cancel_futures=True)
        opponent = (FoulPlay(root, work, deadline) if opponent_workers is None else
                    FoulPlay(root, work, deadline, workers=opponent_workers))
        # On endpoint failure kill opponent processes before joining the decision
        # thread, rather than waiting for a stalled search until the full deadline.
        cleanup.callback(opponent.close)
        opponent.reset()  # Wait for all workers to load before measuring decisions.
        phases["setup_seconds"] = time.monotonic() - started

        def simulate(message):
            before = time.monotonic()
            result = simulator.call(message)
            phases["simulator_seconds"] += time.monotonic() - before
            return result

        def choose_opponent(items, configs):
            before = time.monotonic()
            actions = opponent.choose(items, configs)
            return actions, time.monotonic() - before

        active = dict(enumerate(jobs[:MAX_BATCH]))
        queued = iter(jobs[MAX_BATCH:])
        state = simulate({"op":"init", "games":[c for _,c in active.values()]})
        while active:
            if time.monotonic() >= deadline:
                raise VerificationFailure("Battle evaluation deadline exceeded")
            finished = [slot for slot in active if state["results"][slot] is not None]
            replacements = []
            for slot in finished:
                row, _ = active.pop(slot)
                outcome = state["results"][slot]
                if slot in forfeited:  # ended by the -1 sent for it, or naturally: either way a loss
                    forfeited.discard(slot)
                    outcome = {**outcome, "points":0, "reason":LATE_MOVE}
                records.append({**{k:v for k,v in row.items() if k != "scenario"}, **outcome})
                job = next(queued, None)
                if job is not None:
                    active[slot] = job
                    replacements.append({"slot":slot, "config":job[1]})
            if finished:
                before = time.monotonic()
                opponent.reset(finished)
                phases["opponent_reset_seconds"] += time.monotonic() - before
            if replacements:
                state = simulate({"op":"refill", "games":replacements})
            if not active:
                break
            slots, observations = candidate_batch(state, forfeited)
            payload = {"observations":observations}
            if len(observations) > MAX_BATCH or len(json.dumps(payload).encode()) > MAX_REQUEST_BYTES:
                raise VerificationFailure("Observation batch exceeds the request size cap")
            before = time.monotonic()
            # Both decisions use the same pre-turn state. Only the simulator sees
            # both actions, after both policies have finished; HTTP concurrency is 1.
            pending = decisions.submit(choose_opponent, state["opponents"],
                                       {slot:c for slot,(_,c) in active.items()})
            if not parallel_decisions:
                # Public development on a shared small CPU host: finish opponent
                # search before timing the endpoint so they do not compete for CPU.
                pending.result(timeout=max(0, deadline-time.monotonic()))
            actions = []  # A forced replacement can require only the opponent.
            if observations:
                batch_sizes.append(len(observations))
                candidate_started = time.monotonic()
                if baseline:
                    actions = simulate({"op":"baseline", "style":baseline})["actions"]
                else:
                    # The trusted H100-local timer bounds the complete HTTP response (public validation: the HTTP
                    # client's own timeout and the measured interval); this outer limit also bounds control-plane RPCs.
                    remaining = min(REQUEST_SECONDS, deadline-time.monotonic())
                    if remaining <= 0:
                        raise VerificationFailure("Battle evaluation deadline exceeded")
                    def request(_, timeout):
                        started_request = time.monotonic()
                        timeout = timeout if local_move_timer else min(timeout, move_seconds)
                        try:
                            try:
                                response = transport.request("POST", "/act", payload, timeout=timeout)
                            except Exception as error:
                                left = timeout - (time.monotonic() - started_request)
                                if not transport_fault(error) or left <= 0:
                                    raise
                                transport_retries.append(str(error)[:300])
                                response = transport.request("POST", "/act", payload, timeout=left)
                        except Exception as error:
                            if late_move(error, local_move_timer=local_move_timer):
                                return LATE
                            raise
                        if not local_move_timer and time.monotonic() - started_request > move_seconds:
                            return LATE  # complete, but after the public move deadline
                        return parse_actions(response, observations)
                    result, timing = timed_batch([None], request, concurrency=1, timeout_seconds=remaining)
                    request_times.append(timing["elapsed_seconds"])
                    if result[0] is LATE:  # every game in this batch is forfeited; the evaluation goes on
                        forfeited.update(slots)
                        late_moves.append({"request":len(request_times), "games":len(slots),
                                           "seconds":round(timing["elapsed_seconds"], 3)})
                        slots = []
                    else:
                        actions = result[0]
                if slots:
                    parse_actions({"actions":actions}, observations)
                phases["candidate_seconds"] += time.monotonic() - candidate_started
            try:
                opponent_actions, duration = pending.result(timeout=max(0, deadline-time.monotonic()))
            except TimeoutError as exc:
                raise VerificationFailure("Battle evaluation deadline exceeded") from exc
            phases["opponent_seconds"] += duration
            phases["decision_wall_seconds"] += time.monotonic() - before
            if slots and observer:
                observer(observations, actions)
            state = simulate(step_message(state, slots, actions, opponent_actions, forfeited))
    elapsed = time.monotonic()-started
    if elapsed > min(BATCH_SECONDS, seconds):
        raise VerificationFailure("Battle evaluation deadline exceeded")
    records.sort(key=lambda row: row["id"])
    result = summarize(records,elapsed,request_times,seed)
    result["details"]["matchups"] = {
        "source": "stored_teams" if "scenario" in rows[0] else "development_seed",
        "schedule_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest()}
    result["details"]["opponent"].update(
        workers=FOUL_PLAY_WORKERS if opponent_workers is None else opponent_workers,
        move_index_policy="truncate-before-native-index")
    result["details"]["timing"].update(phases, scheduling="rolling", max_active_games=MAX_BATCH,
        deadline_seconds=seconds,
        opponent_workers=FOUL_PLAY_WORKERS if opponent_workers is None else opponent_workers,
        parallel_decisions=parallel_decisions, request_limit_seconds=REQUEST_SECONDS if local_move_timer else move_seconds,
        transport_retries=len(transport_retries), transport_retry_errors=transport_retries[:10],
        late_requests=len(late_moves), late_moves=late_moves[:100],
        # Where each /act request's wall time went (verifier clock per phase, in-sandbox server time, regions).
        transport_breakdown=transport.transport_breakdown() if hasattr(transport, "transport_breakdown") else None,
        observations=sum(batch_sizes), mean_batch_size=statistics.mean(batch_sizes) if batch_sizes else 0,
        max_batch_size=max(batch_sizes, default=0))
    return result
