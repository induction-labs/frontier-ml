"""Prepare public schedules and freeze generated held-out teams; no uploads."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
SEED = HERE.parent
sys.path.insert(0, str(SEED))

SHOWDOWN_REVISION = "ea822f8863407f7a16995a9db3519f8e250237c1"
FOUL_PLAY_REVISION = "6c467c081e862fb321adb405355beb41aba8e226"
from pokemon_random_battle_teams import scenarios, schedule
from pokemon_random_battle import FINAL_GAMES, RUNTIME_PROTOCOL, checked_rows
from review import ENGINE_SOURCE, ENGINE_SOURCE_SHA256, ENGINE_WHEEL, ENGINE_WHEEL_SHA256


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def persistent_seed(path):
    """Create once outside the payload; rebuilds must reuse this seed file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, "w") as stream:
            stream.write(str(secrets.randbits(256)) + "\n")
    seed = int(path.read_text().strip())
    if not 0 <= seed < 2**256:
        raise ValueError("Expected a persisted 256-bit nonnegative seed")
    return seed


def frozen_rows(showdown, count, seed, prefix="test"):
    """Materialize full teams once per pair with the unmodified pinned generator."""
    rows = schedule(count, prefix)
    configs = list(scenarios(rows, seed))
    script = """
      const {Teams}=require(require('path').resolve(process.argv[1], 'dist/sim'));
      const configs=JSON.parse(require('fs').readFileSync(0, 'utf8'));
      const frozen=configs.map(c=>({
        teams:c.team_seeds.map(seed=>Teams.pack(Teams.generate('gen9randombattle', {seed}))),
        seed:c.seed, policy_seed:c.policy_seed
      }));
      process.stdout.write(JSON.stringify(frozen));
    """
    result = subprocess.run(["node", "-e", script, str(showdown)],
                            input=json.dumps(configs[::2]), capture_output=True, text=True,
                            check=True, timeout=120)
    pairs = json.loads(result.stdout)
    if len(pairs) != count // 2:
        raise ValueError("Wrong number of generated matchup pairs")
    return [{**row, "scenario": pairs[row["pair"]]} for row in rows]


def prepare(output, showdown, foul_play, priors, engine_build, test_seed_file):
    output, showdown, foul_play, priors, engine_build = map(Path, (output, showdown, foul_play, priors, engine_build))
    if output.exists():
        raise FileExistsError("Use a new output directory; preparation never overwrites data")
    if Path(test_seed_file).resolve().is_relative_to(output.resolve()):
        raise ValueError("Keep the preparation seed file outside the uploaded payload")
    for name, expected in ((ENGINE_WHEEL, ENGINE_WHEEL_SHA256), (ENGINE_SOURCE, ENGINE_SOURCE_SHA256)):
        if digest(engine_build / name) != expected:
            raise ValueError("Engine artifact differs from the operator audit: " + name)
    if (engine_build / "engine-build.json").read_bytes() != (HERE / "engine-build.json").read_bytes():
        raise ValueError("Engine build receipt differs from the audited receipt")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=showdown, text=True).strip()
    if revision != SHOWDOWN_REVISION:
        raise ValueError("Wrong Showdown revision")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=foul_play, text=True).strip() != FOUL_PLAY_REVISION:
        raise ValueError("Wrong Foul Play revision")
    if subprocess.check_output(["git", "diff", "HEAD"], cwd=foul_play):
        raise ValueError("Modified Foul Play source")
    # Retain every tracked upstream source and license file, not just imported files.
    source_names = subprocess.check_output(["git", "ls-files", "-z"], cwd=foul_play).decode().split("\0")
    fp_buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=fp_buffer, mode="wb", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as archive:
            for name in sorted(filter(None, source_names)):
                raw = (foul_play / name).read_bytes()
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(raw), 0o644, 0
                archive.addfile(info, io.BytesIO(raw))
    prior_data = json.loads(priors.read_text())
    if sum(sum(sets.values()) for sets in prior_data.values()) != 600000:
        raise ValueError("Generate the prior using foul_play_priors.js (100,000 public development teams)")
    if subprocess.check_output(["git", "diff", "HEAD", "--", "sim", "data", "lib", "config", "package.json", "package-lock.json", "build"], cwd=showdown):
        raise ValueError("Modified Showdown source")
    files = []
    for base in ("dist/sim", "dist/data", "dist/lib", "dist/config", "node_modules/ts-chacha20"):
        files += [p for p in (showdown / base).rglob("*") if p.is_file() and
                  (p.suffix in (".js", ".json") or p.name == "LICENSE")]
    files += [showdown / "package.json", showdown / "LICENSE"]
    if not (showdown / "dist/sim/index.js").is_file():
        raise ValueError("Build the pinned simulator first (npm ci --omit=dev --ignore-scripts; node build)")
    # Deterministic archive: pure JS only, no native modules or build tools.
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as archive:
            for path in sorted(set(files)):
                raw = path.read_bytes()
                info = tarfile.TarInfo(path.relative_to(showdown).as_posix())
                info.size, info.mode, info.mtime = len(raw), 0o644, 0
                archive.addfile(info, io.BytesIO(raw))
    runtime = buffer.getvalue()
    test_seed = persistent_seed(test_seed_file)
    private_rows = frozen_rows(showdown, FINAL_GAMES, test_seed)
    private_teams = {team for row in private_rows for team in row["scenario"]["teams"]}
    if len(private_teams) != FINAL_GAMES:
        raise ValueError("Evaluation teams overlap another matchup")
    output.mkdir(parents=True)
    for split, count, name in [("public",32,"validation"),("private",FINAL_GAMES,"test")]:
        root = output / split
        root.mkdir()
        rows = private_rows if split == "private" else schedule(count, name)
        schedule_path = root / f"{name}.jsonl"
        schedule_path.write_text("".join(json.dumps(r)+"\n" for r in rows))
        checked_rows(schedule_path, expected=count, require_fixed=split == "private")
        (root / "showdown.tar.gz").write_bytes(runtime)
        (root / "foul-play.tar.gz").write_bytes(fp_buffer.getvalue())
        shutil.copyfile(priors, root / "foul-play-priors.json")
        shutil.copyfile(SEED / "foul_play_worker.py", root / "foul_play_worker.py")
        shutil.copyfile(SEED / "pokemon_random_battle_sim.js", root / "simulator.js")
        (root / "runtime.json").write_text(json.dumps({"protocol":RUNTIME_PROTOCOL,"showdown_revision":revision,"format":"gen9randombattle",
            "foul_play_revision":FOUL_PLAY_REVISION,"poke_engine_version":"0.0.48",
            "revival_policy":"native-search-afterstates-shared-budget",
            "move_index_policy":"truncate-before-native-index",
            "speed_inference_policy":"skip-selected-switch",
            "form_change_policy":"defer-intermediate-request-refresh",
            "matchup_policy":"stored-teams-and-seeds",
            "opponent_illegal_policy":"forfeit-game-to-candidate",
            "files":{n:digest(root/n) for n in ("showdown.tar.gz","simulator.js","foul-play.tar.gz",
                "foul-play-priors.json","foul_play_worker.py")}},indent=2)+"\n")
    for source in (HERE / "SCHEMA.md", HERE / "NOTICE.md", HERE / "foul_play_priors.js",
                   SEED / "foul-play-requirements.txt", SEED / "setup_foul_play.sh"):
        shutil.copyfile(source, output / "public" / source.name)
    shutil.copytree(HERE / "example_submission", output / "public/starter",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
    (output / "public/engine").mkdir()
    for name in (ENGINE_WHEEL, ENGINE_SOURCE, "engine-build.json"):
        shutil.copyfile(engine_build / name, output / "public/engine" / name)
    manifest = {"schema_version":1,"task":"pokemon-random-battle","source":{"repo":"smogon/pokemon-showdown",
        "revision":revision,"license":"MIT","generator":"pinned gen9randombattle random-team generator; frozen full teams and seeds"},
        "evaluation":{"policy":"stored-teams-and-seeds","pairs":FINAL_GAMES//2,"distinct_teams":FINAL_GAMES,
            "preparation_seed_sha256":digest(test_seed_file),
            "schedule_sha256":digest(output / "private/test.jsonl")},
        "opponent":{"repo":"pmariglia/foul-play","revision":FOUL_PLAY_REVISION,"license":"GPL-3.0",
            "poke_engine":"0.0.48","search_time_ms":100,"parallelism":1,"threads":1,
            "revival_policy":"native-search-afterstates-shared-budget",
            "move_index_policy":"truncate-before-native-index",
            "speed_inference_policy":"skip-selected-switch",
            "form_change_policy":"defer-intermediate-request-refresh",
            "prior":"100,000 public generator teams, seed [179,421,991,2027], no test samples"},
        "splits":{"validation":32,"heldout":FINAL_GAMES},"split_audit":{"passed":True,
        "validation_policy":"fresh independent team and battle seeds on every invocation",
        "limitations":"Final teams and battle seeds are fixed in private/test.jsonl, excluded from the agent payload. "
        "The HF repository is public: private/ is a harness partition, not an access restriction. "
        "Review prohibits acquiring or using held-out teams. Mechanics and upstream sets are intentionally public. "
        "Validation draws independently at runtime; exact random collisions are not explicitly filtered. "
        "Time-based search can still change decisions and trajectories. "
        "This is not a split of human replays."},
        "files":{p.relative_to(output).as_posix():{"bytes":p.stat().st_size,"sha256":digest(p)}
                 for p in sorted(output.rglob('*')) if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--showdown", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--foul-play", required=True, type=Path)
    parser.add_argument("--priors", required=True, type=Path)
    parser.add_argument("--engine-build", required=True, type=Path,
                        help="directory containing the organizer-audited wheel, source and build receipt")
    parser.add_argument("--test-seed-file", required=True, type=Path,
                        help="persistent seed outside the payload; created once, reused on every rebuild")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output,args.showdown,args.foul_play,args.priors,args.engine_build,args.test_seed_file),indent=2))
