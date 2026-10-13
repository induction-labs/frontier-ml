"""`frontier`: the agent sandbox's command for self-checks and the submission (standard library only).

Shipped in the agent image as the `frontier_agent` package (agent_surface.stage_client). It talks to the Harbor
plugin only through files under SPOOL (selfcheck_service.py): it drops a request file and reads the root-owned status,
progress and report files the plugin writes. It holds no credentials, and reads /workspace/output only to count its
files for `frontier submit`'s confirmation question.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import sys
import time

SPOOL = Path(os.environ.get("FRONTIER_SELFCHECK_SPOOL", "/var/lib/frontier/selfcheck"))
OUTPUT = Path("/workspace/output")
# `frontier submit` records here that it asked for confirmation; `--confirm` is refused without it. Agent-writable:
# it only keeps a submission from happening by accident.
ASKED = Path("/tmp/.frontier-submit-asked")
SUBMIT_PREFIX = "submit-"  # request names the plugin's watcher reports as submissions (selfcheck_service.SUBMIT_PREFIX)
WAIT_SECONDS = 540  # under the usual 10-minute tool limit; `frontier selfcheck wait` continues
POLL_SECONDS = 5
PICKUP_SECONDS = 60  # the plugin's watcher looks every second, but restarts with a backoff after a dropped stream
START_STATUS_SECONDS = 30  # the plugin publishes status.json shortly after the agent's sandbox starts
STATUS_SECONDS = 5
EXIT = {"passed": 0, "failed": 1, "unavailable": 2, "running": 3, "error": 4, "cancelled": 4}
STAGES = {
    "copying": "copying /workspace/output",
    "starting the self-check sandbox": "starting the self-check sandbox",
    "setup": "fetching the public inputs",
    "snapshot": "snapshotting the copy (symlink and file rules)",
    "static": "checking static constraints",
    "provisioning": "waiting for the H100 and copying the submission onto it",
    "training": "running train.sh as the evaluation replays it (a time-limited probe)",
    "startup": "serve.sh started; waiting for GET /health",
    "smoke": "sending public samples",
    "requests": "sending public requests",
    "gates": "measuring gates",
    "done": "writing the report",
}

DESCRIPTION = """Self-check /workspace/output the way it will be graded: the folder is copied and snapshotted with the
grading rules, serve.sh starts on the grading image on one H100 without network access under the startup
deadline, and public samples go through the task's own response checks and speed gates. The report gives the stage
reached, the checked digest, the GPU, the startup time against its deadline and each gate's measured value,
requirement and margin. It gives no score and runs no architecture review. Each run allows a few self-checks
(`frontier selfcheck status`); one that fails for infrastructure reasons, or stops before any GPU is used, is
normally not counted (the report says). A check usually takes 10-40 minutes (`frontier selfcheck status` shows this task's limit) and keeps running
while you work; `frontier selfcheck` waits up to --timeout seconds, then `frontier selfcheck wait` resumes waiting.
One check runs at a time and checks the copy taken when it started; leave /workspace/output unchanged until the
command says the copy is taken.

Exit status: 0 passed, 1 failed, 2 unavailable or none left, 3 still running, 4 error or cancelled."""
COPYING = "is copying /workspace/output now: leave the folder unchanged until the copy is taken"

SUBMIT_DESCRIPTION = """Submit /workspace/output for grading. Submitting is final and ends your session: a few seconds after the
submission is accepted, every process you started is stopped (background jobs included), your Spin machines are
released, and /workspace/output is graded as it is at that moment. You cannot change anything afterwards.

Submitting takes two steps, so that it never happens by accident:
  frontier submit            asks you to confirm and shows what /workspace/output holds; nothing is submitted
  frontier submit --confirm  submits; refused unless `frontier submit` has been run before

If you do not submit, /workspace/output is graded as it is when your session ends, at the latest when your time
runs out. Before submitting, make sure the folder holds everything the evaluation needs (`frontier selfcheck`
tests it the way it will be graded); a self-check that is still running when you submit is stopped.

Exit status: 0 submitted, 1 not submitted (asked to confirm, or --confirm before `frontier submit`), 2 not
available in this run or already submitted, 4 the submission service did not answer."""
NOT_SUBMITTABLE = ("submitting is not available in this run: /workspace/output is graded as it is when your session "
                   "ends.")


def read_json(path):
    """A JSON object the plugin wrote (always by rename, so never partial), or None while there is none."""
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def status(wait=0):
    deadline = time.monotonic() + wait
    while True:
        value = read_json(SPOOL / "status.json")
        if value is not None or time.monotonic() >= deadline:
            return value
        time.sleep(1)


def unavailable(reason, as_json):
    print(f"frontier selfcheck: {reason}", file=sys.stderr)
    if as_json:
        print(json.dumps({"status": "unavailable", "reason": reason}, indent=2))
    return EXIT["unavailable"]


def say(text, as_json):
    """Progress for the reader; with --json it goes to stderr so that stdout holds exactly one JSON document."""
    print(text, file=sys.stderr if as_json else sys.stdout, flush=True)


def pick(value, check_id):
    """The named check (an ID prefix is enough), else the running one, else the latest."""
    checks = value.get("checks") or []
    if check_id:
        matches = [c for c in checks if c["id"] == check_id] or [c for c in checks if c["id"].startswith(check_id)]
        return matches[-1] if matches else None
    running = value.get("running")
    return next((c for c in checks if c["id"] == running), None) or (checks[-1] if checks else None)


def describe(progress):
    stage = progress.get("stage")
    text = STAGES.get(stage) or str(stage or "starting").replace("_", " ")  # task stages such as proof_checking
    if isinstance(progress.get("completed"), int) and isinstance(progress.get("total"), int):
        text += f" ({progress['completed']}/{progress['total']})"
    message = str(progress.get("message") or "").strip()
    if message:
        text += ": " + message.splitlines()[0][:200]
    return text


def number(value):
    return f"{value:g}" if isinstance(value, float) else str(value)


def size(value):
    for unit, scale, digits in (("GB", 1e9, 2), ("MB", 1e6, 1), ("KB", 1e3, 1)):
        if value >= scale:
            return f"{value / scale:.{digits}f} {unit}"
    return f"{value} bytes"


def render(report, value):
    outcome = str(report.get("status") or "?")
    lines = [f"Self-check {str(report.get('check_id') or '')[:12]}: {outcome.upper()}"
             + (f" at stage {report['stage']}" if outcome != "passed" and report.get("stage") else "")]
    if report.get("submission_digest"):
        total = report.get("submission_bytes")
        lines.append(f"Checked {report.get('submission_files')} files"
                     + (f", {size(total)}" if isinstance(total, int) else "")
                     + f", digest {report['submission_digest']}")
    for instance in report.get("instances") or []:
        gpu = (instance.get("gpu") or {}).get("name") if isinstance(instance.get("gpu"), dict) else instance.get("gpu")
        lines.append(f"GPU: {gpu or 'unknown'}")
    for entry in report.get("checks") or []:
        mark = "WARN" if entry.get("warning") else "pass" if entry.get("passed") else "FAIL"
        text = f"  [{mark}] {entry.get('name')}"
        if "measured" in entry:
            text += f": {number(entry['measured'])}{(' ' + entry['unit']) if entry.get('unit') else ''}"
        if "required" in entry:
            text += f" (required {number(entry['required'])}"
            if isinstance(entry.get("margin"), (int, float)):
                margin = entry["margin"]
                text += f", {abs(margin):.0%} {'to spare' if margin >= 0 else 'short'}"
            text += ")"
        if entry.get("detail"):
            text += f"\n         {entry['detail']}"
        lines.append(text)
    if report.get("error"):
        lines.append("Error: " + report["error"])
    if report.get("server_log_tail"):
        lines.append("End of the server log:\n" + report["server_log_tail"].rstrip())
    for note in report.get("notes") or []:
        lines.append("Note: " + note)
    # The report's count dates from when that check finished; status.json is current once it records the outcome
    # (the plugin publishes it just after the report).
    value = value or {}
    entry = next((c for c in value.get("checks") or [] if c.get("id") == report.get("check_id")), {})
    remaining = report.get("remaining_selfchecks")
    if value.get("remaining") is not None and (remaining is None or entry.get("state") not in (None, "running")):
        remaining = value["remaining"]
    if remaining is not None:
        limit = value.get("limit")
        lines.append(f"Self-checks left in this run: {remaining}" + (f" of {limit}." if limit else "."))
    return "\n".join(lines)


def running(check_id, progress, as_json):
    """A check without a report yet; with --json its progress, as one JSON document."""
    if as_json:
        print(json.dumps({"check_id": check_id, "status": "running", "progress": progress}, indent=2))
    else:
        print(f"Self-check {check_id[:12]} is still running ({describe(progress)}). It continues while you work; "
              "run `frontier selfcheck wait` to keep waiting.", flush=True)
    return EXIT["running"]


def follow(check_id, timeout, as_json, copying=False):
    """Wait for one check, printing its stages; return the exit status. `copying`: the check may still be copying
    /workspace/output, so say when it moves on."""
    started = time.monotonic()
    last = None
    while True:
        report = read_json(SPOOL / "checks" / check_id / "report.json")
        if report is not None:
            print(json.dumps(report, indent=2) if as_json else render(report, status()))
            return EXIT.get(report.get("status"), EXIT["error"])
        progress = read_json(SPOOL / "checks" / check_id / "progress.json") or {}
        text = describe(progress)
        elapsed = int(time.monotonic() - started)
        clock = f"[{elapsed // 60:02d}:{elapsed % 60:02d}]"
        if copying and progress.get("stage") not in (None, "copying"):
            say(f"{clock} copy of /workspace/output taken; you can keep editing (later changes are not checked)",
                as_json)
            copying = False
        if text != last and progress.get("stage"):
            say(f"{clock} {text}", as_json)
            last = text
        if time.monotonic() - started >= timeout:
            return running(check_id, progress, as_json)
        time.sleep(POLL_SECONDS)


def attach(check_id, timeout, as_json):
    # Said up front: an agent that edited /workspace/output since would otherwise trust a report on older files.
    progress = read_json(SPOOL / "checks" / check_id / "progress.json") or {}
    copying = progress.get("stage") in (None, "copying")
    say(f"Self-check {check_id[:12]} is already running"
        + (f" and {COPYING}." if copying else ", on the copy of /workspace/output taken when it started.")
        + " Following it; start another once it has finished to check later changes.", as_json)
    return follow(check_id, timeout, as_json, copying)


def time_left(started, timeout):
    """What --timeout leaves after the status read and the pickup."""
    return max(0, timeout - (time.monotonic() - started))


def send(nonce, service, lost):
    """File requests/<nonce>.json for the plugin's watcher and wait for its answer in status.json: the answer, or the
    reason there is none as a string. `service` names the service in that reason; `lost` says where to look when the
    request was taken but its answer did not arrive."""
    request = SPOOL / "requests" / f"{nonce}.json"
    temporary = request.with_name(f".{nonce}.tmp")  # the watcher takes only *.json; renamed in place when complete
    try:
        temporary.write_text(json.dumps({"requested_at": time.time()}))
        temporary.rename(request)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        return f"could not file the request ({type(error).__name__}: {error})."
    deadline = time.monotonic() + PICKUP_SECONDS
    while True:
        answer = ((status() or {}).get("requests") or {}).get(nonce)
        if answer:
            return answer
        if time.monotonic() >= deadline:
            try:
                request.unlink()  # withdrawn, so that a watcher that has not listed it yet does not act on it
            except FileNotFoundError:  # the watcher took it meanwhile; its answer is on the way
                if time.monotonic() < deadline + PICKUP_SECONDS:
                    time.sleep(1)
                    continue
                return f"{service} took the request but did not answer; {lost}"
            return f"{service} did not pick up the request; try again in a minute."
        time.sleep(1)


def start(timeout, as_json):
    started = time.monotonic()
    value = status(wait=START_STATUS_SECONDS)
    if value is None:
        return unavailable("self-checks are not available in this run.", as_json)
    if not value.get("available"):  # closed: no watcher is left to answer a request
        reason = value.get("closed_reason")
        return unavailable("self-checks are closed" + (f": {reason}" if reason else "."), as_json)
    # The running check and the allowance are the plugin's to decide: it attaches the request to the running check
    # or rejects it. status.json can be stale (a publish that failed after a check finished) and must not block.
    answer = send("r" + secrets.token_hex(12), "the self-check service",
                  "`frontier selfcheck status` shows whether a check started.")
    if isinstance(answer, str):
        return unavailable(answer, as_json)
    if answer.get("rejected"):
        return unavailable(answer["rejected"], as_json)
    if not answer.get("check"):
        return unavailable("the self-check service gave an answer this command does not understand.", as_json)
    if answer.get("attached"):
        return attach(answer["check"], time_left(started, timeout), as_json)
    # The plugin answers before it copies: an edit during the copy would be checked half-made.
    say(f"Self-check {answer['check'][:12]} started and {COPYING}.", as_json)
    return follow(answer["check"], time_left(started, timeout), as_json, copying=True)


def show_status(as_json):
    value = status(wait=STATUS_SECONDS)
    if value is None:
        return unavailable("self-checks are not available in this run.", as_json)
    if as_json:
        print(json.dumps(value, indent=2))
        return 0
    print(f"Self-checks: {value.get('remaining')} of {value.get('limit')} left"
          + ("" if value.get("available") else f" (closed: {value.get('closed_reason')})")
          + (f"; running {value['running'][:12]}" if value.get("running") else ""))
    if isinstance(value.get("check_minutes"), int):
        print(f"A check of this task holds the GPU for at most {value['check_minutes']} minutes, plus the wait for an "
              "H100 and the copy.")
    for entry in value.get("checks") or []:
        counted = "" if entry.get("counted", True) else " (not counted)"
        print(f"  {entry['id'][:12]} {entry.get('state')}{counted}: {entry.get('summary') or entry.get('stage')}")
    return 0


def selfcheck(args):
    if args.action == "start":
        return start(args.timeout, args.json)
    if args.action == "status":
        return show_status(args.json)
    started = time.monotonic()
    value = status(wait=STATUS_SECONDS)
    if value is None:
        return unavailable("self-checks are not available in this run.", args.json)
    entry = pick(value, args.check_id)
    if entry is None:
        return unavailable(f"no self-check matches {args.check_id!r}; `frontier selfcheck status` lists them."
                           if args.check_id else "no self-check has run yet.", args.json)
    if args.action == "wait":
        return follow(entry["id"], time_left(started, args.timeout), args.json, entry.get("stage") == "copying")
    report = read_json(SPOOL / "checks" / entry["id"] / "report.json")
    if report is None:
        progress = read_json(SPOOL / "checks" / entry["id"] / "progress.json") or {"stage": entry.get("stage")}
        return running(entry["id"], progress, args.json)
    print(json.dumps(report, indent=2) if args.json else render(report, value))
    return EXIT.get(report.get("status"), EXIT["error"])


def utc(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def submission_closed(value):
    """Why the agent cannot submit now, or None. The plugin publishes `submission_open` (selfcheck_service.py); it is
    closed without a submission when the service could not start, or at the session's end, when no agent process
    is left to ask."""
    if isinstance((value or {}).get("submitted_at"), (int, float)):
        return f"your solution was already submitted at {utc(value['submitted_at'])}; your session is ending."
    if not (value or {}).get("submission_open"):
        return NOT_SUBMITTABLE
    return None


def contents():
    """What /workspace/output holds, in a few words, for the confirmation question."""
    if OUTPUT.is_symlink() or not OUTPUT.is_dir():
        return f"{OUTPUT} is missing or not a directory: there is nothing to grade."
    files = total = 0
    for directory, _, names in os.walk(OUTPUT):
        for name in names:
            try:
                total += os.lstat(os.path.join(directory, name)).st_size
            except OSError:
                continue
            files += 1
    if not files:
        return f"{OUTPUT} is empty: there is nothing to grade."
    return f"{OUTPUT} holds {files} file{'' if files == 1 else 's'} ({size(total)})."


def submit(confirm):
    value = status(wait=START_STATUS_SECONDS)
    reason = submission_closed(value)
    if reason:
        print(f"frontier submit: {reason}", file=sys.stderr)
        return 2
    if not confirm:
        try:
            ASKED.write_text(json.dumps({"asked_at": time.time()}))
        except OSError as error:
            print(f"frontier submit: could not record the confirmation question in {ASKED} ({type(error).__name__}: "
                  f"{error}), so `frontier submit --confirm` would be refused.", file=sys.stderr)
            return 4
        lines = ["Are you sure you want to submit your solution? You will not be able to make any changes after "
                 "this point.",
                 "Submitting ends your session: your processes are stopped, your Spin machines are released and "
                 f"{OUTPUT} is graded as it is then.",
                 contents()]
        if value.get("running"):
            lines.append(f"Self-check {value['running'][:12]} is still running; submitting stops it.")
        lines.append("Run `frontier submit --confirm` to submit.")
        print("\n".join(lines))
        return 1
    if not ASKED.is_file():
        print("frontier submit: not submitted. Run `frontier submit` without --confirm first: it shows what "
              "submitting does and what would be submitted, and asks you to confirm.", file=sys.stderr)
        return 1
    answer = send(SUBMIT_PREFIX + secrets.token_hex(12), "the submission service",
                  "run `frontier submit` to see whether your solution was submitted.")
    if isinstance(answer, str):
        print(f"frontier submit: {answer}", file=sys.stderr)
        return 4
    if answer.get("rejected"):
        print(f"frontier submit: {answer['rejected']}", file=sys.stderr)
        return 2
    if not isinstance(answer.get("submitted_at"), (int, float)):
        print("frontier submit: the submission service gave an answer this command does not understand; run "
              "`frontier submit` to see whether your solution was submitted.", file=sys.stderr)
        return 4
    print(f"Submitted at {utc(answer['submitted_at'])}. Your session is over: your processes are stopped in a few "
          f"seconds and {OUTPUT} is graded as it is then. There is nothing more to do.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="frontier", description="Frontier task tools in the agent sandbox.")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("selfcheck", help="check /workspace/output the way it will be graded",
                                description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    check.add_argument("action", nargs="?", default="start", choices=["start", "wait", "status", "show"],
                       help="start (default) a check or follow the running one; wait for one; status; show a report")
    check.add_argument("check_id", nargs="?", help="a check ID or prefix (wait/show); default the running or latest")
    check.add_argument("--timeout", type=int, default=WAIT_SECONDS, help=f"seconds to wait (default {WAIT_SECONDS})")
    check.add_argument("--json", action="store_true", help="print the report (or the status) as JSON")
    final = commands.add_parser("submit", help="submit /workspace/output for grading, which ends your session",
                                description=SUBMIT_DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    final.add_argument("--confirm", action="store_true",
                       help="submit now; refused unless `frontier submit` has been run before")
    args = parser.parse_args(argv)
    if args.command == "submit":
        try:
            return submit(args.confirm)
        except KeyboardInterrupt:
            print("\nStopped waiting; run `frontier submit` to see whether your solution was submitted.",
                  file=sys.stderr)
            return 130
    if args.check_id and args.action in ("start", "status"):
        check.error(f"`{args.action}` takes no check ID; use `frontier selfcheck wait ID` or `show ID`")
    try:
        return selfcheck(args)
    except KeyboardInterrupt:  # only the waiting stops: the plugin, not this command, runs the check
        print("\nStopped waiting; a self-check that has started keeps running (`frontier selfcheck wait`).",
              file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
