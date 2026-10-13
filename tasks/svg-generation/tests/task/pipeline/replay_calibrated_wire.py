#!/usr/bin/env python3
"""Prove canonical JSON payload identity using archived responses; never call an API.

Run inside the pinned renderer image with --network none. HTTP JSON whitespace
can differ in the shared client; every parsed message and parameter must match.
Payloads are matched by canonical-JSON digest within each control group, so the
scorer's concurrent judge pool cannot reorder which receipt answers which request.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import sys

SEED = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SEED))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encoded(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def replay(work_dir, data_path):
    verifier = load("svg_wire_verifier", SEED / "verifier.py")
    calibrated = load("svg_wire_v4_calibration", SEED / "pipeline/calibrate_v4.py")
    # Also fail closed if accidentally launched outside the network-disabled image.
    def no_network(*args, **kwargs):
        raise RuntimeError("Offline archive replay cannot access the network")
    socket.socket = no_network
    socket.create_connection = no_network
    work = Path(work_dir)
    plan = json.loads((work / "plan.json").read_text())
    raw_report = (work / "report.json").read_bytes()
    report = json.loads(raw_report)
    if report["execution"]["status"] != "complete" or len(plan["probes"]) != 96:
        raise ValueError("A complete 96-call calibration is required")
    if digest(encoded(plan)) != report["plan_sha256"] or plan["code_sha256"] != calibrated.code_hashes():
        raise ValueError("Exact paid scorer, verifier, renderer code, plan and frozen sources must match")
    if digest(Path(data_path).read_bytes()) != plan["input"]["sha256"]:
        raise ValueError("Evaluation data differ from calibration")
    verifier.renderer_binary(official=True)
    attempts = {a["id"]: a for a in report["execution"]["attempts"]}
    if len(attempts) != 96 or any(a["status"] != "confirmed" for a in attempts.values()):
        raise ValueError("All native responses must be confirmed")
    records, groups = [], {}
    for kind in calibrated.KINDS:
        probes = [p for p in plan["probes"] if p["kind"] == kind]
        by_request = {p["request_sha256"]: p for p in probes}
        if len(probes) != 24 or len(by_request) != 24:
            raise ValueError("Each control group needs 24 distinct probes")
        rows = [next(r for r in plan["samples"] if r["id"] == p["sample_id"]) for p in probes]
        replayed = []

        def archive_judge(**payload):
            raw = encoded(payload)
            probe = by_request.get(digest(raw))
            if probe is None or raw != (work / probe["request_path"]).read_bytes():
                raise ValueError("Official wrapped payload differs from paid canonical JSON")
            response = (work / "responses" / (probe["id"] + ".json")).read_bytes()
            attempt = attempts[probe["id"]]
            if digest(response) != attempt["response_sha256"] or probe["request_sha256"] != attempt["request_sha256"]:
                raise ValueError("Native archived receipt differs")
            replayed.append({k: attempt[k] for k in ("id", "request_sha256", "response_sha256")})
            return json.loads(response)

        score = verifier.public.score(rows, [{"id": r["id"], "svg": calibrated.candidate_svg(kind, r)} for r in rows],
                                      {"judge": verifier.calibrated_judge(archive_judge)})
        if sorted(r["id"] for r in replayed) != sorted(p["id"] for p in probes):
            raise ValueError("A paid probe was not replayed exactly once")
        records.extend(replayed)
        groups[kind] = {"complete": True, "confirmed": 24, **score}
    if groups != report["quality_scores"]["groups"] or len(records) != 96:
        raise ValueError("Exact archived response scores differ")
    return {"status": "passed", "api_calls": 0, "network": "disabled",
            "identity": "byte-identical canonical JSON payloads; HTTP whitespace is not constrained",
            "matched_payloads": 96, "matched_archived_responses": 96,
            "binding": verifier.calibration_binding(data_path), "plan_sha256": report["plan_sha256"],
            "report_sha256": digest(raw_report), "replay_code_sha256": digest(Path(__file__).read_bytes()),
            "replayed_groups_sha256": digest(encoded(groups)), "records": sorted(records, key=lambda r: r["id"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt = replay(args.work_dir, args.data_path)
    Path(args.output).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
