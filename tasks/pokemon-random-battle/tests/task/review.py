"""Shared evidence review; no candidate source is imported or executed."""
from pathlib import Path

from frontier_ml.review_binaries import UpstreamWheelInspector, sha256_file

SHOWDOWN_REVISION = "ea822f8863407f7a16995a9db3519f8e250237c1"
SHOWDOWN_SHA256 = "e5f7ca974b2dadf59d1606d268687a089b2a66aeb3f57ddb5a390149bcdd28e1"
ENGINE_WHEEL = "poke_engine-0.0.48-cp312-cp312-linux_x86_64.whl"
# Organizer build from the hash-pinned PyPI sdist, never from candidate input.
# Build recipe and receipt: pipeline/build_poke_engine.py and pipeline/engine-build.json.
ENGINE_WHEEL_SHA256 = "b91209b2738c06dc14c7b37cf592574a6262bc4001af6349aee3323d35481747"
ENGINE_SOURCE = "poke_engine-0.0.48.tar.gz"
ENGINE_SOURCE_SHA256 = "070010686f2aedff11e25137e696e301ccd80fd57c805d255464067fc905ca12"
_wheels = UpstreamWheelInspector()


def binary_inspector(path):
    """Exact organizer dependencies are audited; all other wheels need upstream provenance."""
    path = Path(path)
    if path.name in {ENGINE_WHEEL, ENGINE_SOURCE}:
        digest = sha256_file(path)
        expected = ENGINE_WHEEL_SHA256 if path.name == ENGINE_WHEEL else ENGINE_SOURCE_SHA256
        audit = {"approved": digest == expected, "sha256": digest,
                 "reviewer": "frontier-ml pinned Pokemon engine audit"}
        if digest != expected:
            return {**audit, "reason": "Use the exact engine artifact from /task/public/engine; "
                    "a candidate-built wheel or modified source archive is not covered by this audit"}
        return {**audit, "provenance": "poke-engine 0.0.48 PyPI sdist SHA-256 " + ENGINE_SOURCE_SHA256 +
                "; organizer-built CPython 3.12 Linux x86-64 wheel, Rust 1.89.0, maturin 1.9.4, "
                "locked Cargo dependencies, poke-engine/terastallization, no default features; "
                "pipeline/build_poke_engine.py. No candidate build or provenance claim is trusted."}
    if path.name == "showdown.tar.gz" and sha256_file(path) == SHOWDOWN_SHA256:
        return {"approved": True, "sha256": SHOWDOWN_SHA256,
                "reviewer": "frontier-ml pinned Pokemon simulator audit",
                "provenance": f"Exact task-provided MIT-licensed source archive from "
                f"https://github.com/smogon/pokemon-showdown/tree/{SHOWDOWN_REVISION}"}
    print(f"Auditing dependency: {path.name}", flush=True)
    audit = _wheels(path)
    print(f"Dependency audit approved={audit['approved']}: {path.name}", flush=True)
    return audit


def submission_inspector(snapshot):
    """Carry exact simulator provenance through upstream's archive-aware review."""
    snapshot = Path(snapshot)
    audits = {}
    # Enforced here, not only by binary_inspector: an unapproved native file is otherwise handed to the reviewer.
    for name in ("showdown.tar.gz", ENGINE_SOURCE, ENGINE_WHEEL):
        for path in snapshot.rglob(name):
            audit = binary_inspector(path)
            if audit.get("approved") is not True:
                raise ValueError("Bundled dependency archive differs from the pinned public artifact: " + name)
            audits[path.relative_to(snapshot).as_posix()] = audit
    return audits


POLICY = {
    "base_model": None,
    "base_revision": None,
    "base_optional": True,
    "checks": {
        "implementation_provenance": (
            "Inspect serving source, configuration, architecture.json and provenance for reused code and models. "
            "Existing public bots and pretrained policies are allowed, including unchanged or modified Foul Play. "
            "New training, randomly initialized parameters and learned weights are not required. A search-only or "
            "rule-only bot with no checkpoint is valid; base_model and base_revision may both be none, with reused "
            "code repositories and revisions declared separately. For pretrained weights, declare the model and "
            "revision. If training or fine-tuning is performed, inspect its source, data provenance and readable "
            "logs for consistency with the shipped weights. Teacher demonstrations and public human data are allowed."
        ),
        "actual_served_policy": (
            "Trace serve.sh through POST /act, observation handling and every action-selection path. Decisions must "
            "come from the declared, shipped implementation operating offline on the supplied observation. "
            "Unchanged Foul Play, modified existing bots, heuristic search, learned policies, ensembles and local "
            "fallback policies are allowed; no learned model or weight change is required. Any claimed model use "
            "must agree with the actual loading and serving path. Reject remote decision services and undeclared "
            "decision paths. Do not reject a submission merely because it reuses an existing bot without training."
        ),
        "permitted_information": (
            "Only supplied player-visible observations and public game mechanics may drive "
            "inference. No simulator process inspection, opponent private state/RNG access, network service, "
            "private evaluation acquisition, hard-coded test detection or cached evaluation actions. Public "
            "development simulation and validation-guided model selection are allowed. Inspect training and serving paths."
        ),
    },
}
