"""Reward normalization: raw metrics rescaled linearly between measured anchors and clamped to [0, 1].

task.yaml `score_normalization` names one or more raw values in the verifier's result (dotted paths such as
``score`` or ``details.benchmarks.gpqa_diamond.score``), a floor and a ceiling for each, and how the
normalized components combine:

    score_normalization:
      aggregate: single          # single | mean | harmonic_mean (any zero component gives zero)
      components:
      - {name: bpb, path: details.bpb, floor: 1.461, ceiling: 1.25}
      provenance: where each anchor was measured

Each component is (raw - floor) / (ceiling - floor), clamped to [0, 1]; a ceiling below the floor means lower
raw values are better (bits per byte). The raw score, every raw component and the anchors are kept in
details.normalization and result["raw_score"]; only the reward changes. A zero from a gate
(details.failure) is never normalized, and a result without its raw values stays at zero, so gate,
infrastructure and trivial zeros keep their meaning.

The same function renormalizes a stored result.json offline (``python -m frontier_ml.scoring.normalization``).
"""
import argparse
import copy
import json
import math
from pathlib import Path

AGGREGATES = ("single", "mean", "harmonic_mean")
VERSION = "linear-anchors-v1"


def check_spec(spec: dict) -> dict:
    """Raise ValueError unless spec is a well-formed score_normalization block."""
    if not isinstance(spec, dict):
        raise ValueError("score_normalization must be a mapping")
    unknown = set(spec) - {"aggregate", "components", "provenance"}
    if unknown:
        raise ValueError(f"Unknown score_normalization keys: {', '.join(sorted(unknown))}")
    aggregate, components = spec.get("aggregate"), spec.get("components")
    if aggregate not in AGGREGATES:
        raise ValueError(f"score_normalization.aggregate must be one of {', '.join(AGGREGATES)}")
    if not isinstance(components, list) or not components or (aggregate == "single" and len(components) != 1):
        raise ValueError("score_normalization.components must be a nonempty list (exactly one for single)")
    names = set()
    for component in components:
        if not isinstance(component, dict) or set(component) != {"name", "path", "floor", "ceiling"}:
            raise ValueError("Each normalization component needs exactly name, path, floor and ceiling")
        if not isinstance(component["name"], str) or component["name"] in names:
            raise ValueError("Normalization component names must be unique strings")
        names.add(component["name"])
        if not isinstance(component["path"], str) or not component["path"]:
            raise ValueError("Normalization component path must be a dotted path into the result")
        floor, ceiling = component["floor"], component["ceiling"]
        if not all(type(value) in (int, float) and math.isfinite(value) for value in (floor, ceiling)) or floor == ceiling:
            raise ValueError(f"Normalization anchors for {component['name']} must be distinct finite numbers")
    return spec


def _lookup(result: dict, path: str):
    value = result
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def normalize(result: dict, spec: dict | None) -> dict:
    """The result with its score replaced by the normalized reward (a new dict; the input is not modified)."""
    if not spec:
        return result
    check_spec(spec)
    out = copy.deepcopy(result)
    details = out.setdefault("details", {})
    raw = out.get("raw_score", out.get("score"))  # idempotent on an already normalized result
    record = {"version": VERSION, "aggregate": spec["aggregate"], "raw_score": raw,
              "provenance": spec.get("provenance")}
    if details.get("failure"):
        record.update(applied=False, reason=f"gate: {details['failure']}")
        reward = 0.0
    else:
        components, missing = {}, []
        for component in spec["components"]:
            value = _lookup(out, component["path"])
            if component["path"] == "score" and "raw_score" in out:
                value = out["raw_score"]
            if value is None:
                missing.append(component["path"])
                continue
            floor, ceiling = component["floor"], component["ceiling"]
            components[component["name"]] = {"raw": value, "floor": floor, "ceiling": ceiling,
                                             "normalized": _clamp((value - floor) / (ceiling - floor))}
        record["components"] = components
        if missing:
            record.update(applied=False, reason="raw values missing: " + ", ".join(missing))
            reward = 0.0
        else:
            values = [c["normalized"] for c in components.values()]
            if spec["aggregate"] == "harmonic_mean":
                reward = 0.0 if any(v == 0 for v in values) else len(values) / sum(1 / v for v in values)
            else:
                reward = sum(values) / len(values)
            record["applied"] = True
    reward = _clamp(reward)
    record["reward"] = reward
    details["normalization"] = record
    out["raw_score"] = raw
    out["score"] = reward
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Renormalize stored result.json files with a task's current anchors")
    parser.add_argument("task_yaml", type=Path)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--write", action="store_true", help="rewrite result.json and the sibling reward.txt")
    args = parser.parse_args(argv)
    import yaml
    spec = yaml.safe_load(args.task_yaml.read_text()).get("score_normalization")
    for path in args.results:
        result = json.loads(path.read_text())
        if result.get("status") != "completed":
            print(f"{path}: status {result.get('status')}, left at {result.get('score')}")
            continue
        normalized = normalize(result, spec)
        print(f"{path}: {normalized['raw_score']} -> {normalized['score']}")
        if args.write:
            path.write_text(json.dumps(normalized, indent=2, allow_nan=False) + "\n")
            (path.parent / "reward.txt").write_text(str(normalized["score"]) + "\n")


if __name__ == "__main__":
    main()
