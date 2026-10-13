"""Inspect JSON tree checkpoints as weights without executing candidate code.

Numeric tree payloads get the same treatment as binary tensor payloads: inspect
their structure and aggregate metadata, and bind the inventory to the actual file.
Unknown fields remain visible. Unrecognized structures fall back to source review;
this reader is not a checkpoint-format acceptance gate.
"""
from collections import Counter
import hashlib
import json
import math


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def tree_inventory(text, raw):
    """Return metadata for a recognized LightGBM JSON dump, or None.

    Every node is visited. No filename, byte size, number of trees, feature name
    or incidental text determines compliance. Unsupported structures remain raw
    source evidence rather than being silently summarized or rejected.
    """
    if not text.lstrip().startswith("{"):
        return None
    try:
        model = json.loads(text, object_pairs_hook=_unique_object)
        if not isinstance(model, dict) or model.get("name") != "tree" or not isinstance(model.get("tree_info"), list):
            return None
        return _lightgbm_inventory(model, raw)
    except (ValueError, TypeError, KeyError, RecursionError):
        return None


def _lightgbm_inventory(model, raw):
    numeric = {}
    features, decisions, missing, defaults = Counter(), Counter(), Counter(), Counter()
    extra = []
    splits = leaves = max_depth = 0
    inconsistencies = []

    def number(field, value):
        if type(value) not in (int, float):
            raise ValueError("Unsupported numeric field")
        stats = numeric.setdefault(field, {"count": 0, "nonfinite": 0, "min": None, "max": None})
        stats["count"] += 1
        if isinstance(value, float) and not math.isfinite(value):
            stats["nonfinite"] += 1
        else:
            stats["min"] = value if stats["min"] is None else min(stats["min"], value)
            stats["max"] = value if stats["max"] is None else max(stats["max"], value)

    def remainder(value, known, path):
        unknown = {key: item for key, item in value.items() if key not in known}
        if unknown:
            # Keep all nonstandard fields, including arbitrary text, verbatim.
            extra.append({"path": path, "fields": unknown})

    tree_fields = {"tree_index", "num_leaves", "num_cat", "shrinkage"}
    leaf_fields = {"leaf_index", "leaf_value", "leaf_weight", "leaf_count"}
    split_fields = {"split_index", "split_feature", "split_gain", "internal_value", "internal_weight", "internal_count"}
    for index, tree in enumerate(model["tree_info"]):
        if not isinstance(tree, dict) or "tree_structure" not in tree:
            raise ValueError("Unsupported tree record")
        path = f"/tree_info/{index}"
        for key in tree_fields & tree.keys():
            number("tree." + key, tree[key])
        remainder(tree, tree_fields | {"tree_structure"}, path)
        pending = [(tree["tree_structure"], path + "/tree_structure", 0)]
        tree_leaves = 0
        while pending:
            node, node_path, depth = pending.pop()
            if not isinstance(node, dict):
                raise ValueError("Unsupported tree node")
            max_depth = max(max_depth, depth)
            if "leaf_value" in node:
                leaves += 1
                tree_leaves += 1
                for key in leaf_fields & node.keys():
                    number(key, node[key])
                remainder(node, leaf_fields, node_path)
            else:
                if type(node.get("split_feature")) is not int or not {"left_child", "right_child", "threshold"} <= node.keys():
                    raise ValueError("Unsupported split node")
                splits += 1
                features[str(node["split_feature"])] += 1
                for key in split_fields & node.keys():
                    number(key, node[key])
                # Categorical thresholds and unfamiliar split semantics remain
                # verbatim evidence rather than being mistaken for numeric data.
                known = split_fields | {"left_child", "right_child"}
                if type(node["threshold"]) in (int, float):
                    number("threshold", node["threshold"])
                    known.add("threshold")
                for key, counter, values in [("decision_type", decisions, {"<=", "=="}),
                                             ("missing_type", missing, {"None", "Zero", "NaN"})]:
                    value = node.get(key)
                    if isinstance(value, str) and value in values:
                        counter[value] += 1
                        known.add(key)
                if type(node.get("default_left")) is bool:
                    defaults[str(node["default_left"]).lower()] += 1
                    known.add("default_left")
                remainder(node, known, node_path)
                pending.extend((node[side], node_path + "/" + side, depth + 1)
                               for side in ("right_child", "left_child"))
        if "num_leaves" in tree and tree["num_leaves"] != tree_leaves:
            inconsistencies.append({"path": path, "declared_leaves": tree["num_leaves"], "observed_leaves": tree_leaves})
    return {"format": "lightgbm-json", "inspection": "complete", "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "representation": "All tree nodes inspected; numeric parameters summarized, not included verbatim. This establishes checkpoint structure, not provenance, causality or actual inference use.",
            "metadata": {key: value for key, value in model.items() if key != "tree_info"},
            "tree_count": len(model["tree_info"]), "split_nodes": splits, "leaf_nodes": leaves,
            "max_depth": max_depth, "feature_split_counts": dict(features),
            "decision_types": dict(decisions), "missing_types": dict(missing),
            "default_left": dict(defaults), "numeric_fields": numeric,
            "unrecognized_fields": extra, "structural_inconsistencies": inconsistencies}
