"""Exact single-label classification and top-label calibration metric; stdlib only."""
from collections import defaultdict
import math

ECE_BINS = 15
METRIC = "mean_calibrated_macro_f1"


def probabilities(response, classes):
    if not isinstance(response, dict) or set(response) != {"probabilities"}:
        raise ValueError('Expected exactly {"probabilities": [...]}')
    values = response["probabilities"]
    if not isinstance(values, list) or len(values) != classes:
        raise ValueError("Expected one probability per supplied label")
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError("Probabilities must be finite numbers in [0,1], not booleans")
    total = math.fsum(values)
    if abs(total - 1) > 1e-6:
        raise ValueError("Probabilities must sum to one within 1e-6")
    return [p / total for p in values]


def failed_prediction(sample, failure=None):
    """What a request that failed after the retries, timed out or returned an undocumented shape scores as (see
    the task contract): a confidently wrong answer, probability one on the label after the gold one. That is the
    worst outcome under every part of the metric: a false positive and a false negative for macro-F1, a
    full-confidence miss for ECE, and the maximum Brier score and NLL. A uniform distribution would instead be
    rewarded at chance level, more on small label sets."""
    classes = len(sample["labels"])
    wrong = (sample["target"] + 1) % classes
    return {"id": sample["id"], "probabilities": [1.0 if j == wrong else 0.0 for j in range(classes)]}


def validate_samples(samples):
    if not isinstance(samples, list) or not samples:
        raise ValueError("Expected a nonempty dataset")
    ids, inventories = set(), {}
    for row in samples:
        if not isinstance(row, dict):
            raise ValueError("Dataset rows must be objects")
        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in ids:
            raise ValueError("Sample IDs must be unique nonempty strings")
        ids.add(row["id"])
        if not isinstance(row.get("dataset"), str) or not row["dataset"]:
            raise ValueError("Dataset group must be a nonempty string")
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError("Input text must be nonempty")
        labels = row.get("labels")
        if (not isinstance(labels, list) or not 2 <= len(labels) <= 256
                or any(not isinstance(x, str) or not x.strip() for x in labels)
                or len(set(labels)) != len(labels)):
            raise ValueError("Expected 2..256 distinct nonempty label descriptions")
        if type(row.get("target")) is not int or not 0 <= row["target"] < len(labels):
            raise ValueError("Gold target must be a valid integer class index")
        inventory = frozenset(labels)
        if inventories.setdefault(row["dataset"], inventory) != inventory:
            raise ValueError("Candidate label inventory changed within a dataset")
    return inventories


def score(samples, predictions, config=None):
    inventories = validate_samples(samples)
    if not isinstance(predictions, list) or len(samples) != len(predictions):
        raise ValueError("Expected one prediction for every sample")
    groups = defaultdict(list)
    for row, pred in zip(samples, predictions, strict=True):
        if not isinstance(pred, dict) or pred.get("id") != row["id"]:
            raise ValueError("Prediction IDs/order differ from inputs")
        values = probabilities({"probabilities": pred.get("probabilities")}, len(row["labels"]))
        winner = max(range(len(values)), key=values.__getitem__)
        groups[row["dataset"]].append((row, values, winner))
    details = {}
    for name, records in groups.items():
        counts = {}  # TP, FP, FN over the union of gold and predicted classes, as in macro-F1.
        bins = [[0, 0, 0.0] for _ in range(ECE_BINS)]
        n, correct, brier, nll = len(records), 0, 0.0, 0.0
        for row, values, winner in records:
            gold, guess = row["labels"][row["target"]], row["labels"][winner]
            counts.setdefault(gold, [0, 0, 0])
            counts.setdefault(guess, [0, 0, 0])
            hit = gold == guess
            if hit:
                counts[gold][0] += 1
            else:
                counts[guess][1] += 1
                counts[gold][2] += 1
            confidence = values[winner]
            slot = bins[min(ECE_BINS - 1, int(ECE_BINS * confidence))]
            slot[0] += 1
            slot[1] += int(hit)
            slot[2] += confidence
            correct += int(hit)
            brier += math.fsum((p - int(j == row["target"])) ** 2 for j, p in enumerate(values))
            nll -= math.log(max(values[row["target"]], 1e-15))
        f1 = math.fsum(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
                       for tp, fp, fn in counts.values()) / len(counts)
        ece = min(1.0, math.fsum(abs(hits - confidence_sum) for _, hits, confidence_sum in bins) / n)
        details[name] = {"samples": n, "classes": len(inventories[name]), "scored_classes": len(counts), "macro_f1": f1,
                         "accuracy": correct / n, "ece": ece, "score": f1 * (1 - ece),
                         "brier": brier / n, "nll": nll / n,
                         "calibration_bins": [{"count": count,
                             "accuracy": hits / count if count else None,
                             "confidence": total / count if count else None}
                             for count, hits, total in bins]}
    mean = lambda key: math.fsum(d[key] for d in details.values()) / len(details)
    return {"score": mean("score"), "details": {"metric": METRIC, "samples": len(samples),
            "datasets": len(details), "macro_f1": mean("macro_f1"), "ece": mean("ece"),
            "accuracy": mean("accuracy"), "ece_bins": ECE_BINS, "per_dataset": details}}
