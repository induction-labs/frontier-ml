"""Human-readable tables for terminals; JSON stays the output for pipes and --json."""
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import os
import sys


def wanted(args):
    return not getattr(args, "json", False) and sys.stdout.isatty()


def _bold(text):
    return text if os.environ.get("NO_COLOR") else f"\033[1m{text}\033[0m"


def _dim(text):
    return text if os.environ.get("NO_COLOR") else f"\033[2m{text}\033[0m"


def _cell(row, get):
    value = get(row) if callable(get) else row.get(get)
    return "—" if value is None or value == "" else str(value)


def table(rows, columns):
    """columns: (header, key-or-function, align) with align "<" or ">"."""
    if not rows:
        return ""
    cells = [[_cell(row, get) for _, get, _ in columns] for row in rows]
    widths = [max([len(header), *(len(r[i]) for r in cells)]) for i, (header, _, _) in enumerate(columns)]
    line = lambda values: "  ".join(f"{v:{a}{w}}" for v, w, (_, _, a) in zip(values, widths, columns)).rstrip()
    return "\n".join([_bold(line([h for h, _, _ in columns])), *(line(r) for r in cells)])


def usd(value, places=2):
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, TypeError):
        return None
    if 0 < amount < Decimal("0.01"):
        return "$" + format(amount.normalize(), "f")  # Show sub-cent inference spend exactly.
    return f"${amount:,.{places}f}"


def duration(seconds):
    if seconds is None:
        return None
    seconds = int(seconds)
    if seconds < 3600:
        return f"{max(1, round(seconds / 60))}m"
    return f"{seconds / 3600:.1f}h".replace(".0h", "h") if seconds < 86400 * 2 else f"{round(seconds / 86400)}d"


def until(iso):
    if not iso:
        return None
    left = (datetime.fromisoformat(iso.replace("Z", "+00:00")) - datetime.now(timezone.utc)).total_seconds()
    return "expired" if left <= 0 else "in " + duration(left)


def offers(result):
    rows = sorted(result.get("offers", []), key=lambda o: (o.get("provider"), Decimal(str(o.get("hourly_usd") or 0))))
    notes = lambda o: ", ".join(x for x in ("spot" if o.get("spot") else None, o.get("location"),
                                            "any image" if o.get("custom_image") else None,
                                            "extendable" if o.get("supports_extension") else None,
                                            "simulated" if o.get("simulation") else None) if x)
    out = table(rows, [("PROVIDER", "provider", "<"), ("OFFER", "id", "<"),
                       ("GPU", lambda o: o.get("gpu_model") or o.get("gpu") or "CPU", "<"), ("GPUS", "gpus", ">"),
                       ("$/HOUR", lambda o: usd(o.get("hourly_usd")), ">"),
                       ("MAX", lambda o: duration((o.get("limits") or {}).get("max_seconds")), ">"),
                       ("NOTES", notes, "<")]) or "No offers."
    if rows:
        example = rows[0]
        out += "\n\n" + _dim(f"Create one: spin compute create --provider {example['provider']} --request-id my-run-1 "
                             f"--spec '{json.dumps(example['spec'], separators=(',', ':'))}'")
        out += "\n" + _dim('Add "image": "<registry/ref>" to the spec where NOTES says any image. '
                           "Full specs: spin compute offers --json")
    return out


def resources(result):
    return table(result.get("resources", []), [
        ("ID", "id", "<"), ("PROVIDER", "provider", "<"), ("STATUS", "status", "<"),
        ("GPU", lambda r: ((r.get("spec") or {}).get("gpu") or ("CPU" if (r.get("spec") or {}).get("gpus") == 0 else None)), "<"),
        ("EXPIRES", lambda r: until(r.get("expires_at")) if r.get("status") not in {"stopped", "deleted", "rejected"} else None, "<"),
        ("CHARGED", lambda r: usd(r.get("charged_usd")), ">")]) or "No resources."


def fields(result, keys=None):
    pairs = [(k, v) for k, v in result.items() if (keys is None or k in keys) and not isinstance(v, (dict, list))]
    width = max((len(k) for k, _ in pairs), default=0)
    fmt = lambda k, v: usd(v) if k.endswith("_usd") else v
    return "\n".join(f"{_bold(k.ljust(width))}  {fmt(k, v)}" for k, v in pairs)


def wallets(result):
    budget = lambda w: w["budget_usd"] if "budget_usd" in w else Decimal(w["budget_cents"]) / 100
    return table(result.get("data", []), [
        ("ID", "id", "<"), ("NAME", "name", "<"), ("STATUS", "status", "<"),
        ("BUDGET", lambda w: "No limit" if w.get("unlimited") else usd(budget(w))
                             + (" static" if w.get("static_pricing") else ""), ">"),
        ("SPENT", lambda w: usd(w.get("spent_usd", Decimal(w.get("spent_cents", 0)) / 100)), ">"),
        ("EXPIRES", lambda w: until(w.get("expires_at")) if w.get("status") == "active" else None, "<")]) or "No wallets."


def models(result):
    return "\n".join(m["id"] for m in result.get("data", []))


def per_million(price):
    try:
        return f"{Decimal(str(price)) * 1_000_000:.3f}".rstrip("0").rstrip(".")
    except (InvalidOperation, TypeError):
        return None


def endpoints(result):
    out = table(result.get("endpoints", []), [
        ("PROVIDER", "tag", "<"), ("QUANT", "quantization", "<"), ("CONTEXT", "context_length", ">"),
        ("$/M IN", lambda e: per_million((e.get("pricing") or {}).get("prompt")), ">"),
        ("$/M OUT", lambda e: per_million((e.get("pricing") or {}).get("completion")), ">"),
        ("TOK/S", lambda e: None if e.get("throughput_last_30m") is None else str(e["throughput_last_30m"]), ">"),
        ("DEFAULT", lambda e: "yes" if e.get("within_default_cap") else "no", "<")]) or "No endpoints."
    return out + "\n\n" + _dim('DEFAULT: reached by calls without a provider object. Choose with "provider": '
                                '{"only": ["<PROVIDER>"]}; see spin docs.')


def render(args, result):
    """Table text for terminal-facing commands, else None to print JSON."""
    if not wanted(args) or not isinstance(result, dict) or "error" in result:
        return None
    command = (args.group, getattr(args, "action", None))
    if command == ("compute", "offers"):
        return offers(result)
    if command in {("compute", "list"), ("wallet", "resources")}:
        return resources(result)
    if command == ("compute", "status"):
        return fields(result, {"id", "provider", "status", "billing_state", "created_at", "expires_at",
                               "purchased_seconds", "price_hourly_usd", "charged_usd", "reserved_usd"})
    if command == ("budget", "status"):
        return fields(result)
    if command == ("wallet", "list"):
        return wallets(result)
    if command == ("openrouter", "models"):
        return models(result)
    if command == ("openrouter", "endpoints"):
        return endpoints(result)
    return None


def emit(args, result):
    text = render(args, result)
    print(text if text is not None else json.dumps(result, indent=2))
