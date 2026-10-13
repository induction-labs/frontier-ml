"""GPU names and static prices.

canonical() reads a provider's GPU name as (model, variant); gpu_model() is the name Spin lists and prices a
GPU by. H100 and H200 are listed and priced by form factor ("H100 SXM", "H100 PCIe", "H100 NVL"); an H100 whose
provider doesn't say which lists as "H100 (unknown form factor)".

A wallet created with static pricing sees one fixed USD price per GPU-hour by GPU model instead of the
providers' prices, and its budget is spent in them; the operator still pays what the provider costs.
"""
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import json
from pathlib import Path
import re

DEFAULT_PRICES = Path(__file__).resolve().parents[1] / "prices.json"
FORM_FACTORS = {"H100": ("SXM", "PCIe", "NVL"), "H200": ("SXM", "NVL")}  # Models listed and priced by form factor.
UNKNOWN_FORM_FACTOR = "unknown form factor"
# Modal's names that gpu_model() alone would read differently: its H100 is an SXM part.
MODAL_GPU_MODELS = {"H100": "H100 SXM"}


def canonical(raw):
    """Provider GPU name -> (model, variant). Variant is None where providers don't distinguish."""
    s = re.sub(r"[_\-]", " ", (raw or "").lower())
    s = re.sub(r"\b(nvidia|geforce|tesla|quadro)\b", " ", s).replace("®", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if not s or "mig" in s:
        return None, None
    if "mi350" in s: return "MI350X", None
    if "mi300" in s: return "MI300X", None
    if "gb300" in s: return "GB300", None
    if "gb200" in s: return "GB200", None
    if "b300" in s: return "B300", None
    if "b200" in s: return "B200", None
    if "gh200" in s: return "GH200", None  # Before H200, which its name contains.
    if "h200" in s: return "H200", "NVL" if re.search(r"\bnvl\b", s) else "SXM"
    if "h100" in s:
        if re.search(r"\bnvl\b", s): return "H100", "NVL"
        if "pcie" in s: return "H100", "PCIe"
        if re.search(r"sxm|hbm3|nvlink", s): return "H100", "SXM"
        return "H100", None
    if "a100" in s: return "A100", "40GB" if "40" in s else ("80GB" if "80" in s or "dgx" in s else None)
    if re.search(r"pro ?6000|6000 ?pro", s): return "RTX PRO 6000", None
    if re.search(r"pro ?(\d{4})", s): return "RTX PRO " + re.search(r"pro ?(\d{4})", s)[1], None
    m = re.search(r"(\d{4}) ?ada", s)
    if m: return f"RTX {m[1]} Ada", None
    m = re.search(r"\b(?:rtx )?a(\d{4})\b", s)
    if m: return f"RTX A{m[1]}", None
    for pat, name in [(r"\bl40s\b", "L40S"), (r"\bl40\b", "L40"), (r"\bl4\b", "L4"), (r"\ba40\b", "A40"), (r"\ba30\b", "A30"),
                      (r"\ba16\b", "A16"), (r"\ba10\b", "A10"), (r"\bv100\b", "V100"), (r"\bp100\b", "P100"), (r"\bt4\b", "T4"),
                      (r"cmp ?170", "CMP 170HX")]:
        if re.search(pat, s):
            return name, ("32GB" if name == "V100" and "32" in s else None)
    quadro = s.startswith("q ") or "quadro" in (raw or "").lower()  # "Quadro" itself was stripped above.
    m = re.search(r"\bq?\s?rtx (\d{4})\b", s) if quadro else None
    if m: return f"Quadro RTX {m[1]}", None
    if "titan" in s:
        return "Titan " + (re.search(r"titan (xp|x|v|rtx)", s)[1].upper() if re.search(r"titan (xp|x|v|rtx)", s) else ""), None
    m = re.search(r"\b(rtx|gtx)? ?(\d{4})(s)?\b((?: (?:ti|super|s|laptop)\b)*)", s)
    if m and m[2][0] in "123456789":
        family = m[1] or ("gtx" if m[2][:2] in ("10", "16") else "rtx")
        words = (["super"] if m[3] else []) + m[4].split()  # "4070s ti" is the 4070 Ti Super
        order = {"ti": 0, "super": 1, "s": 1, "laptop": 2}
        suffix = " ".join({"ti": "Ti", "super": "Super", "s": "Super", "laptop": "Laptop"}[w]
                          for w in sorted(set(words), key=order.get))
        return f"{family.upper()} {m[2]}" + (f" {suffix}" if suffix else ""), None
    return "Other: " + (raw or "?"), None


def gpu_model(raw):
    """The name a GPU is listed and priced by: "H100 SXM", "H100 (unknown form factor)", "A100"; None for no GPU."""
    model, variant = canonical(raw)
    if model in FORM_FACTORS:
        return f"{model} {variant}" if variant else f"{model} ({UNKNOWN_FORM_FACTOR})"
    return model


def modal_gpu_model(name):
    return MODAL_GPU_MODELS.get(name) or gpu_model(name)


def with_gpu_model(offer):
    """An offer with its gpu_model beside the provider's own GPU name; an adapter's own gpu_model wins."""
    if not isinstance(offer.get("gpu"), str):
        return offer
    return {**offer, "gpu_model": offer.get("gpu_model") or gpu_model(offer["gpu"])}


def cost(hourly_usd, seconds):
    """A lease's static price, rounded up to the micro-dollar like provider quotes."""
    return str((Decimal(hourly_usd) * seconds / 3600).quantize(Decimal("0.000001"), rounding=ROUND_CEILING))


def market_price_field(key):
    """Offer and lease fields that carry a provider's own price; a static wallet never sees them."""
    return key.endswith(("_usd", "_eur")) or "_usd_" in key or key in {"rates", "rate_version", "usd_per_eur", "window"}


class StaticPricing:
    """A wallet's price table: {"gpu_hour_usd": {model: price per GPU-hour}, "cpu_hour_usd": price per CPU-hour}.

    A model can name a form factor ("H100 SXM"); a bare model ("H100") prices the form factors the table
    doesn't name. Machines with a GPU the table lacks cannot be leased. CPU-only machines are priced per CPU
    when the table has cpu_hour_usd, and otherwise at their market price, which then counts against the budget.
    """

    def __init__(self, table):
        self.version = table.get("version")
        self.gpu_hour_usd = {}
        for name, rate in (table.get("gpu_hour_usd") or {}).items():
            model = gpu_model(name)
            if model:  # Two names for one model ("H100", "H100 (unknown form factor)") take the higher price.
                self.gpu_hour_usd[model] = max(Decimal(str(rate)), self.gpu_hour_usd.get(model, Decimal(0)))
        rate = table.get("cpu_hour_usd")
        self.cpu_hour_usd = None if rate is None else Decimal(str(rate))

    def price(self, gpu, gpus, cpus=None):
        """How the table prices a machine: {"gpu", "gpus", "hourly_usd"}, {"cpus", "hourly_usd"} for CPU-only,
        "market" when a CPU-only machine keeps its market price, or None when it cannot be priced."""
        if type(gpus) is int and gpus > 0:
            model = gpu_model(gpu) if isinstance(gpu, str) else None
            bare = gpu_model(canonical(gpu)[0]) if model else None  # "H100 SXM" falls back to the table's "H100".
            rate = self.gpu_hour_usd.get(model, self.gpu_hour_usd.get(bare))
            return None if rate is None else {"gpu": model, "gpus": gpus, "hourly_usd": str(rate * gpus)}
        if gpu is not None or gpus not in (0, None):
            return None
        if self.cpu_hour_usd is None:
            return "market"
        if type(cpus) is not int or cpus < 1:
            return None
        return {"cpus": cpus, "hourly_usd": str(self.cpu_hour_usd * cpus)}

    def offers(self, offers):
        """Offers repriced for the wallet, cheapest first. Among equal static prices the provider's
        cheaper machine comes first, since the operator pays that difference."""
        priced = []
        for offer in offers:
            price = self.price(offer.get("gpu_model") or offer.get("gpu"), offer.get("gpus"), offer.get("cpus"))
            if price is None:
                continue
            market = Decimal(str(offer.get("hourly_usd") or 0))
            if price == "market":
                priced.append((market, market, {**offer, "pricing": "market"}))
                continue
            shown = {k: v for k, v in offer.items() if not market_price_field(k)}
            priced.append((Decimal(price["hourly_usd"]), market, {**shown, "hourly_usd": price["hourly_usd"], "pricing": "static",
                                                                  "priced_as": {k: v for k, v in price.items() if k != "hourly_usd"}}))
        return [offer for _, _, offer in sorted(priced, key=lambda item: item[:2])]


def hourly_price(value):
    """A positive USD price of at most $10,000 with at most nine decimals (the ledger's resolution), as a string."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError("price must be a number or decimal string")
    text = value.strip() if isinstance(value, str) else repr(value) if isinstance(value, float) else str(value)
    try:
        amount = Decimal(text)
    except InvalidOperation:
        raise ValueError("price must be a number or decimal string") from None
    if not amount.is_finite() or not 0 < amount <= 10_000 or amount.as_tuple().exponent < -9:
        raise ValueError("price must be a positive USD amount up to 10000 with at most 9 decimals")
    return format(amount.normalize(), "f")


def parse_table(table):
    """A wallet's price table, validated: {"gpu_hour_usd": {...}, "cpu_hour_usd"?: ..., "version"?: ...}."""
    if not isinstance(table, dict):
        raise ValueError("static_pricing must be true, false, or a price table")
    unknown = set(table) - {"gpu_hour_usd", "cpu_hour_usd", "version"}
    if unknown:
        raise ValueError(f"unknown field {sorted(unknown)[0]!r}; a price table has gpu_hour_usd, cpu_hour_usd and version")
    gpus = table.get("gpu_hour_usd")
    if not isinstance(gpus, dict) or not 1 <= len(gpus) <= 200:
        raise ValueError("gpu_hour_usd must map 1-200 GPU models to a USD price per GPU-hour")
    result = {"gpu_hour_usd": {}}
    for model, price in gpus.items():
        name = model.strip() if isinstance(model, str) else ""
        if not 1 <= len(name) <= 80:
            raise ValueError("GPU model names must be 1-80 characters")
        try:
            result["gpu_hour_usd"][name] = hourly_price(price)
        except ValueError as exc:
            raise ValueError(f"gpu_hour_usd {model!r}: {exc}") from None
    if table.get("cpu_hour_usd") is not None:
        try:
            result["cpu_hour_usd"] = hourly_price(table["cpu_hour_usd"])
        except ValueError as exc:
            raise ValueError(f"cpu_hour_usd: {exc}") from None
    if table.get("version") is not None:
        if not isinstance(table["version"], str) or len(table["version"]) > 100:
            raise ValueError("version must be a string of at most 100 characters")
        result["version"] = table["version"]
    return result


def load_prices(path=None):
    """The installation's prices: the static table `static_pricing: true` means, and Modal's real rates."""
    raw = json.loads(Path(path or DEFAULT_PRICES).read_text())
    table = parse_table({k: raw[k] for k in ("gpu_hour_usd", "cpu_hour_usd", "version") if k in raw})
    modal = raw.get("modal")
    if modal is not None:
        modal = {"version": modal.get("version"), "gpu_hour_usd": {name: hourly_price(rate) for name, rate in modal["gpu_hour_usd"].items()},
                 **{key: hourly_price(modal[key]) for key in ("cpu_core_hour_usd", "memory_gib_hour_usd")}}
    return {"static": table, "modal": modal}
