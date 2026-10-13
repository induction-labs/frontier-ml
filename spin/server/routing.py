"""Provider routing: a chat request may carry OpenRouter's "provider" object to choose its endpoints.

Without one, a call is capped at the model's catalog price, which only the cheapest endpoints meet, and quoted
from the catalog. With one, the cap is the dearest base price among the endpoints the selection allows, and the
call is quoted at the dearest price (long-prompt tiers included) of every endpoint under that cap, so whichever
endpoint OpenRouter picks, the charge stays within the hold. Hosted Spin implements the same rules (conav
internal/api/routing.go); tests/test_spin_routing.py checks both against the same fixture.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import logging
import re
import threading
import time

ENDPOINTS_TTL, ENDPOINTS_RETRY = 600, 60  # seconds; a failed refresh keeps serving the last list this long
MAX_LIST = 32
FIELDS = {"order", "only", "ignore", "quantizations", "sort", "allow_fallbacks", "max_price"}
SORTS = {"price", "throughput", "latency"}
QUANTIZATIONS = {"int4", "int8", "fp4", "fp6", "fp8", "fp16", "bf16", "fp32", "unknown"}
SLUG = re.compile(r"[a-z0-9][a-z0-9._/-]{0,79}")
MILLION = Decimal(1_000_000)


@dataclass(frozen=True)
class Prices:
    """USD per token (prompt, completion), per request and per image."""
    prompt: Decimal
    completion: Decimal
    request: Decimal
    image: Decimal

    def under(self, ceiling):
        return (self.prompt <= ceiling.prompt and self.completion <= ceiling.completion
                and self.request <= ceiling.request and self.image <= ceiling.image)

    def max(self, other):
        return Prices(max(self.prompt, other.prompt), max(self.completion, other.completion),
                      max(self.request, other.request), max(self.image, other.image))


@dataclass(frozen=True)
class Endpoint:
    """One OpenRouter endpoint: base holds the prices OpenRouter's max_price compares; quote, the most a call costs."""
    slug: str
    tag: str
    name: str
    quantization: str
    base: Prices
    quote: Prices

    def matches(self, selector):
        return selector in (self.tag, self.slug, self.name) or self.tag.startswith(selector + "/")


def _rate(value):
    rate = Decimal(str(0 if value is None else value))
    if not rate.is_finite() or rate < 0:
        raise ValueError("invalid price")
    return rate


def endpoint_prices(pricing):
    """(base, quote): the worst case is the highest tier, reasoning at no less than completion, web search added
    to the request fee, as catalog_policy does for the catalog. ValueError when a price cannot be bounded."""
    try:
        tiers = [pricing, *(pricing.get("overrides") or [])]
        prompt = max(_rate(t.get("prompt")) for t in tiers)
        completion = max(_rate(t.get(k)) for t in tiers for k in ("completion", "internal_reasoning"))
        request, image = _rate(pricing.get("request")), _rate(pricing.get("image"))
        base = Prices(_rate(pricing.get("prompt")), _rate(pricing.get("completion")), request, image)
        return base, Prices(prompt, completion, request + _rate(pricing.get("web_search")), image)
    except (ArithmeticError, InvalidOperation, AttributeError, TypeError) as exc:
        raise ValueError("invalid price") from exc


def parse_endpoints(model, data):
    """(endpoints, listing) from OpenRouter's /models/{id}/endpoints body; listing adds each endpoint's slug."""
    endpoints, listing = [], []
    for raw in (data.get("data") or {}).get("endpoints") or []:
        try:
            base, quote = endpoint_prices(raw.get("pricing") or {})
        except (ValueError, AttributeError) as exc:
            # One endpoint whose price cannot be bounded makes every cap unsafe.
            raise ValueError(f"an endpoint of {model} has a price that cannot be bounded") from exc
        tag = str(raw.get("tag") or "").lower()
        slug = tag.split("/", 1)[0]
        endpoints.append(Endpoint(slug, tag, str(raw.get("provider_name") or "").lower(),
                                  str(raw.get("quantization") or "").lower(), base, quote))
        listing.append({**raw, "slug": slug})
    if not endpoints:
        raise ValueError(f"OpenRouter lists no endpoints for {model}")
    return endpoints, listing


class EndpointCache:
    """A model's endpoints, fetched on first use and kept for ENDPOINTS_TTL; a failed refresh keeps the last copy."""

    def __init__(self, fetch, *, clock=time.monotonic):
        self.fetch, self.clock = fetch, clock
        self._lock, self._entries = threading.Lock(), {}

    def get(self, model):
        with self._lock:
            entry = self._entries.setdefault(model, {"lock": threading.Lock(), "list": None, "refresh_at": 0})
        with entry["lock"]:
            if entry["list"] is not None and self.clock() < entry["refresh_at"]:
                return entry["list"]
            try:
                entry["list"], entry["refresh_at"] = parse_endpoints(model, self.fetch(model)), self.clock() + ENDPOINTS_TTL
            except Exception as exc:
                if entry["list"] is None:
                    raise
                logging.warning("OpenRouter endpoints refresh for %s failed: %s", model, type(exc).__name__)
                entry["refresh_at"] = self.clock() + ENDPOINTS_RETRY
            return entry["list"]


class Routing:
    """A validated provider object."""

    def __init__(self, raw):
        if not isinstance(raw, dict):
            raise ValueError("provider must be an object")
        unknown = sorted(set(raw) - FIELDS)
        if unknown:
            raise ValueError("unsupported provider fields: " + ", ".join(unknown) + "; allowed: " + ", ".join(sorted(FIELDS)))
        self.forward, self.lists, self.fallbacks, self.max_price = {}, {}, True, {}
        for field in ("order", "only", "ignore", "quantizations"):
            if field not in raw:
                continue
            items = raw[field]
            if not isinstance(items, list) or not 1 <= len(items) <= MAX_LIST:
                raise ValueError(f"provider.{field} must be a list of 1-{MAX_LIST} strings")
            values = [item.strip().lower() if isinstance(item, str) else None for item in items]
            if field == "quantizations" and not all(v in QUANTIZATIONS for v in values):
                raise ValueError("provider.quantizations entries must be int4, int8, fp4, fp6, fp8, fp16, bf16, fp32 or unknown")
            if field != "quantizations" and not all(v is not None and SLUG.fullmatch(v) for v in values):
                raise ValueError(f'provider.{field} entries must be provider slugs such as "deepinfra" or "deepinfra/fp8"')
            self.forward[field] = self.lists[field] = values
        if "sort" in raw:
            if raw["sort"] not in SORTS:
                raise ValueError("provider.sort must be price, throughput or latency")
            self.forward["sort"] = raw["sort"]
        if "allow_fallbacks" in raw:
            if not isinstance(raw["allow_fallbacks"], bool):
                raise ValueError("provider.allow_fallbacks must be a boolean")
            self.forward["allow_fallbacks"] = self.fallbacks = raw["allow_fallbacks"]
        if "max_price" in raw:
            limits = raw["max_price"]
            if not isinstance(limits, dict) or not limits:
                raise ValueError("provider.max_price must be an object with prompt, completion, request or image")
            for key, value in limits.items():
                if key not in {"prompt", "completion", "request", "image"}:
                    raise ValueError("provider.max_price accepts prompt and completion (USD per million tokens), "
                                     "request and image (USD each)")
                try:
                    price = Decimal(str(value)) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
                except InvalidOperation:
                    price = None
                if price is None or not price.is_finite() or price < 0:
                    raise ValueError("provider.max_price values must be non-negative numbers")
                self.max_price[key] = price / MILLION if key in {"prompt", "completion"} else price

    def allows(self, endpoint):
        """Spin's reading of only, ignore, quantizations, order without fallbacks, and max_price."""
        def matches(field):
            return any(endpoint.matches(selector) for selector in self.lists.get(field, ()))
        if ("only" in self.lists and not matches("only")) or matches("ignore"):
            return False
        if not self.fallbacks and "order" in self.lists and not matches("order"):
            return False
        if "quantizations" in self.lists and endpoint.quantization not in self.lists["quantizations"]:
            return False
        return all(getattr(endpoint.base, key) <= limit for key, limit in self.max_price.items())

    def price(self, model, endpoints):
        """(ceiling, quote): the max_price ceiling and the prices to quote at. The quote covers every endpoint under
        the ceiling, not only those Spin reads the selection as allowing, so a difference in how OpenRouter matches
        names cannot route a call above its hold."""
        allowed = [e.base for e in endpoints if self.allows(e)]
        if not allowed:
            raise ValueError(f"no endpoint of {model} matches the provider selection; see spin openrouter endpoints {model}")
        ceiling = allowed[0]
        for base in allowed[1:]:
            ceiling = ceiling.max(base)
        under = [e.quote for e in endpoints if e.base.under(ceiling)]
        quote = under[0]
        for prices in under[1:]:
            quote = quote.max(prices)
        return ceiling, quote
