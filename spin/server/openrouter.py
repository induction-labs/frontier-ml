"""OpenRouter proxy: reserve a trusted worst-case quote before each paid call, settle at OpenRouter's usage.cost.

A call that certainly never reached OpenRouter, or that it refused or dropped before any reply, is released
and its Idempotency-Key may be retried. A call whose cost never arrived keeps its hold; the reaper settles it
from OpenRouter's generation record, or at the hold after an hour.
"""
from __future__ import annotations

import base64
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
import hashlib
import json
import logging
import threading
import time

import httpx

from .broker import ApiError
from .ledger import BudgetExceeded, Conflict, IdempotencyConflict, NotFound
from .providers.base import ProviderError, ProviderRejectedError, ProviderUncertainError
from .routing import MILLION, EndpointCache, Prices, Routing

BASE_URL = "https://openrouter.ai/api/v1"
UNCERTAIN_GRACE_SECONDS = 600      # let an in-flight call finish before reconciling it
UNPROVEN_SETTLE_SECONDS = 3600     # no reply at all: keep the reserved amount after an hour
# Reserved output when a request omits max_tokens; the model maximum can be 100k+.
DEFAULT_MAX_TOKENS = 8192
# Chat templates and tool prompts add tokens beyond the request bytes.
TEMPLATE_TOKENS, MESSAGE_TOKENS = 2048, 32
# Image prompt bound: at most about one token per 14x14 pixel patch, never below a floor that covers fixed
# per-image budgets; an image whose header gives no size (or a side over 65535) reserves full context.
IMAGE_PATCH_PIXELS, IMAGE_TOKEN_FLOOR, MAX_IMAGE_SIDE = 14 * 14, 4096, 65535
BODY_LIMIT = 32 << 20
STREAMED = "This key was used for a streamed request; use a new Idempotency-Key"


def image_size(b):
    """(width, height) from a PNG, JPEG or WebP header, or None."""
    w = h = 0
    if len(b) >= 24 and b[:8] == b"\x89PNG\r\n\x1a\n" and b[12:16] == b"IHDR":
        w, h = int.from_bytes(b[16:20], "big"), int.from_bytes(b[20:24], "big")
    elif len(b) >= 4 and b[0] == 0xFF and b[1] == 0xD8:
        i = 2
        while i + 1 < len(b):
            if b[i] != 0xFF:
                return None
            m = b[i + 1]
            if m == 0xFF:  # fill byte
                i += 1
                continue
            if m == 0x01 or 0xD0 <= m <= 0xD8:  # markers without a length
                i += 2
                continue
            if m in (0xD9, 0xDA):  # end of image or start of scan before any frame header
                return None
            if i + 4 > len(b):
                return None
            if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):  # SOFn
                if i + 9 > len(b):
                    return None
                h, w = int.from_bytes(b[i + 5:i + 7], "big"), int.from_bytes(b[i + 7:i + 9], "big")
                break
            i += 2 + int.from_bytes(b[i + 2:i + 4], "big")
    elif len(b) >= 30 and b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        kind = b[12:16]
        if kind == b"VP8 ":
            if b[23:26] != b"\x9d\x01\x2a":
                return None
            w, h = int.from_bytes(b[26:28], "little") & 0x3FFF, int.from_bytes(b[28:30], "little") & 0x3FFF
        elif kind == b"VP8L":
            if b[20] != 0x2F:
                return None
            bits = int.from_bytes(b[21:25], "little")
            w, h = (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        elif kind == b"VP8X":
            w, h = int.from_bytes(b[24:27], "little") + 1, int.from_bytes(b[27:30], "little") + 1
    if 0 < w <= MAX_IMAGE_SIDE and 0 < h <= MAX_IMAGE_SIDE:
        return w, h
    return None


def image_tokens(data):
    """Prompt-token bound for one inline image, or None when its size cannot be read."""
    size = image_size(data)
    if size is None:
        return None
    return max(IMAGE_TOKEN_FLOOR, -(-size[0] * size[1] // IMAGE_PATCH_PIXELS))


def _rate(value):
    rate = Decimal(str(value))
    if not rate.is_finite() or rate < 0:
        raise ValueError("invalid price")
    return rate


def catalog_policy(listing):
    """Reservation policy for one OpenRouter model listing, or None if it can't be bounded."""
    architecture = listing.get("architecture") or {}
    if architecture.get("output_modalities", ["text"]) != ["text"]:
        return None  # Image/audio output is billed per artifact, outside the token quote.
    pricing = listing.get("pricing") or {}
    tiers = [pricing, *(pricing.get("overrides") or [])]
    try:
        # Long-prompt tiers cost more; reserve at the highest tier.
        prompt = max(_rate(t.get("prompt", 0)) for t in tiers)
        completion = max(_rate(t.get(k, 0)) for t in tiers for k in ("completion", "internal_reasoning"))
        request = _rate(pricing.get("request", 0)) + _rate(pricing.get("web_search", 0))
        image = _rate(pricing.get("image", 0))
    except (ArithmeticError, ValueError, TypeError):
        return None  # e.g. routers priced at -1
    context = listing.get("context_length")
    output = (listing.get("top_provider") or {}).get("max_completion_tokens") or context
    if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (context, output)):
        return None
    images = "image" in (architecture.get("input_modalities") or [])
    million = Decimal(1_000_000)
    return {"context_tokens": context, "max_output_tokens": output, "default_output_tokens": min(output, DEFAULT_MAX_TOKENS),
            "input_usd_per_million": prompt * million, "output_usd_per_million": completion * million,
            "request_usd": request, "image_usd": image, "allow_images": images, "max_images": 16 if images else 0}


class OpenRouterCatalog(Mapping):
    """Every OpenRouter text model (or the allowed ones), priced from OpenRouter's public model list."""

    def __init__(self, fetch, *, allowed=None, ttl=3600, retry=60, clock=time.monotonic):
        self.fetch, self.allowed, self.ttl, self.retry, self.clock = fetch, allowed, ttl, retry, clock
        self._policies, self._refresh_at = None, 0
        self._lock = threading.Lock()

    def _current(self):
        with self._lock:
            if self._policies is None or self.clock() >= self._refresh_at:
                try:
                    policies = {m["id"]: p for m in self.fetch() if (not self.allowed or m.get("id") in self.allowed)
                                and (p := catalog_policy(m))}
                    if not policies:
                        raise ValueError("empty OpenRouter catalog")
                    self._policies, self._refresh_at = policies, self.clock() + self.ttl
                except Exception as exc:
                    if self._policies is None:
                        raise ProviderError("The OpenRouter model catalog is unavailable") from exc
                    logging.warning("OpenRouter catalog refresh failed: %s", type(exc).__name__)
                    self._refresh_at = self.clock() + self.retry
            return self._policies

    def __getitem__(self, model):
        return self._current()[model]

    def __iter__(self):
        return iter(self._current())

    def __len__(self):
        return len(self._current())


def usage_cost(event):
    """usage.cost from a completion or stream chunk, as a Decimal, or None when absent or malformed."""
    usage = event.get("usage") if isinstance(event, dict) else None
    cost = usage.get("cost") if isinstance(usage, dict) else None
    if cost is None or isinstance(cost, bool) or not isinstance(cost, (int, float, str)):
        return None
    try:
        amount = Decimal(str(cost))
    except InvalidOperation:
        return None
    return amount if amount.is_finite() and amount >= 0 else None


def _price(value):
    """A decimal price as OpenRouter writes it ("0.000002")."""
    text = format(value, "f")
    return (text.rstrip("0").rstrip(".") if "." in text else text) or "0"


def upstream_message(status, raw):
    try:
        message = json.loads(raw)["error"]["message"]
    except Exception:
        message = "HTTP " + str(status)
    return str(message)[:300]


class OpenRouterProxy:
    def __init__(self, ledger, api_key, models=None, *, allowed=None, client=None, base_url=BASE_URL):
        """models: {id: policy}, by default OpenRouter's catalog (only the `allowed` ids, when given)."""
        self.ledger, self.api_key, self.base_url = ledger, api_key, base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=httpx.Timeout(600, connect=30), follow_redirects=False)
        self.models = models if models is not None else OpenRouterCatalog(self._catalog, allowed=allowed)
        self.endpoints = EndpointCache(self._endpoints)
        self.in_flight = set()  # Reservations this process is still talking to OpenRouter about.

    def _catalog(self):
        response = self.client.get(self.base_url + "/models", timeout=30)
        response.raise_for_status()
        return response.json()["data"]

    def _endpoints(self, model):
        response = self.client.get(self.base_url + "/models/" + model + "/endpoints", timeout=30)
        response.raise_for_status()
        return response.json()

    def _model_endpoints(self, model):
        try:
            return self.endpoints.get(model)
        except Exception:
            raise ApiError(503, "endpoints_unavailable", f"OpenRouter's endpoint list for {model} is unavailable; "
                                                         "retry shortly, or call without provider") from None

    def endpoint_listing(self, rest):
        """GET /v1/models/{model}/endpoints: OpenRouter's endpoints of a catalog model, with each endpoint's slug
        and whether a call without a provider object (capped at the catalog price) can reach it."""
        if not rest.endswith("/endpoints"):
            raise NotFound("Use /v1/models/{model}/endpoints")
        model = rest[:-len("/endpoints")]
        if model not in self.models:
            raise NotFound("Unknown model; list available models with spin openrouter models")
        policy = self.models[model]
        endpoints, listing = self._model_endpoints(model)
        cap = Prices(policy["input_usd_per_million"] / MILLION, policy["output_usd_per_million"] / MILLION,
                     policy["request_usd"], Decimal(0))
        return {"model": model, "default_cap": {"prompt": _price(cap.prompt), "completion": _price(cap.completion),
                                                "request": _price(cap.request)},
                "endpoints": [{**shown, "within_default_cap": Prices(e.base.prompt, e.base.completion, e.base.request,
                                                                     Decimal(0)).under(cap)}
                              for e, shown in zip(endpoints, listing)]}

    def reconcile(self, reservation, now=None):
        """Settle a model call whose cost was never confirmed, so it cannot hold a wallet open.

        With a reply, OpenRouter's generation record has the real cost. Without one there
        is no way to know what was billed, so after an hour the reserved amount is kept.
        Returns the settled cost, or None if it is too early to decide.
        """
        now = time.time() if now is None else now
        age = now - reservation["created"]
        if reservation["state"] == "settled" or age < UNCERTAIN_GRACE_SECONDS or reservation["id"] in self.in_flight:
            return None
        generation = (reservation["result"] or {}).get("id")
        cost = None
        if isinstance(generation, str) and generation:
            try:
                response = self.client.get(self.base_url + "/generation", params={"id": generation}, timeout=30,
                                           headers={"Authorization": "Bearer " + self.api_key})
                response.raise_for_status()
                data = response.json().get("data") or {}
                cost = data.get("total_cost", data.get("usage"))
            except Exception:
                cost = None  # The record may not exist yet; try again next pass.
        if cost is None and age >= UNPROVEN_SETTLE_SECONDS:
            cost = reservation["maximum_usd"]
        if cost is None:
            return None
        try:  # A cost above the hold is still recorded (and freezes the wallet), like a live call.
            self.ledger.settle(reservation["id"], cost)
        except BudgetExceeded:
            pass
        return str(cost)

    def prepare(self, body):
        """The upstream payload, the worst-case quote in USD, and whether the call streams."""
        allowed = {"model", "messages", "max_tokens", "max_completion_tokens", "temperature", "top_p", "stop", "seed",
                   "tools", "tool_choice", "response_format", "stream", "stream_options", "provider"}
        if set(body) - allowed:
            raise ValueError("unsupported request fields: " + ", ".join(sorted(set(body) - allowed)))
        stream, options = body.get("stream", False), body.get("stream_options") or {}
        if not isinstance(stream, bool) or not isinstance(options, dict):
            raise ValueError("stream must be a boolean and stream_options an object")
        model = body.get("model")
        if not isinstance(model, str) or model not in self.models:
            raise ValueError("Unknown model; list available models with spin openrouter models")
        policy = self.models[model]
        route = Routing(body["provider"]) if "provider" in body else None
        maximum = body.get("max_tokens", body.get("max_completion_tokens", policy["default_output_tokens"]))
        if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= policy["max_output_tokens"]:
            raise ValueError(f"max_tokens must be an integer in [1, {policy['max_output_tokens']}] for this model")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty list")
        # Only text billing is supported by default. No server-side tools, web searches,
        # remote images, audio, plugins, cache-control surcharges or unbounded output.
        image_count = image_prompt = image_url_bytes = 0
        sized = True
        for message in messages:
            if not isinstance(message, dict) or set(message) - {"role", "content", "name", "tool_calls", "tool_call_id"}:
                raise ValueError("unsupported message fields")
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        raise ValueError("invalid content part")
                    if part.get("type") == "text" and set(part) == {"type", "text"} and isinstance(part["text"], str):
                        continue
                    if not policy["allow_images"] or part.get("type") != "image_url" or set(part) != {"type", "image_url"}:
                        raise ValueError("content type is not enabled by host pricing policy")
                    image = part["image_url"]
                    if not isinstance(image, dict) or set(image) - {"url", "detail"}:
                        raise ValueError("invalid image object")
                    url = image.get("url", "")
                    if not isinstance(url, str) or not url.startswith(("data:image/png;base64,", "data:image/jpeg;base64,", "data:image/webp;base64,")):
                        raise ValueError("images must be inline PNG/JPEG/WebP data URLs")
                    raw = base64.b64decode(url.split(",", 1)[1], validate=True)
                    image_count += 1
                    image_url_bytes += len(url)
                    tokens = image_tokens(raw)
                    if tokens is None:
                        sized = False
                    else:
                        image_prompt += tokens
            elif not (isinstance(content, str) or content is None):
                raise ValueError("invalid message content")
        if image_count > policy["max_images"]:
            raise ValueError("too many images for host pricing policy")
        tools = body.get("tools", [])
        if not isinstance(tools, list) or any(not isinstance(tool, dict) or tool.get("type") != "function"
                or set(tool) != {"type", "function"} or not isinstance(tool["function"], dict) for tool in tools):
            raise ValueError("only client-executed function tools are permitted")
        input_rate, output_rate = Decimal(policy["input_usd_per_million"]), Decimal(policy["output_usd_per_million"])
        fee, image_fee = Decimal(policy["request_usd"]), Decimal(policy["image_usd"])
        # Any endpoint at or below the ceiling keeps the charge within the quote.
        provider = {"require_parameters": True, "max_price": {"prompt": float(input_rate), "completion": float(output_rate),
                                                              "request": float(fee)}}
        if route is not None:
            ceiling, worst = route.price(model, self._model_endpoints(model)[0])
            input_rate, output_rate, fee, image_fee = worst.prompt * MILLION, worst.completion * MILLION, worst.request, worst.image
            provider = {"require_parameters": True, "max_price": {
                "prompt": float(ceiling.prompt * MILLION), "completion": float(ceiling.completion * MILLION),
                "request": float(ceiling.request), "image": float(ceiling.image)}, **route.forward}
        # Upstream rejects prompts over the context limit. Every token covers at least one byte of
        # UTF-8 input, so the request size, less the images' data URLs, bounds the text without a
        # tokenizer. Images are bounded by their pixel size; one of unreadable size reserves full context.
        context = policy["context_tokens"]
        prompt_tokens = context
        if sized:
            size = len(json.dumps([messages, tools, body.get("response_format")], ensure_ascii=False).encode()) - image_url_bytes
            prompt_tokens = min(context, size + MESSAGE_TOKENS * len(messages) + TEMPLATE_TOKENS + image_prompt)
        quote = (prompt_tokens * input_rate + maximum * output_rate) / Decimal(1_000_000) + fee + image_count * image_fee
        payload = dict(body)
        payload.pop("max_completion_tokens", None)
        payload.pop("stream_options", None)
        payload.update(max_tokens=maximum, stream=stream, usage={"include": True})
        if stream:
            payload["stream_options"] = {**options, "include_usage": True}
        payload["provider"] = provider
        return payload, quote, stream

    def complete(self, wallet_id, key, body):
        """A completion dict, or a Stream for the server to relay."""
        payload, quote, stream = self.prepare(body)
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        reservation, created = self.ledger.reserve(wallet_id, "openrouter:" + key, quote, "openrouter",
                                                   {"request_sha256": digest, "model": payload["model"]})
        if not created:
            result = reservation["result"]
            if result and result.get("spin_streamed"):
                raise IdempotencyConflict(STREAMED)
            if result is not None:
                return result
            raise Conflict("request is in progress or billing is uncertain; do not retry with a new key")
        self.in_flight.add(reservation["id"])
        streaming = False
        try:
            request = self.client.build_request("POST", self.base_url + "/chat/completions", json=payload, headers={
                "Authorization": "Bearer " + self.api_key, "X-Title": "Spin",
                **({"Accept": "text/event-stream"} if stream else {})})
            try:
                response = self.client.send(request, stream=True)
            except httpx.ReadTimeout:
                # OpenRouter was still working on the call: it may have served and billed it.
                raise ProviderUncertainError("Upstream model request timed out after it was sent; use a new Idempotency-Key") from None
            except httpx.TransportError as exc:
                # Not sent, or dropped before any reply: treated as unbilled, like a rejection.
                self.ledger.discard(reservation["id"])
                raise ProviderError(f"Upstream model request failed before any reply ({type(exc).__name__}); nothing was charged") from None
            if response.status_code != 200:
                raw = _read(response, 1 << 20, tolerant=True)
                self.ledger.discard(reservation["id"])  # OpenRouter does not bill requests it refuses.
                if response.status_code in {401, 402, 403}:
                    logging.error("OpenRouter rejected the operator key: HTTP %s", response.status_code)
                    raise ProviderRejectedError("Inference is temporarily unavailable on this installation")
                if 400 <= response.status_code < 500:
                    raise ProviderRejectedError("Model provider rejected the request: " + upstream_message(response.status_code, raw))
                raise ProviderError(f"Upstream model request failed (HTTP {response.status_code})")
            if stream and response.headers.get("content-type", "").startswith("text/event-stream"):
                streaming = True  # The caller must relay it; the relay settles and leaves in_flight.
                return Stream(self, reservation, response)
            try:
                result = json.loads(_read(response, BODY_LIMIT))
            except (httpx.HTTPError, ValueError):
                raise ProviderUncertainError("Upstream model response failed after it started; use a new Idempotency-Key") from None
            return self._settle(reservation, result)
        except BaseException:
            self.ledger.mark_uncertain(reservation["id"])  # A no-op once settled or discarded.
            raise
        finally:
            if not streaming:
                self.in_flight.discard(reservation["id"])

    def _settle(self, reservation, result, cost=None):
        cost = usage_cost(result) if cost is None else cost
        if cost is None:
            # Keep the reply: reconcile() reads its generation id to learn the real cost later.
            self.ledger.set_result(reservation["id"], result)
            self.ledger.mark_uncertain(reservation["id"])
            return result
        try:
            self.ledger.settle(reservation["id"], cost, result=result)
        except BudgetExceeded:
            pass  # Recorded above the hold; the wallet is frozen.
        return result


def _read(response, limit, *, tolerant=False):
    data = b""
    try:
        for chunk in response.iter_bytes():
            data += chunk
            if len(data) > limit:
                raise ValueError("upstream response exceeds the size limit")
    except (httpx.HTTPError, ValueError):
        if not tolerant:
            raise
    finally:
        response.close()
    return data


class Stream:
    """An upstream SSE reply, relayed event by event. The call settles when OpenRouter finishes, client or not."""

    def __init__(self, proxy, reservation, response):
        self.proxy, self.reservation, self.response = proxy, reservation, response

    def relay(self, write):
        """Pass events to write(bytes); a failing write only stops forwarding, never the upstream drain."""
        cost = generation = None
        pending, event, total = b"", [], 0

        def send(data):
            nonlocal write
            if write and data:
                try:
                    write(data)
                except (OSError, ValueError):  # Gone, or its socket already closed.
                    write = None

        try:
            for chunk in self.response.iter_bytes():
                total += len(chunk)
                if total > BODY_LIMIT:
                    break
                *lines, pending = (pending + chunk).split(b"\n")
                for line in lines:
                    event.append(line + b"\n")
                    if not line.strip(b"\r"):  # An empty line ends an SSE event.
                        send(b"".join(event))
                        event = []
                    elif line.startswith(b"data:") and line[5:].strip() not in (b"", b"[DONE]"):
                        try:
                            data = json.loads(line[5:])
                        except ValueError:
                            continue
                        if isinstance(data, dict):
                            generation = generation or data.get("id")
                            found = usage_cost(data)
                            cost = cost if found is None else found
        except httpx.HTTPError:
            pass  # Cut off: without a usage chunk the reaper reconciles the call.
        finally:
            self.response.close()
            send(b"".join(event) + pending)
            result = {"id": generation, "spin_streamed": True}
            try:
                if cost is None:
                    self.proxy.ledger.set_result(self.reservation["id"], result)
                    self.proxy.ledger.mark_uncertain(self.reservation["id"])
                else:
                    self.proxy._settle(self.reservation, result, cost)
            except Conflict:
                logging.warning("Streamed call %s was settled elsewhere first", self.reservation["id"])
            finally:
                self.proxy.in_flight.discard(self.reservation["id"])
