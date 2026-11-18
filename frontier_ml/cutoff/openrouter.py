"""OpenRouter models as of the cutoff, in front of an OpenAI-compatible upstream (Spin's inference API by default).

A model is available when OpenRouter's catalog lists it with a `created` time at or before the cutoff. The catalog
is OpenRouter's public /models response, polled, or a saved copy of it: a saved copy freezes the list for good,
while the live one also loses models OpenRouter retires. The client's own credentials go upstream untouched, so
spending is accounted exactly as without the proxy.

Requests that would reach newer models or the live web another way are refused: router pseudo-models
(openrouter/auto and friends), the `:online` variant, the web plugin, and web search tools and options.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time

from fastapi.responses import Response

from .core import Refused, forward, request_headers

INFERENCE = {"chat/completions", "completions", "embeddings", "responses"}
PASSTHROUGH = {"key", "credits", "generation"}  # account and usage lookups a client may make


class OpenRouter:
    def __init__(self, settings, client, cache, log):
        self.settings, self.client, self.log = settings, client, log
        self.created: dict[str, float] = {}
        self.loaded = 0.0
        self.lock = asyncio.Lock()

    async def handle(self, request, raw_path: str, run: str, mount: str):
        path = raw_path.strip("/").removeprefix("v1/").removeprefix("v1")
        try:
            return await self._handle(request, path, run)
        except Refused as exc:
            self.log.write(run, "openrouter", "denied", path, reason=exc.reason, status=exc.status,
                           model=getattr(exc, "model", None))
            raise

    async def _handle(self, request, path, run):
        method = request.method
        await self._catalog()
        if path == "models" and method == "GET":
            return await self._models(request, run)
        if path.startswith("models/") and method == "GET":
            # /models/<author>/<slug>/endpoints
            self.check(path.removeprefix("models/").removesuffix("/endpoints"))
            return await self._send(request, path)
        if path in INFERENCE and method == "POST":
            body = await request.body()
            try:
                payload = json.loads(body)
            except ValueError:
                raise Refused(400, "request body must be JSON", reason="bad_request") from None
            models = self.inspect(payload)
            self.log.write(run, "openrouter", "inference", path, model=models[0],
                           fallbacks=models[1:] or None, stream=bool(payload.get("stream")) or None)
            return await self._send(request, path, body)
        if path in PASSTHROUGH and method == "GET":
            return await self._send(request, path)
        raise Refused(404, f"/{path} is not served by the cutoff proxy", reason="unsupported")

    def check(self, model) -> str:
        """Refuse a model that was not listed by the cutoff; returns the catalog id it was matched under."""
        cutoff = self.settings.cutoff_text
        if not isinstance(model, str) or not model:
            raise Refused(400, "name a model explicitly: the account default is not pinned to the cutoff",
                          reason="model_missing")
        base, _, variant = model.partition(":")
        problem = None
        if base.startswith("openrouter/"):
            problem = ("router_model", f"{model} routes to whichever models are current; name a specific model")
        elif variant == "online":
            problem = ("web_search", f"{model}: the :online variant searches the live web")
        elif base not in self.created:
            problem = ("model_unknown", f"{model} is not a model listed on OpenRouter as of {cutoff}")
        elif not self.settings.before(self.created[base]):
            problem = ("model_after_cutoff", f"{model} was listed after {cutoff} and is not available in this run "
                                             f"(this run sees OpenRouter as of that date)")
        if problem:
            exc = Refused(404, problem[1], reason=problem[0])
            exc.model = model
            raise exc
        return base

    def inspect(self, payload) -> list[str]:
        """Check everything in an inference request that selects a model or reaches the web; returns the models."""
        if not isinstance(payload, dict):
            raise Refused(400, "request body must be a JSON object", reason="bad_request")
        models = [payload.get("model"), *(payload.get("models") or [])]
        for model in models:
            self.check(model)
        plugins = payload.get("plugins") or []
        if any(isinstance(plugin, dict) and plugin.get("id") == "web" for plugin in plugins):
            raise Refused(403, "the web plugin searches the live web and is not available in this run", reason="web_search")
        if payload.get("web_search_options") is not None:
            raise Refused(403, "web_search_options searches the live web and is not available in this run", reason="web_search")
        for tool in payload.get("tools") or []:
            if isinstance(tool, dict) and "web_search" in str(tool.get("type", "")):
                raise Refused(403, "web search tools are not available in this run", reason="web_search")
        return models

    async def _models(self, request, run):
        """The upstream's own list (it may offer fewer models than OpenRouter does), minus anything after the cutoff."""
        response = await self.client.get(f"{self.settings.openrouter_upstream}/models", params=request.query_params,
                                         headers=request_headers(request, accept_encoding=None))
        try:
            body = response.json()
        except ValueError:
            body = None
        if response.status_code != 200 or not isinstance(body, dict) or not isinstance(body.get("data"), list):
            return Response(response.content, response.status_code, media_type=response.headers.get("content-type"))
        kept = [item for item in body["data"] if self._listed(item)]
        self.log.write(run, "openrouter", "models", "models", listed=len(body["data"]), kept=len(kept))
        return Response(json.dumps({**body, "data": kept}), media_type="application/json")

    def _listed(self, item) -> bool:
        try:
            self.check(item.get("id") if isinstance(item, dict) else None)
        except Refused:
            return False
        return True

    async def _send(self, request, path, body=None):
        url = f"{self.settings.openrouter_upstream}/{path}" + (f"?{request.url.query}" if request.url.query else "")
        return await forward(self.client, request.method, url, content=body, headers=request_headers(request))

    async def _catalog(self):
        """Listing dates by model id (and by dated canonical slug, which clients may also send)."""
        source = self.settings.openrouter_catalog
        remote = source.startswith(("http://", "https://"))
        if self.created and (not remote or time.monotonic() - self.loaded < self.settings.catalog_seconds):
            return
        async with self.lock:
            if self.created and (not remote or time.monotonic() - self.loaded < self.settings.catalog_seconds):
                return
            try:
                if remote:
                    response = await self.client.get(source, follow_redirects=True)
                    response.raise_for_status()
                    data = response.json()["data"]
                else:
                    data = json.loads(Path(source).read_text())["data"]
            except Exception as exc:
                if self.created:
                    self.loaded = time.monotonic()  # keep the last good catalog; try again after the usual interval
                    return
                raise Refused(503, f"model catalog unavailable: {type(exc).__name__}", reason="catalog_unavailable") from exc
            created = {}
            for item in data:
                if isinstance(item.get("created"), (int, float)):
                    created[item["id"]] = item["created"]
                    if item.get("canonical_slug"):
                        created.setdefault(item["canonical_slug"], item["created"])
            self.created, self.loaded = created, time.monotonic()
