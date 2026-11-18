"""The Hugging Face Hub as of the cutoff, for clients pointed here with HF_ENDPOINT.

Every request that names a revision is forwarded with that revision replaced by a commit: the newest one on the
requested branch dated at or before the cutoff (a commit hash or tag is checked against the cutoff instead). A
repository created after the cutoff answers RepoNotFound, as if it did not exist. Searches drop repositories created
later. File bytes are not proxied: the Hub redirects large files to its content-addressed CDN, and the client
follows that redirect itself.

Anything the proxy has no cutoff rule for (git over HTTPS, the dataset viewer's parquet API, write endpoints, web
pages) is refused and logged, so a gap shows up as a `denied` line in the access log instead of as a leak.
"""
from __future__ import annotations

import json
import re
from urllib.parse import quote, unquote, urlsplit

from fastapi.responses import Response

from .core import Refused, forward, request_headers

KINDS = ("models", "datasets", "spaces")
# Path words that follow a repository name in /api/<kind>/<repo>/...; a one-part (legacy) repo name is told from
# <owner>/<name> by seeing one of these in second position.
ACTIONS = {"revision", "tree", "commits", "paths-info", "xet-read-token", "refs"}
SHA = re.compile(r"[0-9a-f]{40}")
PAGE = 100


class HuggingFace:
    def __init__(self, settings, client, cache, log):
        self.settings, self.client, self.cache, self.log = settings, client, cache, log

    async def handle(self, request, raw_path: str, run: str, mount: str):
        """`raw_path` is the still-encoded path below the mount (revisions arrive as refs%2Fpr%2F1); `mount` is the
        public URL prefix that stands in for https://huggingface.co in redirects and Link headers."""
        raw = [part for part in raw_path.split("/") if part]
        parts = [unquote(part) for part in raw]
        context = _Context(self, request, run, mount, raw)
        try:
            if parts[:1] == ["api"]:
                return await self._api(context, parts[1:])
            return await self._resolve(context, parts)
        except Refused as exc:
            self.log.write(run, "hf", "denied", "/" + "/".join(parts), reason=exc.reason, status=exc.status)
            raise

    async def _api(self, context, parts):
        method = context.request.method
        if parts == ["whoami-v2"] and method == "GET":
            return await context.send("/api/whoami-v2")
        if len(parts) == 1 and parts[0] in KINDS and method == "GET":
            return await self._search(context, parts[0])
        if parts[:1] == ["resolve-cache"] and len(parts) >= 5 and parts[1] in KINDS and method in ("GET", "HEAD"):
            # Where /resolve/ sends small files: /api/resolve-cache/<kind>/<repo>/<commit>/<path>.
            index = next((i for i in (3, 4) if i < len(parts) and SHA.fullmatch(parts[i])), None)
            if index is None:
                raise Refused(403, "this endpoint is not served by the cutoff proxy", reason="unsupported")
            repo = "/".join(parts[2:index])
            await self.pin(context, parts[1], repo, parts[index])
            return await context.send("/api/" + "/".join(context.raw[1:]))
        if len(parts) < 2 or parts[0] not in KINDS:
            raise Refused(403, "this endpoint is not served by the cutoff proxy", reason="unsupported")
        kind, rest = parts[0], parts[1:]
        size = 1 if len(rest) == 1 or rest[1] in ACTIONS else 2
        repo, rest = "/".join(rest[:size]), rest[size:]
        base = f"/api/{kind}/{quote(repo)}"
        if not rest and method == "GET":  # repository info at the default branch
            commit = await self.pin(context, kind, repo, "main")
            return await context.send(f"{base}/revision/{commit}", pinned=(kind, repo, "main", commit))
        action = rest[0] if rest else None
        if action == "refs" and len(rest) == 1 and method == "GET":
            return await self._refs(context, kind, repo)
        allowed = {"revision": ("GET",), "tree": ("GET",), "commits": ("GET",), "paths-info": ("POST",),
                   "xet-read-token": ("GET",)}
        if action not in allowed or method not in allowed[action] or len(rest) < 2:
            raise Refused(403, "this endpoint is not served by the cutoff proxy", reason="unsupported")
        commit = await self.pin(context, kind, repo, rest[1])
        tail = "".join("/" + quote(part) for part in rest[2:])
        return await context.send(f"{base}/{action}/{commit}{tail}", pinned=(kind, repo, rest[1], commit))

    async def _resolve(self, context, parts):
        """/<repo>/resolve/<revision>/<path> for models, with a datasets/ or spaces/ prefix for the others."""
        if context.request.method not in ("GET", "HEAD"):
            raise Refused(403, "the cutoff proxy is read-only", reason="unsupported")
        kind = parts[0] if parts[:1] and parts[0] in KINDS[1:] else "models"
        rest = parts[1:] if kind != "models" else parts
        index = next((i for i in (1, 2) if i < len(rest) and rest[i] == "resolve"), None)
        if index is None or len(rest) < index + 3:
            raise Refused(403, "only the Hub API and /resolve/ downloads are served by the cutoff proxy "
                               "(use `hf download` or huggingface_hub instead of git or the website)",
                          reason="unsupported")
        repo, revision, path = "/".join(rest[:index]), rest[index + 1], rest[index + 2:]
        commit = await self.pin(context, kind, repo, revision)
        prefix = "" if kind == "models" else f"/{kind}"
        target = f"{prefix}/{quote(repo)}/resolve/{commit}/" + "/".join(quote(part) for part in path)
        return await context.send(target, pinned=(kind, repo, revision, commit), entry="/".join(path))

    async def pin(self, context, kind, repo, revision) -> str:
        """The commit that `revision` of the repository meant at the cutoff."""
        cutoff = self.settings.cutoff_text
        if (given := self.settings.allowed_commit(context.run, kind, repo)) is not None:
            # The task's own starting weights: the pinned commit stands in for the branch, and nothing else is served.
            if revision in ("main", given):
                return given
            raise Refused(404, f"Revision not found: only {given[:12]} of {repo} is available in this run.",
                          code="RevisionNotFound", reason="revision_not_allowed")
        created = await self._created(context, kind, repo)
        if not self.settings.before(created):
            raise Refused(404, f"Repository not found: {repo} did not exist on {cutoff} (this run sees Hugging Face "
                               f"as of that date).", code="RepoNotFound", reason="repo_after_cutoff")
        key = f"hf:pin:{cutoff}:{kind}:{repo}:{revision}"
        if (commit := self.cache.get(key)) is not None:
            return commit
        path = f"/api/{kind}/{quote(repo)}/commits/{quote(revision, safe='')}"
        url = f"{self.settings.hf_upstream}{path}?limit={1 if SHA.fullmatch(revision) else PAGE}"
        commit = None
        while url and commit is None:
            response = await self.client.get(url, headers=context.lookup_headers, follow_redirects=True)
            if response.status_code in (401, 403) and response.headers.get("x-error-code") == "GatedRepo":
                # Not a cutoff matter: the token has not been granted this gated repository. Say what the Hub says.
                raise Refused(403, response.headers.get("x-error-message", f"Access to {repo} is restricted."),
                              code="GatedRepo", reason="repo_gated")
            if response.status_code in (401, 403, 404):
                raise Refused(404, f"Revision not found: {revision} in {repo}.", code="RevisionNotFound",
                              reason="revision_missing")
            response.raise_for_status()
            page = response.json()
            if SHA.fullmatch(revision):
                # History is listed from the commit itself, so the first entry dates it.
                if not page or not self.settings.before(page[0].get("date")):
                    raise Refused(404, f"Revision not found: {revision[:12]} of {repo} is newer than {cutoff}.",
                                  code="RevisionNotFound", reason="revision_after_cutoff")
                commit = revision
                break
            commit = next((entry["id"] for entry in page if self.settings.before(entry.get("date"))), None)
            url = response.links.get("next", {}).get("url") if commit is None else None
        if commit is None:
            raise Refused(404, f"Revision not found: {revision} of {repo} has no commit before {cutoff}.",
                          code="RevisionNotFound", reason="revision_after_cutoff")
        if self.settings.settled:
            self.cache.set(key, commit)
        return commit

    async def _created(self, context, kind, repo):
        key = f"hf:created:{kind}:{repo}"
        if (created := self.cache.get(key)) is not None:
            return created
        response = await self.client.get(f"{self.settings.hf_upstream}/api/{kind}/{quote(repo)}",
                                         headers=context.lookup_headers, follow_redirects=True)
        if response.status_code in (401, 404):
            # The Hub's own answer for a missing or private repository; not cached, it may be created or opened later.
            raise Refused(404, f"Repository not found: {repo}.", code="RepoNotFound", reason="repo_missing")
        if response.status_code == 403:
            raise Refused(403, response.headers.get("x-error-message", f"Access to {repo} is restricted."),
                          code=response.headers.get("x-error-code"), reason="repo_restricted")
        response.raise_for_status()
        created = response.json().get("createdAt")
        if created:
            self.cache.set(key, created)
        return created

    async def _search(self, context, kind):
        """/api/<kind>?search=...: the Hub's answer without repositories created after the cutoff. Pages can come
        back shorter than `limit`; the Link header still walks the full result set."""
        query = context.request.url.query
        if "expand" in query and "createdAt" not in query:
            query += "&expand[]=createdAt"  # a client that picks its fields would otherwise drop the one we filter on
        headers = {"authorization": context.authorization} if context.authorization else {}
        response = await self.client.get(f"{self.settings.hf_upstream}/api/{kind}" + (f"?{query}" if query else ""),
                                         headers=headers)
        items = response.json() if response.status_code == 200 else None
        if not isinstance(items, list):
            return Response(response.content, response.status_code,
                            media_type=response.headers.get("content-type"))
        kept = [item for item in items if isinstance(item, dict) and self.settings.before(item.get("createdAt"))]
        self.log.write(context.run, "hf", "search", f"{kind}?{context.request.url.query}", listed=len(items),
                       kept=len(kept))
        headers = {"Link": context.rewrite("link", response.headers["link"])} if "link" in response.headers else {}
        return Response(json.dumps(kept), 200, headers=headers, media_type="application/json")

    async def _refs(self, context, kind, repo):
        """Branches and tags moved back to their cutoff commits; ones with nothing that old are left out."""
        main = await self.pin(context, kind, repo, "main")  # existence check, also covers a repository newer than the cutoff
        if self.settings.allowed_commit(context.run, kind, repo):
            return Response(json.dumps({"branches": [{"name": "main", "ref": "refs/heads/main", "targetCommit": main}],
                                        "tags": [], "converts": [], "pullRequests": []}), 200, media_type="application/json")
        response = await self.client.get(f"{self.settings.hf_upstream}/api/{kind}/{quote(repo)}/refs",
                                         headers=context.lookup_headers, follow_redirects=True)
        response.raise_for_status()
        refs, result = response.json(), {"branches": [], "tags": [], "converts": [], "pullRequests": []}
        for group in ("branches", "tags"):
            for ref in refs.get(group, []):
                try:
                    commit = await self.pin(context, kind, repo, ref["name"])
                except Refused:
                    continue
                result[group].append({**ref, "targetCommit": commit})
        return Response(json.dumps(result), 200, media_type="application/json")


class _Context:
    """One request: who asked, how upstream URLs map back to the proxy, and which credentials go upstream."""

    def __init__(self, service, request, run, mount, raw):
        self.service, self.request, self.run, self.mount, self.raw = service, request, run, mount.rstrip("/"), raw
        settings = service.settings
        self.authorization = request.headers.get("authorization") or (
            f"Bearer {settings.hf_token}" if settings.hf_token else None)
        # The proxy's own lookups (creation date, history) prefer its token: they are cached across runs.
        lookup = f"Bearer {settings.hf_token}" if settings.hf_token else self.authorization
        self.lookup_headers = {"authorization": lookup} if lookup else {}

    def rewrite(self, name, value):
        upstream = self.service.settings.hf_upstream
        if name == "link":
            return value.replace(f"<{upstream}/", f"<{self.mount}/")
        if value.startswith("/"):  # relative redirects: renamed repositories and the small-file cache
            return urlsplit(self.mount).path + value
        return self.mount + value[len(upstream):] if value.startswith(upstream + "/") else value

    async def send(self, path, *, pinned=None, entry=None):
        request, settings = self.request, self.service.settings
        if pinned:
            kind, repo, revision, commit = pinned
            self.service.log.write(self.run, "hf", "fetch" if entry else "info", f"{kind}/{repo}",
                                   requested=revision, commit=commit, path=entry, method=request.method)
        url = settings.hf_upstream + path + (f"?{request.url.query}" if request.url.query else "")
        body = await request.body() if request.method in ("POST", "PUT", "PATCH") else None
        return await forward(self.service.client, request.method, url, content=body, rewrite=self.rewrite,
                             headers=request_headers(request, authorization=self.authorization))
