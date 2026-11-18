"""GitHub as of the cutoff, served from local mirrors.

The first request for a repository fetches all of its branches and tags, then works out the view clients see: each
branch at its newest first-parent commit committed at or before the cutoff, each tag only if it was created by
then. Only what that view reaches is copied into the mirror that is served, so later history is not on disk at all
and cannot be fetched even by commit hash (hiding refs is not enough: git's protocol 2 hands out any object the
repository holds). The view never changes afterwards (the cutoff is fixed), so a repository is fetched once.

Served from the mirror: clone/fetch over smart HTTP, source archives and raw files. Release downloads and the
releases API are forwarded to GitHub after checking the release and the asset against the cutoff. Git LFS batch
requests are forwarded as they are (objects are addressed by content hash). Web pages, issues, pushes and the rest
of the API are refused.

Limits worth knowing: a commit's date is whatever its author's machine wrote, so a commit made before the cutoff
but pushed after it is visible; and a repository's creation date (GitHub API) is when it was created, not when it
became public.
"""
from __future__ import annotations

import asyncio
import gzip
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
from urllib.parse import quote, unquote

from fastapi.responses import Response, StreamingResponse

from .core import Refused, forward, request_headers

NAME = re.compile(r"[A-Za-z0-9_.-]{1,100}")
REF = re.compile(r"[A-Za-z0-9_.@+/-]{1,250}")
HEX = re.compile(r"[0-9a-f]{7,40}")
ARCHIVES = {".tar.gz": "tar.gz", ".tgz": "tar.gz", ".zip": "zip"}
UNSUPPORTED = ("only git clone/fetch, /archive/, /raw/ and /releases/download/ are served by the cutoff proxy "
               "(web pages, issues, pushes and the wider API are not)")


class GitHub:
    def __init__(self, settings, client, cache, log):
        self.settings, self.client, self.cache, self.log = settings, client, cache, log
        self.root = settings.state_dir / "github"
        self.mirroring: dict[str, asyncio.Task] = {}  # first-time fetches in progress, by owner/name

    async def handle(self, request, raw_path: str, run: str, mount: str, *, host: str = "github"):
        """`host` names which GitHub host's path layout this mount stands for: github (github.com),
        raw (raw.githubusercontent.com) or api (api.github.com)."""
        parts = [unquote(part) for part in raw_path.split("/") if part]
        try:
            if host == "api":
                return await self._api(request, parts, run, mount)
            if len(parts) < 2:
                raise Refused(403, UNSUPPORTED, reason="unsupported")
            owner, name = parts[0], parts[1].removesuffix(".git")
            rest = parts[2:]
            if host == "raw":
                return await self._raw(request, owner, name, rest, run)
            return await self._github(request, owner, name, rest, run)
        except Refused as exc:
            self.log.write(run, "github", "denied", "/".join(parts), reason=exc.reason, status=exc.status, host=host)
            raise

    async def _github(self, request, owner, name, rest, run):
        method, query = request.method, request.query_params
        if rest == ["info", "refs"] and method == "GET":
            if query.get("service") != "git-upload-pack":
                raise Refused(403, "the cutoff proxy is read-only and speaks smart HTTP only", reason="unsupported")
            repo = await self.repository(owner, name)
            version2 = "version=2" in request.headers.get("git-protocol", "")
            output = await self._git(repo, "upload-pack", "--stateless-rpc", "--advertise-refs", str(repo),
                                     protocol=request.headers.get("git-protocol"))
            # Protocol 2 answers with its capability list directly; older clients expect the service banner first.
            banner = b"" if version2 else _packet(b"# service=git-upload-pack\n") + b"0000"
            self.log.write(run, "github", "git", f"{owner}/{name}", head=await self._head(repo))
            return Response(banner + output, media_type="application/x-git-upload-pack-advertisement",
                            headers={"Cache-Control": "no-cache"})
        if rest == ["git-upload-pack"] and method == "POST":
            repo = await self.repository(owner, name)
            body = await request.body()
            if request.headers.get("content-encoding") == "gzip":
                body = gzip.decompress(body)
            return await self._stream(repo, ["upload-pack", "--stateless-rpc", str(repo)], stdin=body,
                                      media_type="application/x-git-upload-pack-result",
                                      protocol=request.headers.get("git-protocol"))
        if rest == ["info", "lfs", "objects", "batch"] and method == "POST":
            await self.repository(owner, name)
            body = await request.body()
            if json.loads(body or b"{}").get("operation") != "download":
                raise Refused(403, "the cutoff proxy is read-only", reason="unsupported")
            self.log.write(run, "github", "lfs", f"{owner}/{name}")
            return await forward(self.client, "POST", f"{self.settings.github_upstream}/{owner}/{name}.git/info/lfs/objects/batch",
                                 content=body, headers=request_headers(request, authorization=self._basic()))
        if method not in ("GET", "HEAD"):
            raise Refused(403, UNSUPPORTED, reason="unsupported")
        if rest[:1] == ["archive"] and len(rest) >= 2:
            last = rest[-1]
            suffix = next((s for s in ARCHIVES if last.endswith(s)), None)
            if suffix is None:
                raise Refused(404, "archives are .tar.gz or .zip", reason="unsupported")
            ref = "/".join([*rest[1:-1], last[:-len(suffix)]])
            return await self._archive(owner, name, ref, ARCHIVES[suffix], run)
        if rest[:1] in (["tarball"], ["zipball"]) and len(rest) >= 2:
            return await self._archive(owner, name, "/".join(rest[1:]), "tar.gz" if rest[0] == "tarball" else "zip", run)
        if rest[:1] == ["raw"] and len(rest) >= 3:
            return await self._raw(request, owner, name, rest[1:], run)
        if rest[:2] == ["releases", "download"] and len(rest) == 4:
            return await self._asset(request, owner, name, rest[2], rest[3], run)
        if rest[:3] == ["releases", "latest", "download"] and len(rest) == 4:
            return await self._asset(request, owner, name, None, rest[3], run)
        raise Refused(403, UNSUPPORTED, reason="unsupported")

    # --- mirrors -----------------------------------------------------------------------------------------------

    async def repository(self, owner, name) -> Path:
        """The mirror of owner/name with its cutoff view built; raises Refused when it did not exist by the cutoff."""
        if not NAME.fullmatch(owner) or not NAME.fullmatch(name) or name in (".", "..") or owner in (".", ".."):
            raise Refused(404, "Repository not found.", reason="repo_missing")
        slug = f"{owner}/{name}".lower()
        key = f"gh:repo:{self.settings.cutoff_text}:{slug}"
        path = self.root / slug.split("/")[0] / (slug.split("/")[1] + ".git")
        state = self.cache.get(key)
        if state is None:
            # One fetch per repository however many requests ask for it, and it outlives a request that gives up.
            task = self.mirroring.get(slug)
            if task is None:
                task = self.mirroring[slug] = asyncio.ensure_future(self._prepare(key, slug, owner, name, path))
                task.add_done_callback(lambda done: done.cancelled() or done.exception())  # retrieved even if nobody waits
            try:
                state = await asyncio.wait_for(asyncio.shield(task), self.settings.mirror_wait_seconds)
            except asyncio.TimeoutError:
                raise Refused(503, f"{owner}/{name} is being mirrored for the first time; retry in a minute",
                              reason="mirror_in_progress") from None
        if state != "ready":
            raise Refused(404, f"Repository not found: {owner}/{name} did not exist on {self.settings.cutoff_text} "
                               "(this run sees GitHub as of that date).", reason=state)
        return path

    async def _prepare(self, key, slug, owner, name, path) -> str:
        try:
            state = await self._mirror(owner, name, path)
            if state == "ready" or self.settings.settled:
                self.cache.set(key, state)
            return state
        finally:
            self.mirroring.pop(slug, None)

    async def _mirror(self, owner, name, path) -> str:
        metadata = await self._metadata(owner, name)
        if metadata is not None:
            if not self.settings.before(metadata.get("created_at")):
                return "repo_after_cutoff"
            if metadata.get("size", 0) > self.settings.github_max_kb:
                raise Refused(403, f"{owner}/{name} is too large to mirror ({metadata['size'] // 1024} MB)",
                              reason="repo_too_large")
        url = f"{self.settings.github_upstream}/{owner}/{name}.git"
        work = path.with_name(path.name + ".tmp")
        shutil.rmtree(work, ignore_errors=True)
        work.parent.mkdir(parents=True, exist_ok=True)
        try:
            await self._git(None, "init", "--quiet", "--bare", str(work))
            # --no-tags: fetch would otherwise also write every tag it follows straight into refs/tags/.
            await self._git(work, *self._auth(), "fetch", "--quiet", "--no-tags", url,
                            "+refs/heads/*:refs/upstream/heads/*", "+refs/tags/*:refs/upstream/tags/*",
                            missing="repo_missing")
            symref = (await self._git(work, *self._auth(), "ls-remote", "--symref", url, "HEAD")).decode()
            default = next((line.split()[1].removeprefix("refs/heads/") for line in symref.splitlines()
                            if line.startswith("ref: refs/heads/")), None)
            if not await self._view(work, default):
                return "repo_after_cutoff"
            # Copy the view alone into the repository that is served; the full fetch is then thrown away.
            final = path.with_name(path.name + ".new")
            shutil.rmtree(final, ignore_errors=True)
            await self._git(None, "init", "--quiet", "--bare", str(final))
            await self._git(final, "fetch", "--quiet", "--no-tags", work.as_uri(),
                            "+refs/heads/*:refs/heads/*", "+refs/tags/*:refs/tags/*")
            await self._git(final, "symbolic-ref", "HEAD", f"refs/heads/{default}")
            await self._git(final, "config", "uploadpack.allowFilter", "true")  # partial clones (--filter=blob:none)
            shutil.rmtree(path, ignore_errors=True)
            final.rename(path)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        return "ready"

    async def _view(self, repo, default) -> bool:
        """Write refs/heads/ and refs/tags/ as of the cutoff into the full fetch; False when the default branch has
        nothing that old."""
        cutoff = int(self.settings.cutoff.timestamp())
        listing = (await self._git(repo, "for-each-ref", "--format=%(refname) %(objectname) %(creatordate:unix)",
                                   "refs/upstream/")).decode().split("\n")
        updates = []
        for line in filter(None, listing):
            ref, target, created = line.split(" ")
            if ref.startswith("refs/upstream/tags/"):
                if created and int(created) <= cutoff:
                    updates.append(f"create refs/tags/{ref.removeprefix('refs/upstream/tags/')} {target}")
                continue
            # First parent only: a merge made after the cutoff must not pull the branch tip onto a side branch
            # commit that happens to carry an older date. `@<seconds>` is git's unambiguous timestamp form.
            commit = (await self._git(repo, "rev-list", "-1", "--first-parent", f"--before=@{cutoff}", ref)).decode().strip()
            if commit:
                updates.append(f"create refs/heads/{ref.removeprefix('refs/upstream/heads/')} {commit}")
        if not default or not any(line.startswith(f"create refs/heads/{default} ") for line in updates):
            return False
        await self._git(repo, "update-ref", "--stdin", stdin=("\n".join(updates) + "\n").encode())
        return True

    async def _metadata(self, owner, name):
        """GitHub's repository record (creation date, size), or None when the API is turned off."""
        if not self.settings.github_api:
            return None
        response = await self.client.get(f"{self.settings.github_api}/repos/{owner}/{name}", headers=self._api_headers(),
                                         follow_redirects=True)
        if response.status_code == 404:
            raise Refused(404, f"Repository not found: {owner}/{name}.", reason="repo_missing")
        if response.status_code != 200:
            # Usually the anonymous rate limit (60 an hour; set a GitHub token on the proxy). Commit dates still
            # bound what is served, so carry on without the creation date rather than fail every clone; the log
            # shows which repositories went unchecked.
            self.log.write("-", "github", "created_unchecked", f"{owner}/{name}", status=response.status_code)
            return None
        return response.json()

    # --- files from the mirror ---------------------------------------------------------------------------------

    async def _commit(self, repo, ref) -> str | None:
        """A branch, tag or commit hash as a commit; the mirror holds nothing but the cutoff view."""
        if not REF.fullmatch(ref) or ref.startswith(("-", "/")) or ".." in ref:
            return None
        names = [ref] if ref.startswith("refs/") else [f"refs/heads/{ref}", f"refs/tags/{ref}"]
        if ref == "HEAD":
            names = ["HEAD"]
        for candidate in [*names, *([ref] if HEX.fullmatch(ref) else [])]:
            commit = (await self._git(repo, "rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}", check=False)).decode().strip()
            if commit:
                return commit
        return None

    async def _archive(self, owner, name, ref, kind, run):
        repo = await self.repository(owner, name)
        commit = await self._commit(repo, ref)
        if commit is None:
            raise Refused(404, f"{ref} is not a branch, tag or commit of {owner}/{name} as of "
                               f"{self.settings.cutoff_text}", reason="revision_after_cutoff")
        label = ref.removeprefix("refs/heads/").removeprefix("refs/tags/").replace("/", "-")
        self.log.write(run, "github", "archive", f"{owner}/{name}", requested=ref, commit=commit)
        return await self._stream(repo, ["archive", f"--format={kind}", f"--prefix={name}-{label}/", commit],
                                  media_type="application/zip" if kind == "zip" else "application/gzip",
                                  headers={"Content-Disposition": f'attachment; filename="{name}-{label}.{kind}"'})

    async def _raw(self, request, owner, name, rest, run):
        """<ref>/<path>, where a branch name may itself contain slashes: the longest-known ref wins."""
        if request.method not in ("GET", "HEAD"):
            raise Refused(403, UNSUPPORTED, reason="unsupported")
        try:
            repo = await self.repository(owner, name)
        except Refused as exc:
            if exc.reason != "repo_too_large" or not self.settings.github_api:
                raise
            return await self._raw_unmirrored(request, owner, name, rest, run)
        for split in range(1, len(rest)):
            ref, path = "/".join(rest[:split]), "/".join(rest[split:])
            commit = await self._commit(repo, ref)
            if commit is None:
                continue
            kind = (await self._git(repo, "cat-file", "-t", f"{commit}:{path}", check=False)).decode().strip()
            if kind != "blob":
                raise Refused(404, f"{path} is not a file in {owner}/{name} at {ref}", reason="entry_missing")
            self.log.write(run, "github", "raw", f"{owner}/{name}", requested=ref, commit=commit, path=path)
            guess = mimetypes.guess_type(path)[0] or ""
            media = "text/plain; charset=utf-8" if guess.startswith("text/") or not guess else "application/octet-stream"
            return await self._stream(repo, ["cat-file", "blob", f"{commit}:{path}"], media_type=media)
        raise Refused(404, f"no branch, tag or commit of {owner}/{name} as of {self.settings.cutoff_text} matches "
                           f"{'/'.join(rest)}", reason="revision_after_cutoff")

    async def _raw_unmirrored(self, request, owner, name, rest, run):
        """One file from a repository too large to mirror: GitHub's API names the branch's newest commit up to the
        cutoff, and the file is fetched at that commit. Coarser than the mirror (the API follows every parent, not
        the first), which is why it is only the fallback."""
        api = f"{self.settings.github_api}/repos/{owner}/{name}/commits"
        for split in range(1, len(rest)):
            ref, path = "/".join(rest[:split]), "/".join(quote(part) for part in rest[split:])
            if not REF.fullmatch(ref) or ".." in ref:
                continue
            if HEX.fullmatch(ref) and len(ref) == 40:
                response = await self.client.get(f"{api}/{ref}", headers=self._api_headers())
                found = response.status_code == 200 and self.settings.before(response.json()["commit"]["committer"]["date"])
                commit = ref if found else None
            else:
                response = await self.client.get(api, headers=self._api_headers(), params={
                    "sha": ref, "until": self.settings.cutoff_text, "per_page": 1})
                commit = response.json()[0]["sha"] if response.status_code == 200 and response.json() else None
            if commit is None:
                continue
            self.log.write(run, "github", "raw", f"{owner}/{name}", requested=ref, commit=commit, path=path, via="api")
            return await forward(self.client, request.method, f"{self.settings.github_raw}/{owner}/{name}/{commit}/{path}",
                                 headers=request_headers(request, authorization=None))
        raise Refused(404, f"no branch, tag or commit of {owner}/{name} as of {self.settings.cutoff_text} matches "
                           f"{'/'.join(rest)}", reason="revision_after_cutoff")

    # --- releases ----------------------------------------------------------------------------------------------

    async def _releases(self, owner, name, *, tag=None, pages=3):
        """Published releases at or before the cutoff, newest first (one release when `tag` is given)."""
        if not self.settings.github_api:
            raise Refused(404, "releases are not available through this proxy", reason="unsupported")
        # Release files live outside the repository, so only its creation date is needed, not a mirror.
        if not NAME.fullmatch(owner) or not NAME.fullmatch(name.removesuffix(".git")):
            raise Refused(404, "Repository not found.", reason="repo_missing")
        key = f"gh:created:{owner}/{name}".lower()
        created = self.cache.get(key)
        if created is None:
            metadata = await self._metadata(owner, name)
            if metadata is None:
                await self.repository(owner, name)  # API unavailable: fall back to what the commit dates say
            else:
                created = metadata.get("created_at")
                self.cache.set(key, created)
        if created is not None and not self.settings.before(created):
            raise Refused(404, f"Repository not found: {owner}/{name} did not exist on {self.settings.cutoff_text}.",
                          reason="repo_after_cutoff")
        base = f"{self.settings.github_api}/repos/{owner}/{name}/releases"
        result = []
        for page in range(1, pages + 1):
            url = f"{base}/tags/{tag}" if tag else f"{base}?per_page=100&page={page}"
            response = await self.client.get(url, headers=self._api_headers(), follow_redirects=True)
            if response.status_code == 404:
                break
            if response.status_code != 200:
                raise Refused(503, f"GitHub API unavailable: HTTP {response.status_code}", reason="github_api_unavailable")
            batch = [response.json()] if tag else response.json()
            for release in batch:
                if release.get("draft") or not self.settings.before(release.get("published_at")):
                    continue
                # A rolling release keeps its date while its files are replaced; date every asset as well.
                release["assets"] = [asset for asset in release.get("assets", [])
                                     if self.settings.before(asset.get("updated_at"))]
                result.append(release)
            if tag or len(batch) < 100:
                break
        return result

    async def _asset(self, request, owner, name, tag, filename, run):
        if tag is not None and (not REF.fullmatch(tag) or ".." in tag):
            raise Refused(404, "Release not found.", reason="release_missing")
        releases = await self._releases(owner, name, tag=tag)
        if tag is None:  # "latest" is the newest full release, as on GitHub
            releases = [release for release in releases if not release.get("prerelease")][:1]
        asset = next((asset for release in releases for asset in release["assets"] if asset["name"] == filename), None)
        if asset is None:
            raise Refused(404, f"no release asset {filename} in {owner}/{name} ({tag or 'latest'}) as of "
                               f"{self.settings.cutoff_text}", reason="release_after_cutoff")
        self.log.write(run, "github", "release", f"{owner}/{name}", requested=tag or "latest",
                       tag=releases[0]["tag_name"] if tag is None else tag, path=filename)
        # GitHub answers with a redirect to signed storage; fetch it here so the client needs no other host.
        return await forward(self.client, request.method, asset["browser_download_url"], follow=True,
                             headers=request_headers(request, authorization=None))

    async def _api(self, request, parts, run, mount):
        """api.github.com, releases only: /repos/<owner>/<repo>/releases[/latest|/tags/<tag>]."""
        if request.method != "GET" or len(parts) < 4 or parts[0] != "repos" or parts[3] != "releases":
            raise Refused(403, "only /repos/<owner>/<repo>/releases is served by the cutoff proxy", reason="unsupported")
        owner, name, rest = parts[1], parts[2], parts[4:]
        if rest == []:
            body = await self._releases(owner, name)
        elif rest == ["latest"]:
            body = next((release for release in await self._releases(owner, name) if not release.get("prerelease")), None)
        elif len(rest) == 2 and rest[0] == "tags":
            body = next(iter(await self._releases(owner, name, tag=rest[1])), None)
        else:
            raise Refused(403, "only /repos/<owner>/<repo>/releases is served by the cutoff proxy", reason="unsupported")
        if body is None:
            raise Refused(404, "Not Found", reason="release_after_cutoff")
        self.log.write(run, "github", "releases", f"{owner}/{name}", requested="/".join(rest) or "list")
        # Download links in the answer lead back through the proxy's github.com mount.
        site = mount.rstrip("/").removesuffix("-api")
        text = json.dumps(body).replace(self.settings.github_upstream + "/", site + "/").replace(
            self.settings.github_api + "/", site + "-api/")
        return Response(text, media_type="application/json")

    # --- git ---------------------------------------------------------------------------------------------------

    def _basic(self):
        if not self.settings.github_token:
            return None
        import base64
        return "Basic " + base64.b64encode(f"x-access-token:{self.settings.github_token}".encode()).decode()

    def _auth(self):
        return ["-c", f"http.extraHeader=Authorization: {self._basic()}"] if self.settings.github_token else []

    def _api_headers(self):
        headers = {"accept": "application/vnd.github+json", "user-agent": "cutoff-proxy"}
        if self.settings.github_token:
            headers["authorization"] = f"Bearer {self.settings.github_token}"
        return headers

    def _command(self, repo, args, protocol):
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(self.root), "GIT_TERMINAL_PROMPT": "0",
               "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "LC_ALL": "C"}
        if protocol and re.fullmatch(r"[a-z0-9=:.-]{1,64}", protocol):
            env["GIT_PROTOCOL"] = protocol
        self.root.mkdir(parents=True, exist_ok=True)
        return ["git", *(["--git-dir", str(repo)] if repo else []), *args], env

    async def _git(self, repo, *args, stdin=None, check=True, protocol=None, missing=None) -> bytes:
        command, env = self._command(repo, args, protocol)
        process = await asyncio.create_subprocess_exec(
            *command, env=env, stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        output, error = await process.communicate(stdin)
        if process.returncode and check:
            if missing:  # the upstream fetch: no such repository, or one that needs credentials we do not have
                raise Refused(404, "Repository not found.", reason=missing)
            raise RuntimeError(f"git {args[0]} failed: {error.decode(errors='replace')[-400:]}")
        return output

    async def _head(self, repo):
        return (await self._git(repo, "rev-parse", "--verify", "--quiet", "HEAD", check=False)).decode().strip() or None

    async def _stream(self, repo, args, *, media_type, stdin=None, protocol=None, headers=None):
        command, env = self._command(repo, args, protocol)
        process = await asyncio.create_subprocess_exec(
            *command, env=env, stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)

        async def feed():
            process.stdin.write(stdin)
            try:
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            process.stdin.close()

        feeder = asyncio.ensure_future(feed()) if stdin is not None else None

        async def body():
            try:
                while chunk := await process.stdout.read(1 << 16):
                    yield chunk
            finally:
                if feeder:
                    feeder.cancel()
                if process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                await process.wait()

        return StreamingResponse(body(), media_type=media_type, headers={"Cache-Control": "no-cache", **(headers or {})})


def _packet(data: bytes) -> bytes:
    return f"{len(data) + 4:04x}".encode() + data
