"""Task seeds: tasks/<slug>/ holds task.yaml (metadata), validate.py (public request template and metric,
shipped to the agent as an executable script) and verifier.py (trusted final evaluation, never shipped).

A seed is the only source for a task. `frontier build` renders seeds into Harbor packages
under tasks_rendered/, copying the whole seed into tests/task/ for the verifier and only
validate.py, the sibling modules it imports and the shared validation core into /task/.
Data and starting weights are pinned Hugging Face folders named in task.yaml.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import yaml

ROOT = Path(__file__).resolve().parents[1]
SEEDS = Path(os.environ.get("FRONTIER_TASKS_DIR") or ROOT / "tasks")
PUBLIC_SCRIPT = "validate.py"
VERIFIER = "verifier.py"
SEED_FILES = ("task.yaml", PUBLIC_SCRIPT, VERIFIER)
REQUIRED = ("title", "description", "interface", "evaluation", "metric", "concurrency", "timeout", "data", "verifier")


@dataclass(frozen=True)
class Task:
    slug: str
    path: Path
    spec: dict

    @property
    def module(self) -> ModuleType:
        """The trusted verifier: verify(server, data_path, config), optional preflight() and default_config()."""
        return load_module(self.slug, self.path / VERIFIER, require=("verify",))

    @property
    def public(self) -> ModuleType | None:
        """The public script: run(transport, data_path, config) and main(argv); None when the task has none."""
        path = self.path / PUBLIC_SCRIPT
        return load_module(self.slug + ".public", path, require=("run", "main")) if path.is_file() else None


def read_spec(path: Path) -> dict:
    spec = yaml.safe_load((Path(path) / "task.yaml").read_text())
    if not isinstance(spec, dict):
        raise ValueError(f"{path}/task.yaml must be a mapping")
    missing = [key for key in REQUIRED if key not in spec]
    if missing:
        raise ValueError(f"{path}/task.yaml is missing {', '.join(missing)}")
    if script := data_preparation(spec):
        source = Path(path) / script
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"data.prepare must name a regular Python file in the task seed: {source}")
    starting_artifacts(spec)
    if not isinstance(spec.get("requires_env", []), list) or not all(isinstance(n, str) and n for n in spec.get("requires_env", [])):
        raise ValueError(f"{path}/task.yaml requires_env must list environment variable names")
    if spec.get("review") and not (Path(path) / "review.py").is_file():
        raise ValueError(f"{path} requires a trusted review.py policy")
    if "score_normalization" in spec:
        from .scoring.normalization import check_spec
        check_spec(spec["score_normalization"])
    return spec


UNPUBLISHED = "unpublished"


def _data(spec: dict, *, published: bool = False) -> dict:
    """The data block; `revision: unpublished` is allowed until `frontier upload` pins the first commit."""
    data = spec["data"]
    if isinstance(data, dict) and "prepare" in data:
        if not isinstance(data["prepare"], str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.py", data["prepare"]):
            raise ValueError("data.prepare must name a Python file in the task seed")
        if any(key in data for key in ("repo", "revision", "path")):
            raise ValueError("data.prepare cannot be combined with mirrored repo/revision/path")
        sources = data.get("sources", {})
        if not isinstance(sources, dict):
            raise ValueError("data.sources must map names to pinned Hugging Face sources")
        from spin.transfer import hf_source
        for source in sources.values():
            if not isinstance(source, dict):
                raise ValueError("data.sources must map names to pinned Hugging Face sources")
            hf_source({"repo_type": "dataset", **source})
        return data
    if not isinstance(data, dict) or not all(isinstance(data.get(key), str) and data[key] for key in ("repo", "revision", "path")):
        raise ValueError("data needs repo, revision (a commit sha, or `unpublished`) and path (the task's folder in that dataset)")
    safe_relative(data["path"])
    if data["revision"] == UNPUBLISHED:
        if published:
            raise ValueError(f"data.revision is {UNPUBLISHED}; run `frontier upload <slug>` to publish data/<slug>/ and pin it")
    else:
        from spin.transfer import hf_source
        hf_source({"repo": data["repo"], "revision": data["revision"], "repo_type": "dataset"})
    return data


def data_preparation(spec: dict) -> str | None:
    """Trusted seed importer, run on the operator for public data and in the verifier for private data."""
    return _data(spec).get("prepare")


def starting_artifacts(spec: dict) -> list[dict]:
    """starting_artifacts entries: upstream Hugging Face repositories at pinned commits, placed at /artifacts/<to>."""
    from spin.transfer import hf_source
    entries = spec.get("starting_artifacts") or []
    if not isinstance(entries, list):
        raise ValueError("starting_artifacts must be a list")
    sources = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - {"repo", "revision", "repo_type", "to", "exclude"}:
            raise ValueError("starting_artifacts entries take repo, revision, to, and optional repo_type and exclude")
        safe_relative(entry.get("to", ""))
        sources.append({**hf_source(entry), "to": entry["to"]})
    return sources


def data_sources(spec: dict) -> dict[str, dict]:
    """The task's pinned dataset folders: public/ (the agent's /task/public) and private/ (the verifier's /tests/private)."""
    from spin.transfer import hf_source
    data = _data(spec, published=True)
    if "prepare" in data:
        raise ValueError("This task prepares upstream data; it has no mirrored public/private folders")
    return {folder: hf_source({"repo": data["repo"], "revision": data["revision"], "repo_type": "dataset",
                               "prefix": f"{data['path']}/{folder}"}) for folder in ("public", "private")}


def data_folder(spec: dict) -> dict:
    """The task's whole pinned dataset folder (public/, private/, artifacts/, manifest.json), for local mirrors."""
    from spin.transfer import hf_source
    data = _data(spec, published=True)
    if "prepare" in data:
        raise ValueError("This task prepares upstream data; it must not be fetched or published as a mirrored folder")
    return hf_source({"repo": data["repo"], "revision": data["revision"], "repo_type": "dataset", "prefix": data["path"]})


def artifact_sources(spec: dict) -> list[dict]:
    """Downloads into the agent's /workspace/artifacts: the starting_artifacts entries, then the task folder's
    optional artifacts/ directory."""
    from spin.transfer import hf_source
    data = _data(spec, published=True)
    if "prepare" in data:
        return starting_artifacts(spec)
    return [*starting_artifacts(spec), {**hf_source({"repo": data["repo"], "revision": data["revision"], "repo_type": "dataset",
                                                     "prefix": f"{data['path']}/artifacts"}), "to": "", "optional": True}]


def safe_relative(value: str, *, prefix: str | None = None) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Expected a relative POSIX path")
    path = Path(value)
    if path.is_absolute() or any(part in (".", "..") for part in value.split("/")) or any(c in value for c in "*?[]"):
        raise ValueError(f"Unsafe relative path: {value}")
    if prefix and (len(path.parts) < 2 or path.parts[0] != prefix):
        raise ValueError(f"Path must be under {prefix}/: {value}")
    return path


def base_mode(policy: dict) -> str:
    """How a review policy constrains the starting checkpoint named in architecture.json.

    "pinned": base_model/base_revision are exact strings (tasks that supply an artifact).
    "allowed": allowed_bases lists acceptable {"model", "revision"} pairs.
    "declared": base_model is None; the candidate names any nonempty model and revision.
    """
    model, revision, allowed = policy.get("base_model"), policy.get("base_revision"), policy.get("allowed_bases")
    if allowed is not None:
        if model is not None or revision is not None:
            raise ValueError("Review policy allowed_bases excludes base_model/base_revision")
        if not isinstance(allowed, list) or not allowed or any(
                not isinstance(entry, dict) or set(entry) != {"model", "revision"}
                or not all(isinstance(entry[key], str) and entry[key] for key in entry) for entry in allowed):
            raise ValueError("Review policy allowed_bases must list nonempty {model, revision} pairs")
        return "allowed"
    if model is None and revision is None and "base_model" in policy:
        return "declared"
    if isinstance(model, str) and model and isinstance(revision, str) and revision:
        return "pinned"
    raise ValueError("Review policy needs pinned base_model/base_revision, allowed_bases, or base_model: None")


def review_policy(slug: str, hint: str | Path | None = None) -> dict:
    task = load_task(slug, hint)
    if not task.spec.get("review"):
        raise ValueError(f"No review policy for {slug}")
    policy = getattr(load_module(slug + ".review", task.path / "review.py"), "POLICY", None)
    if not isinstance(policy, dict):
        raise ValueError(f"{task.path}/review.py needs a POLICY mapping")
    base_mode(policy)
    if type(policy.get("require_weights", True)) is not bool:
        raise ValueError("Review policy require_weights must be a boolean")
    if type(policy.get("served_by_verifier", False)) is not bool:
        raise ValueError("Review policy served_by_verifier must be a boolean")
    checks = policy.get("checks")
    if not isinstance(checks, dict) or not checks or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in checks.items()):
        raise ValueError("Review policy checks must be a nonempty mapping of names to instructions")
    from .review_rules import with_session_checks
    with_session_checks(policy)  # refuses a review.py that defines a shared session check; the policy itself is unchanged
    return policy


def served_by_verifier(task) -> bool:
    """True when the task serves submitted weights with its own trusted server (review policy served_by_verifier),
    so a submission has no serve.sh."""
    return bool(task.spec.get("review")) and review_policy(task.slug, task.path).get("served_by_verifier") is True


def seed_dirs(root: Path = SEEDS) -> dict[str, Path]:
    root = Path(root)
    if not root.is_dir():
        return {}
    return {entry.name: entry for entry in sorted(root.iterdir()) if (entry / "task.yaml").is_file()}


def load_specs(root: Path = SEEDS) -> dict[str, dict]:
    return {slug: read_spec(path) for slug, path in seed_dirs(root).items()}


_modules: dict[tuple[str, str], ModuleType] = {}


def load_module(name: str, path: Path, *, require: tuple[str, ...] = ()) -> ModuleType:
    path = Path(path)
    key = (name, str(path.resolve()))
    if key not in _modules:
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing")
        add_task_path(path.parent)
        spec = importlib.util.spec_from_file_location("frontier_seed_" + name.replace("-", "_").replace(".", "_"), path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        missing = [attr for attr in require if not callable(getattr(module, attr, None))]
        if missing:
            raise ValueError(f"{path} must define {', '.join(missing)}")
        _modules[key] = module
    return _modules[key]


def add_task_path(directory: Path) -> None:
    """Task-local modules are imported by bare name (`from spans import score_rows`), exactly as
    /task/validate.py imports its shipped siblings; module names are unique across tasks."""
    entry = str(Path(directory).resolve())
    if entry not in sys.path:
        sys.path.insert(0, entry)


def task_module(slug: str, name: str, root: Path = SEEDS) -> ModuleType:
    """Import tasks/<slug>/<name>.py under its bare name, e.g. task_module("pii-redaction", "spans")."""
    add_task_path(Path(root) / slug)
    return importlib.import_module(name)


def local_modules(root: Path = SEEDS) -> dict[str, Path]:
    """Every task-local module by bare name; a name used twice, or shadowing the standard library, is an error."""
    found: dict[str, Path] = {}
    for seed in seed_dirs(root).values():
        for path in sorted(seed.glob("*.py")):
            if path.name in (PUBLIC_SCRIPT, VERIFIER, "review.py"):
                continue
            if path.stem in sys.stdlib_module_names or path.stem in ("frontier_ml", "spin"):
                raise ValueError(f"{path} shadows the module {path.stem}; rename it")
            if path.stem in found:
                raise ValueError(f"Task module name {path.stem} is used by both {found[path.stem]} and {path}")
            found[path.stem] = path
    return found


def sibling(file: str | Path, name: str, *, require: tuple[str, ...] = ()) -> ModuleType:
    """Load another file from the same seed directory; verifier.py uses this to reuse validate.py."""
    path = Path(file).resolve().with_name(name)
    return load_module(f"{path.parent.name}.{path.stem}", path, require=require)


def find_task_dir(slug: str, hint: str | Path | None = None) -> Path:
    """Explicit --task-dir, then $FRONTIER_TASK_DIR, then the repository seeds."""
    if hint:
        if not (Path(hint) / "task.yaml").is_file():
            raise FileNotFoundError(f"{hint} has no task.yaml")
        return Path(hint)
    candidates = [Path(os.environ["FRONTIER_TASK_DIR"])] if os.environ.get("FRONTIER_TASK_DIR") else []
    candidates.append(SEEDS / slug)
    for candidate in candidates:
        if (candidate / "task.yaml").is_file():
            return candidate
    raise FileNotFoundError(f"No task seed found for {slug}; pass --task-dir")


def load_task(slug: str, hint: str | Path | None = None) -> Task:
    path = find_task_dir(slug, hint)
    return Task(slug, path, read_spec(path))
