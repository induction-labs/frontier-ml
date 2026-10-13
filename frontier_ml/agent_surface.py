"""Explicit agent payloads: the base image's client package (Spin setup and `frontier`) and task-local validation
code (validate.py and its imports).

This builder uses only the standard library, including in the image build stage.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

# The agent's `spin` command is the published Spin CLI (AGENT_DOCKERFILE installs build.SPIN_CLI_VERSION from
# https://spin.inductionlabs.com/cli/simple/; bump it with each release so docs and commands match the live service). This repo's `spin` package ships
# only the sandbox setup it owns: the Harbor health check (`python -m spin.harbor_setup`) and the pinned-data
# fetch it uses. The client also installs `frontier` (agent_cli.py, as the frontier_agent package): self-checks,
# serviced from outside the sandbox by the Harbor plugin. Never the operator harness (frontier_ml).
CLIENT_FILES = ("__init__.py", "transfer.py", "harbor_setup.py")
CLIENT_PROJECT = '''[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"
[project]
name = "frontier-spin-client"
version = "0.3.0"
requires-python = ">=3.12"
dependencies = ["httpx>=0.28,<1"]
[project.scripts]
frontier = "frontier_agent:main"
[tool.setuptools.packages.find]
include = ["spin*", "frontier_agent*"]
'''


def copy_file(source: Path, target: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Agent inputs must be regular files: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def stage_client(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "pyproject.toml").write_text(CLIENT_PROJECT)
    (destination / "agent-surface.json").write_text('{"profile":"spin-client-frontier-agent-v2"}\n')
    for name in CLIENT_FILES:
        copy_file(source / "spin" / name, destination / "spin" / name)
    copy_file(source / "frontier_ml" / "agent_cli.py", destination / "frontier_agent" / "__init__.py")


def validation_files(seed: Path, package: Path) -> dict[str, Path]:
    """validate.py, the task-local modules it imports (transitively, by bare name), the Apache-2.0 license
    text when a shipped module declares it, and the shared public core in frontier_ml.validation. Any other
    frontier_ml import is rejected, so trusted task modules (scoring, data building) never reach the agent
    unless validate.py imports them."""
    seed = Path(seed)
    script = seed / "validate.py"
    result = {"validate.py": script, "frontier_ml/__init__.py": package / "validation/__init__.py"}
    pending = [(script, None)]

    def add(module):
        if module != "frontier_ml.validation" and not module.startswith("frontier_ml.validation."):
            raise ValueError(f"Public validator imports private module: {module}")
        relative = module.removeprefix("frontier_ml.").replace(".", "/")
        source = package / (relative + ".py")
        if not source.is_file():
            source = package / relative / "__init__.py"
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"Missing public validation module: {module}")
        name = "frontier_ml/" + str(source.relative_to(package))
        if name not in result:
            result[name] = source
            pending.append((source, module if source.name == "__init__.py" else module.rpartition(".")[0]))
        if module != "frontier_ml.validation":
            add("frontier_ml.validation")

    def add_local(name):
        source = seed / f"{name}.py"
        if source.is_symlink():
            raise ValueError(f"Task module is a symlink: {source}")
        if source.name not in result:
            result[source.name] = source
            pending.append((source, None))

    while pending:
        source, parent = pending.pop()
        for node in ast.walk(ast.parse(source.read_text(), filename=str(source))):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    if not parent:
                        raise ValueError(f"{source.name} must use absolute imports (task modules are imported by bare name)")
                    module = importlib.util.resolve_name("." * node.level + module, parent)
                modules = [module]
                if module == "frontier_ml.validation":
                    modules.extend(module + "." + alias.name for alias in node.names)
            for module in modules:
                if module == "frontier_ml" or module.startswith("frontier_ml."):
                    add(module)
                elif parent is None and "." not in module and (seed / f"{module}.py").is_file():
                    add_local(module)
    # Task modules adapted from Apache-2.0 code say so in an SPDX header; the license text ships with them.
    if any("SPDX-License-Identifier: Apache-2.0" in source.read_text() for name, source in result.items() if "/" not in name):
        result["LICENSE-apache-2.0.txt"] = package / "licenses/Apache-2.0.txt"
    if "frontier_ml/validation/livecodebench_runner.py" in result:  # adapts MIT-licensed LiveCodeBench code
        result["frontier_ml/validation/LICENSE-livecodebench.txt"] = package / "scoring/vendor/livecodebench.LICENSE"
    return result


def stage_validation(seed: Path, destination: Path, package: Path, extra_texts: dict[str, str] | None = None) -> None:
    for relative, source in validation_files(seed, package).items():
        copy_file(source, destination / relative)
    for relative, text in (extra_texts or {}).items():  # e.g. CONTRACT.md, generated from the seed
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    files = {str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(destination.rglob("*")) if p.is_file()}
    (destination / "manifest.json").write_text(json.dumps({"version": 1, "files": files}, indent=2) + "\n")


def checked_validation_files(directory: Path) -> list[Path]:
    """Reject extra files, links and stale code in a generated validation payload."""
    manifest = directory / "manifest.json"
    if manifest.is_symlink():
        raise ValueError("Validation manifest cannot be a symlink")
    inventory = json.loads(manifest.read_text())
    files = inventory["files"]
    if inventory.get("version") != 1 or "validate.py" not in files:
        raise ValueError("Invalid validation payload manifest")
    actual = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symlink in validation payload")
        if path.is_file() and path != manifest:
            actual.add(str(path.relative_to(directory)))
    if actual != set(files):
        raise ValueError("Validation payload has missing or undeclared files")
    for name, digest in files.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Validation payload changed: {name}; re-render the task")
    return [directory / name for name in sorted(files)]


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "client":
        raise SystemExit("usage: agent_surface.py client SOURCE DESTINATION")
    stage_client(Path(sys.argv[2]), Path(sys.argv[3]))


def package_present(task: Path) -> bool:
    """Whether `task` is a complete rendered package (what the Harbor plugin accepts)."""
    from .taskspec import SEED_FILES
    task = Path(task)
    if not (task / "task.toml").is_file():
        return False
    if not (task / "environment/package/spin/__init__.py").is_file():
        return False
    if not all((task / "tests/lib" / package / "__init__.py").is_file() for package in ("spin", "frontier_ml")):
        return False
    if not all((task / "tests/task" / name).is_file() for name in SEED_FILES):
        return False
    if not all((task / "tests" / name).is_file() for name in ("Dockerfile", "grade.py", "selfcheck.py", "test.sh")):
        return False
    try:
        checked_validation_files(task / "environment/validation")
        setup = json.loads((task / "environment/setup.json").read_text())
        if "public_prepared" in setup:
            from spin.harbor_setup import prepared_files
            prepared_files(task / "environment/public", setup["public_prepared"])
    except (ValueError, OSError, KeyError, TypeError):
        return False
    return True
