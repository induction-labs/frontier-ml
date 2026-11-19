"""Build the Harbor Hub dataset from tasks/.

    uv run python hub/build.py                              # writes hub/dist/frontier-ml/
    uv run harbor publish hub/dist/frontier-ml -t v1.0.0    # private unless --public is passed

The Hub needs every task to name its package in a `[task]` section of task.toml. With that section Harbor reports
the task as `<org>/<task>` instead of its folder name, in result.json and to `frontier-regrade`, so tasks/ stays
without it and only the copies built here carry it. Everything else in a built task is the file from tasks/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tomllib

from harbor.models.dataset.manifest import DatasetManifest
from harbor.models.task.config import TaskConfig
from harbor.publisher.packager import Packager

ROOT = Path(__file__).resolve().parents[1]
ORG = "induction-labs"
DATASET = "frontier-ml"
AUTHOR = "Induction Labs"
DESCRIPTION = ("FrontierML: budgeted, end-to-end machine learning tasks for coding agents. Each agent spends a fixed "
               "budget on GPUs and model APIs and submits an inference server graded offline on held-out data.")
KEYWORDS = ["machine-learning", "ml-engineering", "long-horizon", "budgeted"]


def package_section(name: str, version: str, description: str, keywords: list[str]) -> str:
    return (f"[task]\nname = {json.dumps(name)}\nversion = {json.dumps(version)}\n"
            f"description = {json.dumps(description, ensure_ascii=False)}\n"
            f"authors = [{{ name = {json.dumps(AUTHOR)} }}]\nkeywords = {json.dumps(keywords)}\n\n")


def build_task(source: Path, target: Path, org: str, version: str) -> str:
    """Copy one task and add its `[task]` section; returns the digest the Hub will record for it."""
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    text = (source / "task.toml").read_text()
    config = tomllib.loads(text)
    if "task" in config or text.count("\n[metadata]\n") != 1:
        raise ValueError(f"{source.name}: unexpected task.toml; update build_task")
    title = (source / "environment/validation/CONTRACT.md").read_text().splitlines()[0].removeprefix("# ").strip()
    section = package_section(f"{org}/{source.name}", version, title, ["machine-learning", *config["metadata"]["tags"]])
    built = text.replace("\n[metadata]\n", "\n" + section + "[metadata]\n")
    without = tomllib.loads(built)
    package = without.pop("task")
    if without != config or TaskConfig.model_validate_toml(built).task.name != package["name"]:
        raise ValueError(f"{source.name}: the built task.toml differs from the source beyond its [task] section")
    (target / "task.toml").write_text(built)
    return "sha256:" + Packager.compute_content_hash(target)[0]


def build(out: Path, org: str = ORG) -> Path:
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    dataset = out / DATASET
    if dataset.exists():
        shutil.rmtree(dataset)
    dataset.mkdir(parents=True)
    tasks = sorted(path for path in (ROOT / "tasks").iterdir() if (path / "task.toml").is_file())
    manifest = (f"[dataset]\nname = {json.dumps(f'{org}/{DATASET}')}\nversion = {json.dumps(version)}\n"
                f"description = {json.dumps(DESCRIPTION)}\nauthors = [{{ name = {json.dumps(AUTHOR)} }}]\n"
                f"keywords = {json.dumps(KEYWORDS)}\n")
    for task in tasks:
        digest = build_task(task, dataset / task.name, org, version)
        manifest += f"\n[[tasks]]\nname = {json.dumps(f'{org}/{task.name}')}\ndigest = {json.dumps(digest)}\n"
    (dataset / "dataset.toml").write_text(manifest)
    DatasetManifest.from_toml_file(dataset / "dataset.toml")
    readme = (ROOT / "hub/README.md").read_text().replace("{{org}}", org).replace("{{version}}", version)
    (dataset / "README.md").write_text(readme)
    print(f"Built {len(tasks)} tasks and dataset.toml for {org}/{DATASET} {version} in {dataset}")
    return dataset


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "hub/dist", help="output directory (default: hub/dist)")
    parser.add_argument("--org", default=ORG, help=f"Harbor Hub organization (default: {ORG})")
    args = parser.parse_args(argv)
    build(args.out, args.org)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
