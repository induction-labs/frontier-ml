"""Regrade a recorded job's trials with the current verifiers: `frontier-regrade jobs/<job> [task ...]`.

`harbor job regrade` grades every trial of a job and loads no plugins, so it can neither skip the trials whose
submissions are gone nor mount the Modal volume that holds a frozen submission. This command picks the trials that
can still be graded, regrades only those, with the frontier plugin, into a new job beside the source, and prints the
source job's rewards with the new grades in place. The source job is never modified.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

from harbor.cli.job_plugins import PluginConfig, attach_job_plugins, finalize_job_plugins
from harbor.cli.jobs import print_job_results_tables
from harbor.cli.utils import run_async
from harbor.environments.factory import EnvironmentFactory
from harbor.job import Job
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import JobConfig, SourceJobConfig
from harbor.models.trial.config import EnvironmentConfig, TaskConfig
from harbor.models.trial.paths import TrialPaths
from harbor.models.trial.result import TrialResult
from harbor.trial.regrade import expand_task_path, local_task_name


def recorded_trials(job_dir):
    """(trial directory, result) for every finished trial, as Harbor's regrade reads them."""
    for trial_dir in sorted(path for path in job_dir.iterdir() if path.is_dir()):
        paths = TrialPaths(trial_dir)
        if paths.config_path.exists() and paths.result_path.exists():
            yield trial_dir, TrialResult.model_validate_json(paths.result_path.read_text())


def submission_kept(trial_dir):
    # The plugin deletes a submission's volume once it is graded, unless the run passed keep_submission_volume=1.
    path = trial_dir / "submission-volume.json"
    return not path.is_file() or json.loads(path.read_text()).get("state") != "deleted"


def grade(result):
    rewards = result.verifier_result.rewards if result.verifier_result is not None else None
    if rewards and "reward" in rewards:
        return float(rewards["reward"])
    return None


def describe(result):
    reward = grade(result)
    if reward is not None:
        return f"{reward:.3f}"
    return result.exception_info.exception_type if result.exception_info is not None else "no grade"


def stage_source(job_dir, trial_dirs, staging):
    """A job directory holding only the chosen trials, since Harbor regrades every trial of its source job. Links
    keep it small, and Harbor records each trial's resolved path, so the regrade still points at the source trial."""
    staging.mkdir(parents=True)
    for name in ("config.json", "result.json"):
        if (job_dir / name).exists():
            (staging / name).symlink_to((job_dir / name).resolve())
    for trial_dir in trial_dirs:
        (staging / trial_dir.name).symlink_to(trial_dir.resolve(), target_is_directory=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="frontier-regrade", description=__doc__.split("\n\n")[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("job", type=Path, help="the job directory, e.g. jobs/2026-10-10__20-20-39")
    parser.add_argument("tasks", nargs="*", help="tasks to regrade (default: every trial whose submission was kept)")
    parser.add_argument("-p", "--task-path", type=Path, default=Path("tasks"),
                        help="directory of the tasks whose verifiers grade (default: tasks)")
    parser.add_argument("-n", "--n-concurrent", type=int, default=4, help="trials graded at once (default: 4)")
    args = parser.parse_args(argv)
    if not (args.job / "config.json").is_file():
        parser.error(f"{args.job} is not a job directory (no config.json)")

    trials = list(recorded_trials(args.job))
    unknown = set(args.tasks) - {result.task_name for _, result in trials}
    if unknown:
        parser.error(f"{args.job} has no trials of {', '.join(sorted(unknown))}")
    chosen, deleted = [], []
    for trial_dir, result in trials:
        if not args.tasks or result.task_name in args.tasks:
            (chosen if submission_kept(trial_dir) else deleted).append((trial_dir, result))
    if deleted:
        names = ", ".join(result.task_name for _, result in deleted) if args.tasks else f"{len(deleted)} trials"
        print(f"Skipping {names}: the submission was deleted after grading. "
              "Runs with --pk keep_submission_volume=1 keep every submission for regrading.")
    if not chosen:
        print("Nothing to regrade.")
        return 1

    task_dirs = {local_task_name(path): path for path in expand_task_path(args.task_path)}
    missing = {result.task_name for _, result in chosen} - task_dirs.keys()
    if missing:
        parser.error(f"{args.task_path} has no task {', '.join(sorted(missing))}")

    job_name = f"{args.job.name}-regrade-{datetime.now():%Y-%m-%d__%H-%M-%S}"
    jobs_dir = args.job.parent
    # Beside Harbor's own .sources cache; `harbor job resume` of the regrade job reads it again.
    staging = jobs_dir / ".regrade-sources" / job_name / args.job.name
    stage_source(args.job, [trial_dir for trial_dir, _ in chosen], staging)
    try:
        source_id = json.loads((args.job / "result.json").read_text())["id"]
    except (OSError, ValueError, KeyError):
        source_id = None
    config = JobConfig(
        job_name=job_name,
        jobs_dir=jobs_dir,
        n_concurrent_trials=args.n_concurrent,
        source_jobs=[SourceJobConfig(action="regrade", type="local", job_id=source_id, path=staging.resolve())],
        tasks=[TaskConfig(path=task_dirs[name]) for name in sorted({result.task_name for _, result in chosen})],
        environment=EnvironmentConfig(type=EnvironmentType.MODAL),
    )
    EnvironmentFactory.run_preflight(type=config.environment.type, import_path=config.environment.import_path)
    print(f"Regrading {len(chosen)} trial(s) of {args.job} into {jobs_dir / job_name}")

    async def regrade():
        job = await Job.create(config)
        plugins = await attach_job_plugins(job, [PluginConfig(import_path="frontier")])
        job_result = await job.run()
        print_job_results_tables(job_result)
        await finalize_job_plugins(plugins, job_result)
        return job_result

    job_result = run_async(regrade())

    # The source job's rewards, with each regraded trial's new grade in place of its old one.
    regraded = {Path(result.config.source_trial.path): result for result in job_result.trial_results}
    print(f"\n{args.job.name} with the new grades:")
    rewards = []
    for trial_dir, result in trials:
        new = regraded.get(trial_dir.resolve())
        print(f"  {result.task_name:<30} {describe(result)}" + (f" -> {describe(new)}" if new is not None else ""))
        rewards.append(grade(new if new is not None else result) or 0.0)
    print(f"Mean reward over {len(rewards)} trials (ungraded trials count as 0): {sum(rewards) / len(rewards):.3f}")
    return 0 if all(grade(result) is not None for result in regraded.values()) else 1
