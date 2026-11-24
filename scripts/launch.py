"""Launch many runs as separate processes: training, evaluation and re-learning.

Run: python scripts/launch.py PLAN.json [--workers W] [--dry-run] [--retry-incomplete]

One process per job, one CPU thread per process (Stage 2 setting: one process
per seed and condition on a rented CPU machine). Up to W jobs run at once
(default: the machine's CPU count). The launcher only schedules: every job is
an ordinary call to one of the scripts, so a launched run is identical to one
started by hand.

The plan (JSON):
    {
      "out": "results/stage2/pilot",          # every job writes under this folder
      "settings": {"purpose": "pilot", ...},  # TrainingSettings fields shared by every run
      "seeds": [0, 1, 2, 3, 4, 5],
      "conditions": ["demonstrations_only", "expert_top_up", "random_actions",
                     "task_directed_practice", "frontier_novelty"],
      "plain_conditions": ["task_directed_practice", "frontier_novelty"],
      "evaluate": {"final": true, "pretrained": true, "skip_adaptation": false, "quick": false},
      "relearning": {"enabled": false, "init": "fresh"}
    }
"out", "seeds" and settings["purpose"] are required. Defaults: every condition, no plain arm,
final and pretrained policies evaluated with adaptation, no re-learning.
"settings" may not set condition, seed or learner: the launcher sets those
per run.

Jobs and their order:
- train: one per seed, condition and learner -> OUT/runs/<learner>/<condition>/seed<N>/
- evaluate final: after its training run -> evaluation_final.json in the run folder
- evaluate pretrained: once per seed and learner, after every training run in
  that group has finished and passed the pretrained check (they share one
  pretrained policy) -> OUT/pretrained/<learner>/seed<N>.json
- re-learn: after its training run, main learner only ->
  OUT/relearning/<condition>/seed<N>/, then its evaluation.
A job whose finished output already exists is skipped, so rerunning the same
plan resumes it, but only if that output was made with this plan's settings
(runs) or evaluation protocol (evaluations); otherwise the job fails, so one
launch folder can never mix budgets, ratios or k. A run folder left by a crash (files but no summary.json) is a
failure unless --retry-incomplete, which renames it to <folder>.incomplete-<time>
and runs the job again. A failed job's dependants are skipped.

Before anything runs, every run's settings and evaluation protocol are built
and checked (so a pilot plan with the k placeholder, or with quick
evaluation, fails here, not in 60 processes), and a plan writing into
results/ needs a clean, committed tree (--allow-dirty passes through to every
job). One launch at a time per output folder (OUT/.launch.lock); if the
launcher stops for any reason, its running jobs are terminated and the
manifest is still written. Afterwards the launcher runs the pretrained check over
every run in OUT and writes OUT/launch_manifest.json: the plan, the commit,
and every job's command, status, duration and log file (OUT/logs/).
"""

import argparse
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

sys.path.insert(0, str(ROOT / "scripts"))

from collection import CONDITIONS, PLAIN_CONDITIONS  # noqa: E402
import evaluate_run  # noqa: E402
from provenance import git_state, guard_results_write  # noqa: E402
from relearning import INITS  # noqa: E402
from runs import check_pretrained, load_run  # noqa: E402
from training import settings_from_dict  # noqa: E402

PER_RUN = ("condition", "seed", "learner")
PRIORITY = {"train": 0, "relearn": 1, "evaluate_pretrained": 2, "evaluate": 3}


@dataclass
class Job:
    name: str
    kind: str
    command: list
    done: Path                     # exists when the job has finished
    folder: Path = None            # run folder, for crash detection (train and relearn)
    deps: list = field(default_factory=list)
    group: list = field(default_factory=list)  # run folders to check before evaluate_pretrained
    expect: dict = field(default_factory=dict)  # what an existing output must match (audit of 24 Nov, SF2)
    status: str = "pending"
    seconds: float = None
    log: str = None


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    unknown = set(plan) - {"out", "settings", "seeds", "conditions", "plain_conditions", "evaluate",
                           "relearning", "note"}
    if unknown:
        raise SystemExit(f"unknown plan keys: {sorted(unknown)}")
    if "out" not in plan or not plan.get("seeds"):
        raise SystemExit("a plan needs 'out' and a non-empty 'seeds' list")
    settings = plan.setdefault("settings", {})
    if "purpose" not in settings:  # audit of 24 Nov, MI3: a forgotten purpose would run as "smoke"
        raise SystemExit("the plan's settings must name a purpose (smoke, tuning, capacity, precondition, "
                         "pilot or main)")
    if set(settings) & set(PER_RUN):
        raise SystemExit(f"the plan's settings may not set {PER_RUN}: the launcher sets them per run")
    plan.setdefault("conditions", list(CONDITIONS))
    plan.setdefault("plain_conditions", [])
    for name in plan["conditions"]:
        if name not in CONDITIONS:
            raise SystemExit(f"unknown condition {name!r}")
    for name in plan["plain_conditions"]:
        if name not in PLAIN_CONDITIONS:
            raise SystemExit(f"the plain-learner arm has {PLAIN_CONDITIONS} only, not {name!r}")
    if len(set(plan["seeds"])) != len(plan["seeds"]):
        raise SystemExit("seeds repeat")
    evaluate = {"final": True, "pretrained": True, "skip_adaptation": False, "quick": False}
    plan["evaluate"] = {**evaluate, **plan.get("evaluate", {})}
    plan["relearning"] = {"enabled": False, "init": "fresh", **plan.get("relearning", {})}
    if plan["relearning"]["init"] not in INITS:
        raise SystemExit(f"relearning init must be one of {INITS}")
    return plan


def build_jobs(plan, allow_dirty=False):
    """Every job in the plan, each run's settings checked by building them."""
    out = (ROOT / plan["out"]).resolve() if not Path(plan["out"]).is_absolute() else Path(plan["out"])
    py = sys.executable
    dirty = ["--allow-dirty"] if allow_dirty else []
    ev = plan["evaluate"]
    eval_flags = (["--skip-adaptation"] if ev["skip_adaptation"] else []) + (["--quick"] if ev["quick"] else [])
    jobs = []
    arms = [("weighted", c) for c in plan["conditions"]] + [("plain", c) for c in plan["plain_conditions"]]
    for seed in plan["seeds"]:
        group = {}
        for learner, condition in arms:
            values = {**plan["settings"], "condition": condition, "seed": seed, "learner": learner}
            settings = settings_from_dict(values)  # raises on any refused setting, before anything runs
            planned_settings = asdict(settings)
            config, adapter = evaluate_run.build(settings, ev["quick"], ev["skip_adaptation"])  # SF4
            eval_expect = {"config": asdict(config), "quick_smoke_only": ev["quick"],
                           "adaptation_run": adapter is not None}
            name = f"{learner}/{condition}/seed{seed}"
            folder = out / "runs" / learner / condition / f"seed{seed}"
            settings_file = out / "job_settings" / f"{learner}_{condition}_seed{seed}.json"
            train = Job(f"train {name}", "train",
                        [py, str(ROOT / "scripts/run_training.py"), "--condition", condition,
                         "--settings", str(settings_file), "--out", str(folder)] + dirty,
                        folder / "summary.json", folder, expect={"settings": planned_settings})
            train.settings_file, train.settings_values = settings_file, values
            jobs.append(train)
            group.setdefault(learner, []).append(folder)
            if ev["final"]:
                jobs.append(Job(f"evaluate {name}", "evaluate",
                                [py, str(ROOT / "scripts/evaluate_run.py"), "--run", str(folder)]
                                + eval_flags + dirty,
                                folder / "evaluation_final.json", deps=[train.name],
                                expect={**eval_expect, "which": "final"}))
            if plan["relearning"]["enabled"] and learner == "weighted":
                re_folder = out / "relearning" / condition / f"seed{seed}"
                relearn = Job(f"relearn {name}", "relearn",
                              [py, str(ROOT / "scripts/run_relearning.py"), "--run", str(folder),
                               "--out", str(re_folder), "--init", plan["relearning"]["init"]] + dirty,
                              re_folder / "summary.json", re_folder, deps=[train.name],
                              expect={"settings": planned_settings, "init": plan["relearning"]["init"]})
                jobs.append(relearn)
                if ev["final"]:
                    jobs.append(Job(f"evaluate relearn {name}", "evaluate",
                                    [py, str(ROOT / "scripts/evaluate_run.py"), "--run", str(re_folder)]
                                    + eval_flags + dirty,
                                    re_folder / "evaluation_final.json", deps=[relearn.name],
                                    expect={**eval_expect, "which": "final"}))
        if ev["pretrained"]:
            for learner, folders in group.items():
                pre_settings = settings_from_dict({**plan["settings"], "condition": Path(folders[0]).parent.name,
                                                   "seed": seed, "learner": learner})
                pre_config, pre_adapter = evaluate_run.build(pre_settings, ev["quick"], ev["skip_adaptation"])
                target = out / "pretrained" / learner / f"seed{seed}.json"
                jobs.append(Job(f"evaluate pretrained {learner}/seed{seed}", "evaluate_pretrained",
                                [py, str(ROOT / "scripts/evaluate_run.py"), "--run", str(folders[0]),
                                 "--which", "pretrained", "--out", str(target)] + eval_flags + dirty,
                                target, deps=[f"train {learner}/{Path(f).parent.name}/seed{seed}"
                                              for f in folders],
                                group=folders,
                                expect={"config": asdict(pre_config), "quick_smoke_only": ev["quick"],
                                        "adaptation_run": pre_adapter is not None, "which": "pretrained"}))
    return out, jobs


def mismatch(job):
    """Why an existing output does not match the plan, or None (audit of 24 Nov, SF2).

    A resumed launch must not keep a finished run made with other settings
    (another N, ratio or k), or an evaluation made with another protocol.
    """
    try:
        if job.kind in ("train", "relearn"):
            settings, _, _ = load_run(job.folder)  # also checks settings.json against summary.json
            if asdict(settings) != job.expect["settings"]:
                return f"{job.folder} was run with other settings than this plan's"
            if job.kind == "relearn":
                init = json.loads((job.folder / "relearning.json").read_text())["init"]
                if init != job.expect["init"]:
                    return f"{job.folder} was re-learned with init {init!r}, not {job.expect['init']!r}"
        else:
            result = json.loads(job.done.read_text())
            recorded = {"config": result["evaluation"]["config"],
                        "quick_smoke_only": result["evaluation"]["quick_smoke_only"],
                        "adaptation_run": result["evaluation"]["adaptation_run"], "which": result["run"]["which"]}
            if recorded != job.expect:
                return f"{job.done} was evaluated with another protocol than this plan's"
    except Exception as error:  # noqa: BLE001
        return f"{job.done} could not be checked against the plan ({type(error).__name__}: {error})"
    return None


class LaunchLock:
    """One launcher per output folder (audit of 24 Nov, SF3). The lock file is
    created with O_EXCL and removed when the launch ends; if a launcher was
    killed without cleaning up, delete OUT/.launch.lock by hand after checking
    that no launcher is running."""

    def __init__(self, out):
        self.path = Path(out) / ".launch.lock"

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise SystemExit(f"another launch holds {self.path} ({self.path.read_text().strip()}); "
                             "if none is running, delete the file") from None
        with os.fdopen(fd, "w") as handle:
            handle.write(f"pid {os.getpid()}, started {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)
        return False


def _environment():
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[name] = "1"
    return env


def run_jobs(out, jobs, workers, retry_incomplete=False, echo=print):
    """Run `jobs` respecting dependencies, at most `workers` at a time."""
    by_name = {job.name: job for job in jobs}
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for job in jobs:
        if job.done.exists():
            reason = mismatch(job) if job.expect else None
            job.status, job.log = ("failed", reason) if reason else ("done_earlier", job.log)
            if reason:
                echo(f"FAILED  {job.name}: {reason}")
        elif job.folder is not None and job.folder.exists() and any(job.folder.iterdir()):
            if retry_incomplete:
                job.folder.rename(job.folder.with_name(f"{job.folder.name}.incomplete-{stamp}"))
            else:
                job.status = "failed"
                job.log = f"{job.folder} holds a crashed run; rerun with --retry-incomplete"
    running = {}
    try:
        _schedule(jobs, by_name, running, logs, workers, echo)
    finally:  # an exception or Ctrl-C here must not leave orphaned children (audit of 24 Nov, SF3)
        for name, (process, handle, _) in running.items():
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
            handle.close()
            by_name[name].status = "failed"
            by_name[name].log = f"{by_name[name].log} (terminated: the launcher stopped)"
    return jobs


def _schedule(jobs, by_name, running, logs, workers, echo):
    while True:
        for job in jobs:  # dependants of failures are skipped
            if job.status == "pending" and any(by_name[d].status in ("failed", "skipped") for d in job.deps):
                job.status = "skipped"
        ready = sorted((j for j in jobs if j.status == "pending"
                        and all(by_name[d].status in ("done", "done_earlier") for d in j.deps)),
                       key=lambda j: PRIORITY[j.kind])
        while ready and len(running) < workers:
            job = ready.pop(0)
            if job.group:
                report = check_pretrained(folders=job.group)
                if not report["passed"]:
                    job.status, job.log = "failed", "pretrained check failed: " + "; ".join(report["problems"])
                    echo(f"FAILED  {job.name}: {job.log}")
                    continue
            if getattr(job, "settings_file", None) is not None:
                job.settings_file.parent.mkdir(parents=True, exist_ok=True)
                job.settings_file.write_text(json.dumps(job.settings_values, indent=2))
            log = logs / (job.name.replace("/", "_").replace(" ", "_") + ".log")
            job.log = str(log)
            handle = open(log, "a")
            handle.write(f"$ {shlex.join(job.command)}\n")
            handle.flush()
            process = subprocess.Popen(job.command, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT,
                                       env=_environment())
            running[job.name] = (process, handle, time.time())
            job.status = "running"
            echo(f"started {job.name}")
        if not running:
            break
        time.sleep(0.5)
        for name, (process, handle, started) in list(running.items()):
            if process.poll() is None:
                continue
            handle.close()
            job = by_name[name]
            job.seconds = round(time.time() - started, 1)
            job.status = "done" if process.returncode == 0 and job.done.exists() else "failed"
            echo(f"{'done   ' if job.status == 'done' else 'FAILED '} {name} ({job.seconds} s)")
            del running[name]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--dry-run", action="store_true", help="list the jobs and stop")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    plan = load_plan(args.plan)
    out, jobs = build_jobs(plan, args.allow_dirty)
    guard_results_write(ROOT, out, args.allow_dirty)
    if args.dry_run:
        for job in jobs:
            state = "done" if job.done.exists() else "to run"
            print(f"[{state}] {job.name}" + (f"  (after {len(job.deps)})" if job.deps else ""))
        print(f"{len(jobs)} jobs, {sum(not j.done.exists() for j in jobs)} to run, up to {args.workers} at once")
        return 0
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    state = git_state(ROOT)
    report = None
    with LaunchLock(out):
        try:
            run_jobs(out, jobs, args.workers, args.retry_incomplete)
            report = check_pretrained(out) if (out / "runs").exists() else None
        finally:  # the manifest is written however the launch ends (audit of 24 Nov, SF3)
            manifest_file = out / "launch_manifest.json"
            history = json.loads(manifest_file.read_text())["launches"] if manifest_file.exists() else []
            history.append({
                "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), **state,
                "workers": args.workers, "plan": plan,
                "jobs": [{"name": j.name, "kind": j.kind, "status": j.status, "seconds": j.seconds,
                          "log": j.log, "command": shlex.join(j.command)} for j in jobs],
                "pretrained_check": report,
            })
            manifest_file.write_text(json.dumps({"launches": history}, indent=2) + "\n")
    counts = {s: sum(j.status == s for j in jobs) for s in ("done", "done_earlier", "failed", "skipped")}
    print(f"jobs: {counts}; pretrained check: "
          f"{'passed' if report and report['passed'] else 'FAILED' if report else 'no runs'}")
    return 0 if counts["failed"] == counts["skipped"] == 0 and (report is None or report["passed"]) else 1


if __name__ == "__main__":
    sys.exit(main())
