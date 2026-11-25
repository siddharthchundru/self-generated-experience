"""Run the learner-capacity check (spec section 7). See `src/capacity.py` for the rules.

Run one learner on one maze:
    python scripts/run_capacity.py --maze M1 --learner weighted --out OUT/M1_weighted
Or all four (M1 and M2 x weighted and plain), as four single-thread processes at once,
then a summary:
    python scripts/run_capacity.py --all --out OUT

Each run folder holds capacity.json (thresholds, every evaluation attempt, pass or
fail and what to do next), final.pt and run_record.json; capacity.json is
written last. --all also writes OUT/capacity_summary.json: the check passes only
if all four pass. --quick shrinks everything for a smoke test and is refused for
results/. Writing into results/ needs a clean, committed tree.
"""

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from capacity import MAZES_CHECKED, ON_FAIL, CapacitySettings, run_capacity  # noqa: E402
from gcsl import pin_single_thread  # noqa: E402
from provenance import claim_output_folder, guard_results_write, run_record, write_text_atomic  # noqa: E402

QUICK = {"updates": 300, "random_pair_episodes": 20, "task_pair_rollouts": 2}
LEARNERS = ("weighted", "plain")


def run_one(args):
    import torch
    settings = CapacitySettings(maze=args.maze, learner=args.learner, seed=args.seed,
                                updates=args.updates, learning_rate=args.learning_rate, on_fail=args.on_fail)
    if args.quick:
        settings = replace(settings, **QUICK)
    record = run_record(ROOT)
    guard_results_write(ROOT, args.out, args.allow_dirty)
    claim_output_folder(args.out)
    record.update({"allow_dirty": args.allow_dirty, "quick_smoke_only": args.quick,
                   "torch_threads": {"intra_op": torch.get_num_threads(),
                                     "inter_op": torch.get_num_interop_threads()}})
    (args.out / "run_record.json").write_text(json.dumps(record, indent=2))
    result, policy = run_capacity(settings)
    result["quick_smoke_only"] = args.quick
    torch.save(policy.network.state_dict(), args.out / "final.pt")
    write_text_atomic(args.out / "capacity.json", json.dumps(result, indent=2) + "\n")
    last = result["attempts"][-1]
    print(f"capacity {settings.maze} {settings.learner}: random pairs {last['random_pairs']:.3f}"
          + (f", task pairs {last['task_pairs']:.3f}" if "task_pairs" in last else "")
          + f" after {last['updates']} updates: {'passed' if result['passed'] else 'FAILED'}")
    return 0 if result["passed"] else 1


def run_all(args):
    guard_results_write(ROOT, args.out, args.allow_dirty)
    if args.quick and "results" in args.out.resolve().parts:
        raise SystemExit("--quick runs never go into results/")
    claim_output_folder(args.out)   # audit of 26 Nov, SF6: never report an earlier call's results
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    processes = {}
    for maze in MAZES_CHECKED:
        for learner in LEARNERS:
            command = [sys.executable, __file__, "--maze", maze, "--learner", learner,
                       "--seed", str(args.seed), "--updates", str(args.updates),
                       "--learning-rate", str(args.learning_rate), "--on-fail", args.on_fail,
                       "--out", str(args.out / f"{maze}_{learner}")]
            command += ["--quick"] * args.quick + ["--allow-dirty"] * args.allow_dirty
            processes[(maze, learner)] = subprocess.Popen(command, cwd=ROOT, env=env)
    expected = replace(CapacitySettings(seed=args.seed, updates=args.updates, learning_rate=args.learning_rate,
                                        on_fail=args.on_fail), **(QUICK if args.quick else {}))
    runs, problems = {}, []
    for (maze, learner), process in processes.items():
        code = process.wait()
        file = args.out / f"{maze}_{learner}" / "capacity.json"
        result = json.loads(file.read_text()) if file.exists() else None
        planned = asdict(replace(expected, maze=maze, learner=learner))
        if code not in (0, 1) or result is None:     # 1 is a failed check; anything else is a crash
            problems.append(f"{maze} {learner}: the run crashed (exit code {code})")
            result = None
        elif result["settings"] != planned or result["quick_smoke_only"] != args.quick:
            problems.append(f"{maze} {learner}: capacity.json does not match this call's settings")
            result = None
        runs[f"{maze}_{learner}"] = result
    summary = {"passed": not problems and all(r["passed"] for r in runs.values()),
               "problems": problems,
               "runs": {k: (None if r is None else {"passed": r["passed"], "attempts": r["attempts"]})
                        for k, r in runs.items()},
               "quick_smoke_only": args.quick}
    summary["next"] = "none" if summary["passed"] else (
        "rerun the crashed runs" if problems else
        "stop before tuning: bring the learner's capacity to Siddharth as a design question (network size)")
    write_text_atomic(args.out / "capacity_summary.json", json.dumps(summary, indent=2) + "\n")
    print("capacity check", "passed" if summary["passed"] else "FAILED", *problems)
    return 0 if summary["passed"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--maze", choices=MAZES_CHECKED)
    parser.add_argument("--learner", choices=LEARNERS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=CapacitySettings.updates)
    parser.add_argument("--learning-rate", type=float, default=CapacitySettings.learning_rate)
    parser.add_argument("--on-fail", choices=ON_FAIL, default=CapacitySettings.on_fail)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    if args.all == bool(args.maze or args.learner):
        raise SystemExit("give either --all, or --maze and --learner")
    if not args.all and not (args.maze and args.learner):
        raise SystemExit("give both --maze and --learner")
    if args.quick and "results" in args.out.resolve().parts:
        raise SystemExit("--quick runs never go into results/")
    return run_all(args) if args.all else run_one(args)


if __name__ == "__main__":
    pin_single_thread()
    sys.exit(main())
