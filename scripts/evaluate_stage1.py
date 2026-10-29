"""Run the complete ring-task evaluation on expert and random stand-ins.

Run: python scripts/evaluate_stage1.py
Use --quick for a smoke check; the default uses 100 sampled rollouts and the
spec's 20-success rule. No policy learning takes place in this Stage 1 gate.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval.policies import ExpertPolicy, RandomPolicy
from eval.protocol import EvaluationConfig, task_pairs
from eval.suite import evaluate_policy, no_update_adaptation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--maze", choices=("M0", "M1"), default="M1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/stage1/evaluation.json")
    args = parser.parse_args()
    config = EvaluationConfig(maze=args.maze, **({"sampled_rollouts": 10, "minimum_successes": 2,
        "greedy_rollouts": 2, "bootstrap_resamples": 200, "adaptation_steps": 100}
        if args.quick else {}))
    reports = []
    for policy in (ExpertPolicy(), RandomPolicy()):
        report = evaluate_policy(policy, config, no_update_adaptation)
        reports.append(report)
        print(json.dumps({"policy": policy.name, **report["aggregate"],
                          "seconds": round(report["elapsed_seconds"], 2)}, indent=2), flush=True)
    expert = reports[0]["aggregate"]
    expected_rows = [p.row for p in task_pairs(config.maze) if p.shorter is not None]
    pipeline_passes = expert["S_in"] == 1.0 and expert["Rep_qualifying_rows"] == expected_rows
    comparison_file = ROOT / "results/stage1/route_handcheck/human_comparison.json"
    handcheck = json.loads(comparison_file.read_text()) if comparison_file.exists() else None
    handcheck_valid = bool(handcheck and handcheck["ready_for_review"] and
        handcheck.get("classifier_source_sha256") == hashlib.sha256((ROOT / "src/routes.py").read_bytes()).hexdigest() and
        handcheck.get("maze_source_sha256") == hashlib.sha256((ROOT / "src/envs/mazes.py").read_bytes()).hexdigest() and
        handcheck["trajectory_sha256"] == hashlib.sha256(
            (ROOT / "results/stage1/route_handcheck/trajectories.json").read_bytes()).hexdigest())
    result = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for folder in (ROOT/"src", ROOT/"scripts") for p in sorted(folder.rglob("*.py"))},
        "machine": {"system": platform.system(), "architecture": platform.machine(), "python": platform.python_version()},
        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "gymnasium", "gymnasium-robotics", "mujoco")},
        "quick_smoke_only": args.quick, "evaluation_pipeline_passes": pipeline_passes,
        "stage1_gate_complete": False,
        "human_classifier_check_valid": handcheck_valid,
        "outstanding_gate_evidence": (["independent human classifier labels"] if not handcheck_valid else []) + ["independent audit"],
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"Wrote {args.output}", flush=True)
    if not pipeline_passes:
        raise SystemExit("expert failed the evaluation pipeline checks")


if __name__ == "__main__":
    main()
