"""Compute Stage 2 scenarios from measured Mac rates and explicit assumptions.

Run: python scripts/estimate_compute.py
No N, seed cap, T or batch size is frozen by this estimate. Training/sampler and
selector integration are absent, so overhead multipliers are sensitivity cases,
not confidence intervals. Sequential CPU time; no parallel speedup is assumed.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from envs.mazes import M1, M2, free_cells
from eval.compute import ComputeAssumptions, counts, estimate_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, default=ROOT / "results/stage1/mac_throughput.json")
    parser.add_argument("--assumptions", type=Path, help="JSON overrides of ComputeAssumptions")
    parser.add_argument("--output", type=Path, default=ROOT / "results/stage1/compute_estimate.json")
    args = parser.parse_args()
    measured = json.loads(args.measurements.read_text())
    overrides = json.loads(args.assumptions.read_text()) if args.assumptions else {}
    actual_pairs = sum(len(free_cells(grid))*(len(free_cells(grid))-1) for grid in (M1, M2))
    a = ComputeAssumptions(**{"capacity_demo_episodes": actual_pairs, **overrides})
    if a.horizon != measured["simulator"]["horizon"]:
        raise ValueError("re-measure with the chosen horizon before estimating it")
    scenarios = []
    for budget in (99960, 249960, 999960):  # tuning, pilot N, cap N (whole episodes)
        for seeds in (6, 20):
            scenario = counts(a, budget, seeds)
            scenario["median_rates"] = estimate_runtime(scenario, measured)
            scenario["slowest_rates"] = estimate_runtime(scenario, measured, conservative=True)
            scenarios.append(scenario)
    result = {"status": "planning_scenarios_not_a_frozen_compute_cap", "assumptions": asdict(a),
              "measurement_file": str(args.measurements.relative_to(ROOT)) if args.measurements.is_relative_to(ROOT) else str(args.measurements),
              "benchmark_batch_size": measured["network"]["batch_size"], "scenarios": scenarios,
              "limitations": [
                  "N, warmup, update ratio, tuning grid and evaluation frequency are explicit provisional assumptions.",
                  "Synthetic CPU training omits replay sampling, real learner integration, logging and checkpoint I/O.",
                  "KDE is a scoring surrogate, not the final C4 implementation.",
                  "Pilot repeats the same N; optional pilot re-learning is off unless requested in assumptions.",
                  "Each main policy has two adaptations; re-learned policies receive unblocked evaluation only (H4).",
                  "Gamma sensitivity gets unblocked evaluation only; the k grid is additional P1 calibration on C0.",
                  "M0 uses 140 balanced demo episodes as a conservative volume allowance, not a frozen D0 allocation.",
                  "Does not cover Stage 3 replication/extra learners, development/debug reruns, or manipulation.",
                  "No claimed statistical precision: 25% and 100% overhead are sensitivity scenarios.",
              ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print("N / seeds / total sequential CPU hours (median rates +25% overhead)")
    for s in scenarios:
        print(s["budget_N"], s["main_seeds"], round(s["median_rates"]["hours_with_25_percent_overhead"], 2))


if __name__ == "__main__":
    main()
