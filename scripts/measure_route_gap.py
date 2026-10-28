"""Measure the route-length gap (Δ) on M1, set γ by the spec's geometry rule, and
measure simulator throughput. This is a Stage 1 measurement, run once.

Run from the repository root:  python scripts/measure_route_gap.py

What it does:
1. For each shorter-route pair in M1's ring (west-east pairs in rows 1-3 and
   5-7; row 4 is a tie and excluded), it runs the scripted expert N_SEEDS times
   on each route and records the step at which the ball first enters the goal
   radius. A pair's gap is the median over seeds of (longer - shorter).
2. Δ is the median of the pair gaps (spec section 5).
3. γ is set so that the predicted pretrained shorter-route share,
   1 / (1 + γ^Δ), equals 0.65:  γ = (0.35 / 0.65) ** (1 / Δ).
   The pilot-only sensitivity values (target shares 0.58 and 0.72) are computed
   the same way. γ uses only maze geometry and the expert, so it is fixed before
   any practice result exists, and it is not tuned.
4. It also times plain environment stepping (random actions) and expert
   episodes, for the compute estimate.

Writes results/stage1/route_gap.json and prints a summary.
"""

import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from envs.mazes import M1  # noqa: E402
from envs.pointmaze import make_env  # noqa: E402
from expert import run_episode  # noqa: E402

N_SEEDS = 50
T = 100  # long enough that every expert episode reaches its goal; not the experiment's T
TARGET_SHARE = 0.65
SENSITIVITY_SHARES = (0.58, 0.72)
SHORTER_ROUTE_ROWS = (1, 2, 3, 5, 6, 7)


def gamma_for(share, delta):
    return ((1 - share) / share) ** (1 / delta)


def main():
    env = make_env("M1", max_episode_steps=T)
    pairs = []
    for row in SHORTER_ROUTE_ROWS:
        shorter, longer = ("north", "south") if row < 4 else ("south", "north")
        steps = {}
        for route in (shorter, longer):
            steps[route] = []
            for seed in range(N_SEEDS):
                ep = run_episode(env, M1, (row, 1), (row, 7), route, seed=seed)
                assert ep["success"] and ep["first_success_step"] is not None
                steps[route].append(ep["first_success_step"])
        gaps = np.array(steps[longer]) - np.array(steps[shorter])  # same seed, same noise
        pairs.append({
            "start": [row, 1], "goal": [row, 7],
            "shorter_route": shorter,
            "shorter_steps_median": float(np.median(steps[shorter])),
            "longer_steps_median": float(np.median(steps[longer])),
            "longer_steps_max": int(max(steps[longer])),
            "gap_median": float(np.median(gaps)),
            "gap_min": int(gaps.min()), "gap_max": int(gaps.max()),
        })

    delta = float(np.median([p["gap_median"] for p in pairs]))
    gamma = gamma_for(TARGET_SHARE, delta)
    half_life_steps = math.log(0.5) / math.log(gamma)
    for p in pairs:  # what the rule predicts for each pair, given its own gap
        p["predicted_shorter_share"] = round(1 / (1 + gamma ** p["gap_median"]), 3)

    # Throughput: plain stepping with random actions, then expert episodes.
    rng = np.random.default_rng(0)
    env.reset(seed=0, options={"reset_cell": [4, 1], "goal_cell": [4, 7]})
    n, t0 = 20_000, time.perf_counter()
    for i in range(n):
        *_, truncated, _ = env.step(rng.uniform(-1, 1, 2).astype(np.float32))
        if truncated:
            env.reset(seed=i, options={"reset_cell": [4, 1], "goal_cell": [4, 7]})
    random_sps = n / (time.perf_counter() - t0)
    t0 = time.perf_counter()
    for seed in range(100):
        run_episode(env, M1, (2, 1), (2, 7), "south", seed=seed)
    expert_sps = 100 * T / (time.perf_counter() - t0)
    env.close()

    result = {
        "n_seeds_per_route": N_SEEDS,
        "pairs": pairs,
        "delta_steps": delta,
        "delta_seconds": delta / 10,
        "target_shorter_share": TARGET_SHARE,
        "gamma": round(gamma, 5),
        "half_life_steps": round(half_life_steps, 1),
        "half_life_seconds": round(half_life_steps / 10, 2),
        "pilot_sensitivity_gammas": {str(s): round(gamma_for(s, delta), 5) for s in SENSITIVITY_SHARES},
        "longest_expert_longer_route_steps": max(p["longer_steps_max"] for p in pairs),
        "throughput": {
            "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
            "single_process_env_steps_per_second_random_actions": round(random_sps),
            "single_process_env_steps_per_second_expert": round(expert_sps),
        },
    }
    out = ROOT / "results" / "stage1" / "route_gap.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
