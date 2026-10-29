"""Metrics from rollout records, with explicit denominators and missing values.

Successful 'neither' labels are errors. Low-success Rep is None, never silently
zero. Per-route quality includes failed attempts classified into that route;
time-to-goal uses the successful subset and is reported in actions and seconds.
"""

import math

import numpy as np


ROUTES = ("north", "south")


def mean_defined(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def summarise_rollouts(records, minimum_successes=20, bootstrap_resamples=2000, seed=0):
    if not records:
        raise ValueError("cannot summarise zero rollouts")
    for r in records:
        if r["route"] not in (*ROUTES, "neither"):
            raise ValueError(f"unknown route {r['route']}")
        if r["success"] and r["route"] == "neither":
            raise ValueError("successful task rollout did not cross the ring midline")
    n = len(records)
    counts = {route: sum(r["route"] == route for r in records)
              for route in (*ROUTES, "neither")}
    successes = {route: sum(r["success"] and r["route"] == route for r in records)
                 for route in ROUTES}
    total = sum(successes.values())
    minority_count = min(successes.values())
    raw_share = minority_count / total if total else None
    qualifies = total >= minimum_successes
    rep = raw_share if qualifies else None
    interval = None
    if qualifies:
        # Resample all rollouts, retaining failures and the conditional denominator.
        probabilities = [successes["north"] / n, successes["south"] / n, (n-total) / n]
        draws = np.random.default_rng(seed).multinomial(n, probabilities, size=bootstrap_resamples)
        den = draws[:, 0] + draws[:, 1]
        shares = np.minimum(draws[den > 0, 0], draws[den > 0, 1]) / den[den > 0]
        interval = np.quantile(shares, [0.025, 0.975]).tolist()
    quality = {}
    for route in ROUTES:
        reached = [r["first_success_step"] for r in records
                   if r["route"] == route and r["success"]]
        if any(t is None for t in reached):
            raise ValueError("successful rollout lacks a first-success time")
        quality[route] = {
            "attempts": counts[route], "successes": successes[route],
            "success_rate_given_route": successes[route] / counts[route] if counts[route] else None,
            "first_success_actions_mean": mean_defined(reached),
            "first_success_seconds_mean": mean_defined([t / 10 for t in reached]),
        }
    entropy = -sum((v/total) * math.log(v/total) for v in successes.values() if v) if total else None
    return {
        "rollouts": n, "successes": total, "sampled_success_rate": total/n,
        "route_attempts": counts, "route_successes": successes,
        "Rep": rep, "Rep_qualifies": qualifies, "Rep_bootstrap_95": interval,
        "Rep_excluded_as_zero": rep if qualifies else 0.0,
        "minority_route": (None if successes["north"] == successes["south"]
                           else min(successes, key=successes.get)),
        "minority_route_success_rate": minority_count/n,
        "route_class_counts": {str(threshold): sum(v/total >= threshold for v in successes.values())
                               if total else 0 for threshold in (0.05, 0.10, 0.20)},
        "route_entropy_nats": entropy,
        "per_route_quality": quality,
    }


def best_of_m(records, m):
    """Empirical success of disjoint independent attempt groups, not 1-(1-p)^m."""
    if m <= 0 or not records or len(records) % m:
        raise ValueError("best-of-m requires nonempty complete groups")
    outcomes = [any(r["success"] for r in records[i:i+m]) for i in range(0, len(records), m)]
    return {"m": m, "groups": len(outcomes), "successful_groups": sum(outcomes),
            "success_rate": float(np.mean(outcomes))}


def common_pair_rep(left, right):
    """Within-seed C4-C3 style comparison; rows must match, None means no comparison."""
    if set(left) != set(right):
        raise ValueError("comparison requires the same task rows")
    common = sorted(row for row in left if left[row]["Rep"] is not None and right[row]["Rep"] is not None)
    return {"common_rows": common, "dropped_rows": sorted(set(left)-set(common)),
            "left_minus_right": mean_defined([left[r]["Rep"]-right[r]["Rep"] for r in common])}
