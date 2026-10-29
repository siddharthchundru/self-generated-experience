"""End-to-end evaluation with policy-independent seeds and an adaptation hook.

Evaluation never updates a policy. Adaptation receives a fresh copy per
perturbation, all blocked pairs, and ONE total budget; its caller supplies the learner.
The Stage 1 no-update adapter tests collection/accounting only, not learning.
"""

import copy
from dataclasses import asdict, dataclass
import time

import numpy as np

from envs.pointmaze import make_env
from routes import trajectory_route
from eval.metrics import best_of_m, mean_defined, summarise_rollouts
from eval.protocol import EvaluationConfig, episode_seed, task_pairs


def rollout(env, policy, pair, seed, sampled, horizon):
    obs, _ = env.reset(seed=seed, options={"reset_cell": list(pair.start), "goal_cell": list(pair.goal)})
    goal = obs["desired_goal"].copy()
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    act = policy.start_episode(obs, pair, rng, sampled)
    positions = [obs["observation"][:2].copy()]
    first_success = None
    rewards = 0.0
    for step in range(1, horizon + 1):
        obs, reward, terminated, truncated, info = env.step(act(obs))
        positions.append(obs["observation"][:2].copy())
        rewards += float(reward)
        if info["success"] and first_success is None:
            first_success = step
        if not np.array_equal(obs["desired_goal"], goal):
            raise AssertionError("goal moved during evaluation")
        if terminated or (truncated != (step == horizon)):
            raise AssertionError("evaluation episode violated the fixed horizon")
    route = trajectory_route(positions, pair.maze)
    if info["success"] and route == "neither":
        raise AssertionError("successful west-east rollout has no crossing")
    return {
        "seed": seed, "row": pair.row, "sampled": sampled, "route": route,
        "success": bool(info["success"]), "first_success_step": first_success,
        "steps": horizon, "reward_sum": rewards,
        "final_goal_distance": float(np.linalg.norm(positions[-1]-goal)),
        "path_length": float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
    }


def collect(policy, pair, count, config, phase, sampled, wall=None, index_offset=0):
    blocked = frozenset({wall}) if wall is not None else frozenset()
    env = make_env(pair.maze, max_episode_steps=config.horizon, blocked_cells=blocked)
    try:
        return [rollout(env, policy, pair, episode_seed(config.seed, pair.row, phase, i+index_offset),
                        sampled, config.horizon) for i in range(count)]
    finally:
        env.close()


@dataclass
class AdaptationResult:
    policy: object
    environment_steps: int
    gradient_steps: int
    mode: str  # real learner implementations must distinguish themselves from the stand-in
    collection: list


def no_update_adaptation(policy, tasks, config, phase):
    """Stage 1 adapter: collect the exact budget, perform ZERO parameter updates.

    No old data is supplied or reused. This function is not the Stage 2 common
    adaptation procedure, and reports must not present it as learned recovery.
    """
    episodes = config.adaptation_steps // config.horizon
    rng = np.random.default_rng(episode_seed(config.seed, 0, phase, 0))
    records = []
    # Uniform task sampling, one fresh episode at a time, as in C3.
    for index in range(episodes):
        pair, wall = tasks[int(rng.integers(len(tasks)))]
        record = collect(policy, pair, 1, config, phase, True, wall, index_offset=index)[0]
        record["wall"] = list(wall)
        records.append(record)
    return AdaptationResult(policy, episodes*config.horizon, 0, "standin_no_parameter_updates", records)


def _adapt(policy, tasks, config, adapter, phase):
    if adapter is None:
        return None
    # One independent adapted policy per perturbation, trained across task pairs.
    result = adapter(copy.deepcopy(policy), tasks, config, phase)
    if result.environment_steps != config.adaptation_steps:
        raise AssertionError("adapter did not use the requested interaction budget")
    if result.gradient_steps < 0:
        raise AssertionError("negative gradient count")
    return result


def evaluate_policy(policy, config=EvaluationConfig(), adapter=None):
    started = time.perf_counter()
    pairs, all_records = [], []

    def keep(records, row, phase, wall=None):
        all_records.extend({**r, "phase": phase, "wall": list(wall) if wall else None} for r in records)

    for pair in task_pairs(config.maze):
        sampled = collect(policy, pair, config.sampled_rollouts, config, 10, True)
        greedy = collect(policy, pair, config.greedy_rollouts, config, 20, False)
        keep(sampled, pair.row, "unblocked_sampled")
        keep(greedy, pair.row, "unblocked_greedy")
        summary = summarise_rollouts(sampled, config.minimum_successes, config.bootstrap_resamples,
                                     episode_seed(config.seed, pair.row, 30, 0))
        row = {"row": pair.row, "start": list(pair.start), "goal": list(pair.goal),
               "shorter_route": pair.shorter, "is_primary_pair": pair.shorter is not None,
               "S_in": float(np.mean([r["success"] for r in greedy])), "sampled": summary}
        if pair.shorter is not None:
            wall = pair.wall(pair.shorter)
            blocked = collect(policy, pair, config.sampled_rollouts, config, 40, True, wall)
            keep(blocked, pair.row, "P1_zero_shot_sampled", wall)
            row["P1"] = {"wall": list(wall), "A_0": float(np.mean([r["success"] for r in blocked])),
                         "best_of_m": best_of_m(blocked, config.best_of_m)}
            used = {r: sum(ep["route"] == r for ep in greedy) for r in ("north", "south")}
            if max(used.values()) == 0:
                row["P1_own"] = {"blocked_route": None, "A_k": None,
                                  "status": "undefined_no_greedy_crossing", "greedy_route_counts": used}
            else:
                dominant = max(used, key=lambda r: (used[r], r == pair.shorter))
                own_wall = pair.wall(dominant)
                row["P1_own"] = {"blocked_route": dominant, "wall": list(own_wall),
                                  "greedy_route_counts": used}
        pairs.append(row)

    primary = [r for r in pairs if r["is_primary_pair"]]
    adaptation_runs = {}
    by_row = {pair.row: pair for pair in task_pairs(config.maze)}
    for name, phase in (("P1", 1000), ("P1_own", 2000)):
        eligible = [r for r in primary if r[name].get("wall") is not None]
        tasks = [(by_row[r["row"]], tuple(r[name]["wall"])) for r in eligible]
        result = _adapt(policy, tasks, config, adapter, phase) if tasks else None
        adaptation_runs[name] = {
            "status": result.mode if result else "not_run_no_adapter_or_eligible_pairs",
            "environment_steps": result.environment_steps if result else 0,
            "gradient_steps": result.gradient_steps if result else 0,
            "task_rows": [pair.row for pair, _ in tasks],
            "buffer_initialisation": "empty_required_by_adapter_contract",
        }
        if result:
            # Keep one shared adaptation run, not a separate budget for every pair.
            for record in result.collection:
                all_records.append({**record, "phase": f"{name}_adaptation_collection"})
        for row in eligible:
            pair = by_row[row["row"]]
            wall = tuple(row[name]["wall"])
            post = collect(result.policy, pair, config.greedy_rollouts, config, phase+500, False, wall) if result else []
            keep(post, pair.row, f"{name}_post_adaptation_greedy", wall)
            metrics = {"A_k": mean_defined([float(r["success"]) for r in post]),
                       "status": adaptation_runs[name]["status"]}
            if name == "P1":
                row[name]["adaptation"] = metrics
            else:
                row[name].update(metrics)
    aggregate = {
        "primary_rows": [r["row"] for r in primary],
        "S_in": mean_defined([r["S_in"] for r in primary]),
        "Rep": mean_defined([r["sampled"]["Rep"] for r in primary]),
        "Rep_qualifying_rows": [r["row"] for r in primary if r["sampled"]["Rep_qualifies"]],
        "Rep_excluded_rows": [r["row"] for r in primary if not r["sampled"]["Rep_qualifies"]],
        "Rep_excluded_as_zero": mean_defined([r["sampled"]["Rep_excluded_as_zero"] for r in primary]),
        "minority_route_success_rate": mean_defined([r["sampled"]["minority_route_success_rate"] for r in primary]),
        "A_0": mean_defined([r["P1"]["A_0"] for r in primary]),
        "best_of_m": mean_defined([r["P1"]["best_of_m"]["success_rate"] for r in primary]),
        "A_k": mean_defined([r["P1"]["adaptation"]["A_k"] for r in primary]),
        "A_k_status": "not_run" if adapter is None else sorted({r["P1"]["adaptation"]["status"] for r in primary}),
        "P1_own_by_blocked_route": {
            route: {"rows": [r["row"] for r in primary if r["P1_own"]["blocked_route"] == route],
                    "A_k": mean_defined([r["P1_own"]["A_k"] for r in primary
                                         if r["P1_own"]["blocked_route"] == route])}
            for route in ("north", "south")},
    }
    return {
        "schema_version": 1, "policy": policy.name, "config": asdict(config),
        "protocol_status": "stage1_planning_defaults_not_preregistered",
        "temperature": 1, "aggregate": aggregate, "pairs": pairs, "rollouts": all_records,
        "adaptation_runs": adaptation_runs,
        "environment_steps": sum(r["steps"] for r in all_records),
        "elapsed_seconds": time.perf_counter()-started,
        "limitations": [
            "No neural learner is implemented. Stand-in adaptation performs zero updates.",
            "Random policy has no greedy action: its ordinary random actions fill greedy evaluation slots.",
            "P1-own counts all greedy route crossings; ties choose the shorter route; no crossings is undefined.",
            "Best-of-m uses disjoint groups of zero-shot attempts, sharing data with A_0.",
            "M2, training exposure diagnostics, S_out and P2/P3 are outside this ring-task suite.",
        ],
    }
