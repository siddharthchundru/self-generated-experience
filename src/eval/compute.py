"""Transparent operation counts for Stage 2; no unchosen budget is presented as fixed.

Counts main, re-learning, both recovery adaptations, initial/final evaluation,
pilot, gamma sensitivity, adaptation-budget calibration, M0 tuning and capacity
checks. Runtime uses measured CPU rates plus explicit overhead scenarios.
"""

from dataclasses import asdict, dataclass

from episode import EPISODE_STEPS


@dataclass(frozen=True)
class ComputeAssumptions:
    horizon: int = EPISODE_STEPS
    updates_per_env_step: float = 1.0
    warmup_updates: int = 10000
    demo_episodes: int = 140  # 20 per pair, balanced by route; scenario only
    include_expert_topup: bool = True
    pilot_seeds: int = 6
    pilot_relearning: bool = False
    adaptation_fraction: float = 0.10
    sampled_rollouts: int = 100
    greedy_rollouts: int = 20
    main_evaluation_checkpoints: int = 1  # final; shared pretraining counted separately
    tuning_trials_per_condition: int = 6
    tuning_seeds: int = 3
    tuning_budget: int = 99960  # whole 60-step episodes (audit of 30 Oct, B2)
    capacity_updates_per_learner_maze: int = 100000
    capacity_demo_episodes: int = 1840  # replaced by actual all-pairs M1+M2 count in CLI
    capacity_eval_episodes_per_learner_maze: int = 200


def counts(assumptions, budget, seeds):
    a = assumptions
    if a.horizon != EPISODE_STEPS:
        raise ValueError(f"compute horizon must be the project-wide T = {EPISODE_STEPS}")
    if budget <= 0 or budget % a.horizon or seeds <= 0:
        raise ValueError("positive seeds and an episode-aligned budget are required")
    if not 0 <= a.adaptation_fraction <= 1:
        raise ValueError("adaptation fraction must lie in [0, 1]")
    r, T = a.updates_per_env_step, a.horizon
    weighted = 5 if a.include_expert_topup else 4
    main = weighted+2  # C3-plain and C4-plain; no re-learning for these two
    # Round each k DOWN to whole episodes (audit of 30 Oct, B2); report the realised budget.
    k = int(budget*a.adaptation_fraction//T)*T
    unblocked_eval = 7*(a.sampled_rollouts+a.greedy_rollouts)*T
    full_eval = unblocked_eval+6*(a.sampled_rollouts+2*a.greedy_rollouts)*T
    components = []

    def add(name, random_env=0, expert_env=0, updates=0, inference=0, selections=0):
        components.append({"name": name, "random_dynamics_steps": random_env,
                           "expert_steps": expert_env, "gradient_updates": updates,
                           "network_inferences": inference, "frontier_selections": selections})

    def family(label, n, relearn):
        add(label+": D0", expert_env=n*a.demo_episodes*T)
        add(label+": shared weighted/plain pretraining", updates=n*2*a.warmup_updates,
            random_env=n*2*unblocked_eval, inference=n*2*unblocked_eval)
        add(label+": main practice", random_env=n*5*budget,
            expert_env=n*budget if a.include_expert_topup else 0,
            updates=n*main*budget*r, inference=n*4*budget,
            selections=n*2*budget/T)
        add(label+": final/checkpoint evaluation", random_env=n*main*full_eval*a.main_evaluation_checkpoints,
            inference=n*main*full_eval*a.main_evaluation_checkpoints)
        add(label+": P1 and P1-own adaptation", random_env=n*main*2*k,
            inference=n*main*2*k, updates=n*main*2*k*r)
        if relearn:
            add(label+": fresh-buffer re-learning", updates=n*weighted*(a.warmup_updates+budget*r),
                random_env=n*weighted*unblocked_eval, inference=n*weighted*unblocked_eval)

    family("main", seeds, True)
    family("pilot", a.pilot_seeds, a.pilot_relearning)
    # Two alternate gammas x C3/C4; no re-learning or recovery for sensitivity.
    add("pilot: gamma sensitivity", random_env=a.pilot_seeds*4*(budget+unblocked_eval),
        inference=a.pilot_seeds*4*(budget+unblocked_eval),
        updates=a.pilot_seeds*(4*budget*r+2*a.warmup_updates),
        selections=a.pilot_seeds*2*budget/T)
    calibration_k = sum(int(budget*f//T)*T for f in (0.05, 0.10, 0.20))
    add("pilot: additional C0 P1 k-grid calibration", random_env=a.pilot_seeds*(calibration_k+3*6*a.greedy_rollouts*T),
        inference=a.pilot_seeds*(calibration_k+3*6*a.greedy_rollouts*T),
        updates=a.pilot_seeds*calibration_k*r)
    tuning_pairs = a.tuning_trials_per_condition*a.tuning_seeds
    tuning_eval = 5*(a.sampled_rollouts+a.greedy_rollouts)*T
    add("M0: equal C3/C4 tuning", random_env=2*tuning_pairs*(a.tuning_budget+tuning_eval),
        inference=2*tuning_pairs*(a.tuning_budget+tuning_eval),
        updates=tuning_pairs*(2*a.tuning_budget*r+a.warmup_updates),
        selections=tuning_pairs*a.tuning_budget/T,
        expert_env=tuning_pairs*a.demo_episodes*T)
    # Two learners x two mazes, sharing generated expert data per maze.
    capacity_eval = 4*a.capacity_eval_episodes_per_learner_maze*T
    add("capacity: weighted/plain on M1/M2", expert_env=a.capacity_demo_episodes*T,
        updates=4*a.capacity_updates_per_learner_maze,
        random_env=capacity_eval, inference=capacity_eval)
    keys = ("random_dynamics_steps", "expert_steps", "gradient_updates", "network_inferences", "frontier_selections")
    totals = {key: sum(c[key] for c in components) for key in keys}
    totals["environment_steps"] = totals["random_dynamics_steps"]+totals["expert_steps"]
    return {"budget_N": budget, "main_seeds": seeds, "main_conditions": main,
            "relearning_conditions": weighted, "adaptation_k_per_run": k,
            "assumptions": asdict(a), "components": components, "totals": totals}


def estimate_runtime(operation_counts, measurements, conservative=False):
    key = "slowest_per_second" if conservative else "median_per_second"
    simulator = measurements["simulator"]
    network = measurements["network"]
    rates = {
        "random_dynamics_steps": simulator["random_actions"][key],
        "expert_steps": simulator["expert_episodes"][key],
        "gradient_updates": network["synthetic_updates"][key],
        "network_inferences": network["sampled_batch1_inference"][key],
        "frontier_selections": measurements["selector_surrogate"]["selections"][key],
    }
    parts = {name: operation_counts["totals"][name]/rate for name, rate in rates.items()}
    return {"seconds_by_operation": parts, "hours_before_overhead": sum(parts.values())/3600,
            "hours_with_25_percent_overhead": sum(parts.values())/3600*1.25,
            "hours_with_100_percent_overhead": sum(parts.values())/3600*2}
