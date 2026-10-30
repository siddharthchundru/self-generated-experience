"""End-to-end evaluation with policy-independent seeds and a suite-owned adaptation loop.

What this file does, in one sentence: `evaluate_policy` runs every evaluation
phase of spec section 8 on one policy (competence, repertoire, zero-shot and
best-of-m recovery, and recovery after the common adaptation procedure), and
checks along the way that the policy only sees what the spec allows and that
the budgets are what the report says they are.

Why it is built this way (independent audit of 28 Oct, findings S1 to S3):
- S1, budgets are counted here, not reported by the adapter. The adapter never
  touches an environment. It asks an `AdaptationSession` for episodes; the
  session picks the blocked task, runs the episode, counts the steps and refuses
  to exceed k. After adaptation the suite checks that exactly k steps were used
  and that the gradient steps equal the configured ratio times k, cross-checked
  against the optimiser's own step counter.
- S2, no old data and no hidden updates. The suite makes the adapter's empty
  buffer and checks it is empty; learned policies must hold parameters only (no
  buffer or optimiser attributes, so a deep copy cannot smuggle old data into
  adaptation); and a hash of the parameters is checked before and after every
  evaluation phase, so evaluation provably never updates a policy.
- S3, observation and goal only. A learned policy's `start_episode` and action
  function receive a dictionary with exactly `observation` (position and
  velocity) and `desired_goal`. Only the Stage 1 stand-ins, marked
  `stand_in = True`, receive the task pair, because the scripted expert needs
  it to plan. The session gives the adapter trajectories, never the task pair
  or the wall cell.

Policy interfaces:
- Stand-in (expert, random): `start_episode(obs, pair, rng, sampled)` -> act(obs).
- Learned: `start_episode(view, rng, sampled)` -> act(view), where view is
  {"observation", "desired_goal"}; plus `parameter_digest()`.
Adapter interface: `updates_per_env_step`, `new_buffer()`,
`adapt(policy, session, buffer) -> AdaptationResult`.
"""

import copy
from dataclasses import asdict, dataclass
import hashlib
import time

import numpy as np

try:  # torch is needed only for learned policies; the Stage 1 stand-ins run without it
    import torch
    from torch.optim.optimizer import register_optimizer_step_post_hook
except ImportError:  # pragma: no cover
    torch = None

from envs.pointmaze import make_env
from routes import trajectory_route
from eval.metrics import best_of_m, mean_defined, summarise_rollouts
from eval.protocol import EvaluationConfig, episode_seed, task_pairs

_SCALARS = (str, int, float, bool, type(None))


def is_stand_in(policy):
    return getattr(policy, "stand_in", False) is True


def policy_view(obs):
    """What a learned policy may see: position and velocity, and the goal (spec section 5)."""
    return {"observation": obs["observation"].copy(), "desired_goal": obs["desired_goal"].copy()}


def check_parameters_only(policy):
    """Refuse a learned policy that holds anything besides scalars and network modules.

    A replay buffer, an optimiser or a list of past trajectories stored on the
    policy would travel with `copy.deepcopy` into adaptation and break the
    empty-buffer rule (audit S2). Network modules are checked one level deep for
    public non-scalar attributes, for the same reason.
    """
    for name, value in vars(policy).items():
        if isinstance(value, _SCALARS):
            continue
        if hasattr(value, "parameters") and hasattr(value, "state_dict") and hasattr(value, "modules"):
            for module in value.modules():
                for key, attr in vars(module).items():
                    if not key.startswith("_") and key != "training" and not isinstance(attr, _SCALARS):
                        raise TypeError(f"policy module attribute {name}.{key} is not a parameter "
                                        "or scalar; learned policies hold parameters only")
            continue
        raise TypeError(f"policy attribute {name!r} ({type(value).__name__}) is not a parameter "
                        "or scalar; learned policies hold parameters only")
    if not callable(getattr(policy, "parameter_digest", None)):
        raise TypeError("a learned policy must provide parameter_digest()")


def _policy_modules(policy):
    return [v for v in vars(policy).values() if hasattr(v, "state_dict") and hasattr(v, "parameters")]


def suite_parameter_digest(policy):
    """Fingerprint of every tensor in the policy's modules, computed here rather than
    trusting the policy's own `parameter_digest()` (audit of 30 Oct, SF3)."""
    digest = hashlib.sha256()
    for index, module in enumerate(_policy_modules(policy)):
        for key, tensor in module.state_dict(keep_vars=False).items():
            digest.update(f"{index}.{key}".encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _parameter_ids(policy):
    return {id(p) for module in _policy_modules(policy) for p in module.parameters()}


class _ParameterGuard:
    """Asserts that evaluation phases leave a learned policy's parameters unchanged."""

    def __init__(self, policy):
        self.policy = policy
        self.digest = None if is_stand_in(policy) else suite_parameter_digest(policy)

    def check(self, phase):
        if self.digest is not None and suite_parameter_digest(self.policy) != self.digest:
            raise AssertionError(f"policy parameters changed during evaluation phase {phase}")


def rollout(env, policy, pair, seed, sampled, horizon, keep_trajectory=False):
    obs, _ = env.reset(seed=seed, options={"reset_cell": list(pair.start), "goal_cell": list(pair.goal)})
    goal = obs["desired_goal"].copy()
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    if is_stand_in(policy):
        view = lambda o: o
        act = policy.start_episode(obs, pair, rng, sampled)
    else:
        view = policy_view
        act = policy.start_episode(view(obs), rng, sampled)
    observations = [obs["observation"].copy()]
    actions = []
    first_success = None
    rewards = 0.0
    for step in range(1, horizon + 1):
        action = np.asarray(act(view(obs)), dtype=np.float32)
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs["observation"].copy())
        rewards += float(reward)
        if info["success"] and first_success is None:
            first_success = step
        if not np.array_equal(obs["desired_goal"], goal):
            raise AssertionError("goal moved during evaluation")
        if terminated or (truncated != (step == horizon)):
            raise AssertionError("evaluation episode violated the fixed horizon")
    positions = [o[:2] for o in observations]
    route = trajectory_route(positions, pair.maze)
    if info["success"] and route == "neither":
        raise AssertionError("successful west-east rollout has no crossing")
    record = {
        "seed": seed, "row": pair.row, "sampled": sampled, "route": route,
        "success": bool(info["success"]), "first_success_step": first_success,
        "steps": horizon, "reward_sum": rewards,
        "final_goal_distance": float(np.linalg.norm(positions[-1]-goal)),
        "path_length": float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
    }
    if not keep_trajectory:
        return record
    # What a learner may store: the states it saw, the actions it took and the
    # goal it was given. No task pair, no wall, no route label.
    trajectory = {"observations": np.array(observations), "actions": np.array(actions), "goal": goal}
    return record, trajectory


def collect(policy, pair, count, config, phase, sampled, wall=None, index_offset=0):
    blocked = frozenset({wall}) if wall is not None else frozenset()
    env = make_env(pair.maze, max_episode_steps=config.horizon, blocked_cells=blocked)
    try:
        return [rollout(env, policy, pair, episode_seed(config.seed, pair.row, phase, i+index_offset),
                        sampled, config.horizon) for i in range(count)]
    finally:
        env.close()


class BudgetExhausted(RuntimeError):
    """Raised when an adapter asks for an episode beyond the adaptation budget k."""


class AdaptationSession:
    """The only route by which an adapter interacts with the blocked environments.

    Each `run_episode()` draws one blocked task uniformly at random (C3-style
    practice on the blocked pairs, spec section 8), runs one full episode of the
    policy being adapted with actions sampled at temperature 1, counts its steps
    and returns the trajectory. The task pair and wall stay inside the session.
    The policy is fixed at construction: the adapter cannot run a different one.
    """

    def __init__(self, policy, tasks, config, phase):
        self._policy = policy
        self._tasks = list(tasks)
        self._config = config
        self._phase = phase
        self._rng = np.random.default_rng(episode_seed(config.seed, 0, phase, 0))
        self._envs = {}
        self._records = []
        self.budget = config.adaptation_steps
        self.episode_steps = config.horizon
        self.phase = phase
        self._steps_used = 0

    @property
    def steps_used(self):
        """Read-only: only `run_episode` advances it (audit of 30 Oct, SF3)."""
        return self._steps_used

    @property
    def remaining_steps(self):
        return self.budget - self.steps_used

    @property
    def episodes_run(self):
        return len(self._records)

    def run_episode(self):
        horizon = self._config.horizon
        if self.steps_used + horizon > self.budget:
            raise BudgetExhausted(f"adaptation budget of {self.budget} steps is used up")
        pair, wall = self._tasks[int(self._rng.integers(len(self._tasks)))]
        key = (pair, wall)
        if key not in self._envs:
            self._envs[key] = make_env(pair.maze, max_episode_steps=horizon, blocked_cells=frozenset({wall}))
        index = len(self._records)
        seed = episode_seed(self._config.seed, pair.row, self._phase, index)
        record, trajectory = rollout(self._envs[key], self._policy, pair, seed, True, horizon,
                                     keep_trajectory=True)
        self._steps_used += horizon
        self._records.append({**record, "wall": list(wall)})
        return trajectory

    def close(self):
        for env in self._envs.values():
            env.close()
        self._envs.clear()


@dataclass
class AdaptationResult:
    policy: object
    gradient_steps: int
    mode: str  # learned adapters must not use the "standin_" prefix
    optimizer: object = None  # a torch optimiser, for the independent step count


class NoUpdateAdapter:
    """Stage 1 stand-in: collects the exact budget, performs ZERO parameter updates.

    Not the Stage 2 common adaptation procedure; reports must not present it as
    learned recovery.
    """

    updates_per_env_step = 0.0

    def new_buffer(self):
        return []

    def adapt(self, policy, session, buffer):
        while session.remaining_steps >= session.episode_steps:
            buffer.append(session.run_episode())
        return AdaptationResult(policy, 0, "standin_no_parameter_updates")


no_update_adaptation = NoUpdateAdapter()


def optimizer_step_count(optimizer):
    """Steps taken, read from the optimiser's own per-parameter state (Adam keeps one)."""
    steps = {int(state["step"]) for state in optimizer.state.values() if "step" in state}
    if not steps:
        return 0
    if len(steps) != 1:
        raise AssertionError(f"optimiser parameters disagree on step count: {sorted(steps)}")
    return steps.pop()


def _adapt(policy, tasks, config, adapter, phase):
    # One independent adapted copy per perturbation, trained across its blocked pairs.
    clone = copy.deepcopy(policy)
    if not is_stand_in(clone):
        check_parameters_only(clone)
    buffer = adapter.new_buffer()
    if len(buffer) != 0:
        raise AssertionError("adaptation must start from an empty buffer")
    session = AdaptationSession(clone, tasks, config, phase)
    # Count every optimiser step taken anywhere during adaptation, including on an
    # optimiser the adapter does not return (audit of 30 Oct, SF3).
    counted = [0]
    handle = None
    if torch is not None:
        handle = register_optimizer_step_post_hook(
            lambda *_: counted.__setitem__(0, counted[0] + 1))
    try:
        result = adapter.adapt(clone, session, buffer)
    finally:
        session.close()
        if handle is not None:
            handle.remove()
    if result.policy is not clone:
        raise AssertionError("adapter must return the copy it was given")
    if session.steps_used != config.adaptation_steps:
        raise AssertionError(f"adapter used {session.steps_used} of {config.adaptation_steps} "
                             "adaptation steps; the budget must be used exactly")
    if result.mode.startswith("standin_"):
        if (type(adapter) is not NoUpdateAdapter or result.gradient_steps != 0
                or result.optimizer is not None or counted[0] != 0):
            raise AssertionError("only the no-update stand-in may report a standin_ mode")
        optimizer_steps = 0
    else:
        if adapter.updates_per_env_step != config.adaptation_updates_per_env_step:
            raise AssertionError("adapter's update ratio differs from the configured training ratio")
        expected = round(config.adaptation_updates_per_env_step * config.adaptation_steps)
        if result.gradient_steps != expected:
            raise AssertionError(f"adapter reported {result.gradient_steps} gradient steps; "
                                 f"ratio x k requires {expected}")
        if result.optimizer is None:
            raise AssertionError("a learned adapter must return its optimiser for the step check")
        optimized = {id(p) for group in result.optimizer.param_groups for p in group["params"]}
        if optimized != _parameter_ids(clone):
            raise AssertionError("the returned optimiser does not hold exactly the adapted policy's parameters")
        optimizer_steps = optimizer_step_count(result.optimizer)
        if optimizer_steps != result.gradient_steps or counted[0] != result.gradient_steps:
            raise AssertionError(f"optimiser took {optimizer_steps} steps ({counted[0]} counted across all "
                                 f"optimisers), adapter reported {result.gradient_steps}")
    return result, {
        "status": result.mode,
        "environment_steps": session.steps_used,
        "episodes": session.episodes_run,
        "gradient_steps": result.gradient_steps,
        "optimizer_steps": optimizer_steps,
        "updates_per_env_step": 0.0 if result.mode.startswith("standin_") else config.adaptation_updates_per_env_step,
        "buffer_size_at_start": 0,
        "buffer_size_at_end": len(buffer),
    }, session._records


def evaluate_policy(policy, config=EvaluationConfig(), adapter=None):
    started = time.perf_counter()
    if not is_stand_in(policy):
        check_parameters_only(policy)
    guard = _ParameterGuard(policy)
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
    guard.check("unblocked and zero-shot")

    primary = [r for r in pairs if r["is_primary_pair"]]
    adaptation_runs = {}
    by_row = {pair.row: pair for pair in task_pairs(config.maze)}
    for name, phase in (("P1", 1000), ("P1_own", 2000)):
        eligible = [r for r in primary if r[name].get("wall") is not None]
        tasks = [(by_row[r["row"]], tuple(r[name]["wall"])) for r in eligible]
        result = None
        if adapter is not None and tasks:
            result, run, collected = _adapt(policy, tasks, config, adapter, phase)
            # One shared adaptation run, not a separate budget for every pair.
            all_records.extend({**record, "phase": f"{name}_adaptation_collection"} for record in collected)
        else:
            run = {"status": "not_run_no_adapter_or_eligible_pairs", "environment_steps": 0,
                   "gradient_steps": 0}
        adaptation_runs[name] = {**run, "task_rows": [pair.row for pair, _ in tasks]}
        guard.check(f"{name} adaptation (original policy must be untouched)")
        adapted_guard = _ParameterGuard(result.policy) if result else None
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
        if adapted_guard:
            adapted_guard.check(f"{name} post-adaptation evaluation")
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
        "schema_version": 2, "policy": policy.name,
        "policy_kind": "stand_in_with_task_pair" if is_stand_in(policy) else "learned_observation_and_goal_only",
        "parameter_digest": guard.digest, "config": asdict(config),
        "protocol_status": "planning_defaults_not_preregistered",
        "temperature": 1, "aggregate": aggregate, "pairs": pairs, "rollouts": all_records,
        "adaptation_runs": adaptation_runs,
        "environment_steps": sum(r["steps"] for r in all_records),
        "elapsed_seconds": time.perf_counter()-started,
        "limitations": [
            "Stand-in policies receive the task pair; learned policies receive observation and goal only.",
            "Random policy has no greedy action: its ordinary random actions fill greedy evaluation slots.",
            "P1-own counts all greedy route crossings; ties choose the shorter route; no crossings is undefined.",
            "Best-of-m uses disjoint groups of zero-shot attempts, sharing data with A_0.",
            "M2, training exposure diagnostics, S_out and P2/P3 are outside this ring-task suite.",
        ],
    }
