"""The training runner: warm-up on demonstrations, then practice to a budget.

What this file does, in one sentence: `run_training(settings)` builds D0,
warms the learner up on it, resets Adam, runs one condition's practice for a
whole-episode budget N, logs every environment step and gradient step for the
budget audit, and returns the pretrained and final policies with the logs.

Why it matters: every condition goes through this one function, so they share
everything except the goal rule (spec sections 5 to 7):
- Warm-up (spec v0.13). The learner trains on D0 alone, producing the
  pretrained policy. It depends only on (maze, seed, learner, learning rate,
  batch size, warm-up rule), never on the condition, so every condition in a
  seed starts from the same pretrained policy; the summary records its
  fingerprint so the budget audit can check that across processes.
- Adam reset (spec v0.12). A new learner, and so a new Adam with zero running
  averages and step count, is built when the updates after warm-up begin; only
  the weights carry over. Demonstrations only resets at the same point. An
  optional linear learning-rate warm-up of fixed length (`lr_warmup_steps`) is
  the pre-agreed fallback, off by default.
- Practice. Collecting conditions gather N / T episodes and take
  `updates_per_env_step` x T gradient steps per episode. Demonstrations only
  takes the same total, N x updates_per_env_step, on D0 alone, in chunks of
  the same size so its log lines up with the others. Batches for practice come
  from one random stream per seed, the same in every condition.
- Budget audit. Environment steps are counted by the collector's episodes and
  checked against N; gradient steps are counted by the learner, by Adam's own
  step counter and by a process-wide hook on every optimiser step, and all
  three must agree with warm-up plus N x ratio. Any mismatch raises.

Undecided settings, each a parameter with its current default (audit of 4 Nov,
"Decisions for you"; also listed in the PR and spec v0.14):
- `warmup_rule`: "fixed" (10,000 updates, agreed 30 Oct) or "validation",
  decision 8's rule (stop when held-out demonstration loss stops improving).
- warm-up uses the practice learning rate (decision 9).
- `schedule`: "episode_then_updates" (collect an episode, then take its
  updates; the adapter's schedule) or "interleaved" (updates after every step).
- recovery adaptation starts a fresh optimiser (`adaptation_fresh_optimizer`;
  only True is implemented, because the evaluation suite's optimiser-step check
  assumes it), and whether the learning-rate fallback also applies there
  (`lr_warmup_in_adaptation`, decision 14).
- k (`adaptation_steps`), placeholder 960; refused for pilot and main runs.
- start distribution, random actions' rule and expert top-up's route split:
  see `collection.py`.
- frontier novelty's position normalisation: see `frontier.py`.
"""

from dataclasses import asdict, dataclass, fields
import time

import numpy as np
import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

from collection import (COLLECTING, CONDITIONS, EXPERT, FRONTIER, PLAIN_CONDITIONS, RANDOM,
                        RANDOM_ACTION_RULES, START_DISTRIBUTIONS, TOPUP_ROUTE_RULES, Collector,
                        fork_cells, held_out_cells)
from data_diversity import DataDiversityLog
from demonstrations import DEMOS_PER_PAIR, make_demonstrations
from envs.mazes import RINGS
from envs.pointmaze import xy_to_cell
from episode import EPISODE_STEPS, whole_episodes
from eval.protocol import EvaluationConfig
from frontier import BANDWIDTH, CANDIDATES, DENSITY_FIT_POINTS, FrontierSelector
from gcsl import (GAMMA_PLAIN, GAMMA_WEIGHTED, POSITION_SCALE, GCSLAdapter, GCSLLearner, GCSLPolicy,
                  TrajectoryBuffer, updates_per_episode, weighted_loss)
from routes import trajectory_route

TAG_WARMUP = 0xA1
TAG_PRACTICE = 0xA2
TAG_VALIDATION = 0xA3       # validation demonstrations for warmup_rule="validation"
TAG_VALIDATION_BATCH = 0xA4
TAG_INIT = 0xA5

PILOT_BUDGET = 249_960        # spec v0.13: 4,166 episodes of 60 steps
CAP_BUDGET = 999_960          # 16,666 episodes
K_PLACEHOLDER = 16 * EPISODE_STEPS  # 960; k is set by the spec section 8 rule in the pilot
PURPOSES = ("smoke", "tuning", "capacity", "precondition", "pilot", "main")
SCHEDULES = ("episode_then_updates", "interleaved")
WARMUP_RULES = ("fixed", "validation")


@dataclass(frozen=True)
class TrainingSettings:
    """Everything that defines one run: one seed, one condition, one learner, one maze."""

    condition: str
    seed: int = 0
    maze: str = "M1"
    learner: str = "weighted"            # "weighted" (gamma 0.962) or "plain" (gamma 1)
    purpose: str = "smoke"
    # Agreed 30 Oct (spec v0.13).
    batch_size: int = 256
    demos_per_pair: int = DEMOS_PER_PAIR  # 20: 140 on M1, 100 on M0
    budget_steps: int = PILOT_BUDGET      # N, whole episodes
    updates_per_env_step: float = 1.0     # starting value; tuned on M0 from {0.5, 1, 2}
    learning_rate: float = 1e-3           # tuned on M0 from {0.0003, 0.001}
    # Warm-up (decisions 8 and 9 open).
    warmup_rule: str = "fixed"
    warmup_updates: int = 10_000          # fixed rule
    validation_per_pair: int = 4          # validation rule: 2 per route per pair
    validation_examples: int = 4096
    validation_check_every: int = 1000
    validation_window: int = 5000
    validation_min_improvement: float = 0.01
    warmup_min_updates: int = 10_000
    warmup_max_updates: int = 50_000
    # Optimiser after warm-up (spec v0.12): reset; learning-rate warm-up off by default.
    lr_warmup_steps: int = 0
    # Practice schedule (open).
    schedule: str = "episode_then_updates"
    # Collection (decisions 1, 4, 5 open).
    start_distribution: str = "task_pair_starts"
    random_actions: str = "continuous"
    topup_routes: str = "alternate"
    # Frontier novelty (spec section 6; normalisation open).
    frontier_candidates: int = CANDIDATES
    frontier_bandwidth: float = BANDWIDTH
    frontier_fit_points: int = DENSITY_FIT_POINTS
    frontier_position_scale: float = POSITION_SCALE
    # Recovery adaptation, for the evaluation that follows a run (SF4; k and decision 14 open).
    adaptation_steps: int = K_PLACEHOLDER
    adaptation_fresh_optimizer: bool = True
    lr_warmup_in_adaptation: bool = False

    def __post_init__(self):
        if self.condition not in CONDITIONS:
            raise ValueError(f"unknown condition {self.condition!r}")
        if self.learner not in ("weighted", "plain"):
            raise ValueError("learner must be 'weighted' or 'plain'")
        if self.learner == "plain" and self.condition not in PLAIN_CONDITIONS:
            raise ValueError("the plain-learner arm has task-directed practice and frontier novelty only")
        if self.maze not in RINGS:
            raise ValueError("training runs on M0 or M1 (the ring mazes)")
        if self.purpose not in PURPOSES:
            raise ValueError(f"unknown purpose {self.purpose!r}")
        whole_episodes(self.budget_steps, "practice budget N")
        whole_episodes(self.adaptation_steps, "adaptation budget k")
        updates_per_episode(self.updates_per_env_step)
        if self.schedule not in SCHEDULES or self.warmup_rule not in WARMUP_RULES:
            raise ValueError("unknown schedule or warm-up rule")
        if (self.start_distribution not in START_DISTRIBUTIONS or self.random_actions not in RANDOM_ACTION_RULES
                or self.topup_routes not in TOPUP_ROUTE_RULES):
            raise ValueError("unknown collection rule")
        if not self.adaptation_fresh_optimizer:
            raise NotImplementedError("carrying the practice optimiser into adaptation is not implemented: "
                                      "the evaluation suite checks that the adapter's optimiser took exactly "
                                      "ratio x k steps")
        if self.purpose in ("pilot", "main") and self.adaptation_steps == K_PLACEHOLDER:
            raise ValueError("k = 960 is a placeholder; set k by the spec section 8 rule before the pilot")
        if self.warmup_updates < 0 or self.lr_warmup_steps < 0 or self.batch_size <= 0:
            raise ValueError("counts must be non-negative")
        if self.warmup_rule == "validation" and not (
                0 < self.warmup_min_updates <= self.warmup_max_updates
                and self.validation_check_every > 0 and self.validation_window > 0
                and self.validation_window % self.validation_check_every == 0):
            raise ValueError("inconsistent validation warm-up settings")

    @property
    def gamma(self):
        return GAMMA_WEIGHTED if self.learner == "weighted" else GAMMA_PLAIN

    @property
    def practice_episodes(self):
        return whole_episodes(self.budget_steps, "practice budget N")

    @property
    def practice_updates(self):
        return self.practice_episodes * updates_per_episode(self.updates_per_env_step)

    def warmup_key(self):
        """The settings warm-up depends on; equal keys give the same pretrained policy."""
        names = ("seed", "maze", "learner", "batch_size", "demos_per_pair", "learning_rate", "warmup_rule",
                 "warmup_updates", "validation_per_pair", "validation_examples", "validation_check_every",
                 "validation_window", "validation_min_improvement", "warmup_min_updates",
                 "warmup_max_updates")
        return {name: getattr(self, name) for name in names}


def adapter_for(settings):
    """The common adaptation procedure with this run's frozen settings (audit SF4)."""
    return GCSLAdapter(settings.gamma, settings.learning_rate, settings.updates_per_env_step,
                       settings.batch_size, seed=settings.seed,
                       lr_warmup_steps=settings.lr_warmup_steps if settings.lr_warmup_in_adaptation else 0)


def evaluation_config_for(settings):
    """Evaluation settings whose adaptation budget and ratio match training (audit SF4)."""
    return EvaluationConfig(maze=settings.maze, adaptation_steps=settings.adaptation_steps,
                            adaptation_updates_per_env_step=settings.updates_per_env_step)


# --- Demonstrations ------------------------------------------------------------------

def build_demonstrations(maze, per_pair, seed, seed_tag=None):
    """D0 (or the validation set), checked: every demonstration succeeds, drives the
    route it was told to, and never enters the held-out region (audit MI6)."""
    kwargs = {} if seed_tag is None else {"seed_tag": seed_tag}
    demonstrations = make_demonstrations(maze, per_pair, seed, **kwargs)
    held_out = held_out_cells(maze)
    grid = RINGS[maze]["grid"]
    for d in demonstrations:
        if not d["success"]:
            raise AssertionError(f"a demonstration on row {d['row']} ({d['route']}) failed")
        if trajectory_route(d["observations"][:, :2], maze) != d["route"]:
            raise AssertionError(f"a demonstration on row {d['row']} did not drive the {d['route']} route")
        if any(xy_to_cell(grid, p) in held_out for p in d["observations"][:, :2]):
            raise AssertionError("a demonstration entered the held-out region")
    return demonstrations


def _buffer_of(trajectories, source):
    buffer = TrajectoryBuffer()
    for trajectory in trajectories:
        buffer.add(trajectory, source)
    return buffer


# --- Exposure logging ------------------------------------------------------------------

class _Exposure:
    """Per-chunk averages of the batches' effective sample size, loss, gap and
    held-out goal share (spec section 8, Exposure row)."""

    def __init__(self, maze):
        grid = RINGS[maze]["grid"]
        self.n_rows, self.n_cols = len(grid), len(grid[0])
        self.held_out = np.zeros((self.n_rows, self.n_cols), dtype=bool)
        for r, c in held_out_cells(maze):
            self.held_out[r, c] = True
        self.reset()

    def reset(self):
        self.rows = []

    def add(self, stats, batch):
        goals = batch["goal"]
        r = np.clip(np.floor(self.n_rows / 2 - goals[:, 1]).astype(int), 0, self.n_rows - 1)
        c = np.clip(np.floor(goals[:, 0] + self.n_cols / 2).astype(int), 0, self.n_cols - 1)
        in_held_out = self.held_out[r, c]
        weights = batch["weight"]
        self.rows.append((stats["loss"], stats["effective_sample_size"], stats["mean_gap"],
                          float(in_held_out.mean()), float(weights[in_held_out].sum() / weights.sum())))

    def summary(self):
        if not self.rows:
            return {"updates": 0}
        a = np.array(self.rows)
        return {"updates": len(self.rows), "loss": float(a[:, 0].mean()),
                "effective_sample_size": float(a[:, 1].mean()), "mean_gap": float(a[:, 2].mean()),
                "held_out_goal_share": float(a[:, 3].mean()), "held_out_weight_share": float(a[:, 4].mean())}


def _step(learner, buffer, rng, exposure):
    stats = learner.update(buffer, rng)
    exposure.add(stats, learner.last_batch)


# --- Warm-up -------------------------------------------------------------------------

def validation_loss(policy, batch):
    with torch.no_grad():
        logits = policy.network(torch.from_numpy(batch["observation"]), torch.from_numpy(batch["goal"]))
        return float(weighted_loss(logits, torch.from_numpy(batch["action"]),
                                   torch.from_numpy(batch["weight"]).float()))


def warm_up(settings, d0_buffer):
    """Train a seed-initialised policy on D0 alone. Returns (policy, log).

    The warm-up learner (and its Adam) is discarded afterwards: practice starts
    a new one (spec v0.12).
    """
    policy = GCSLPolicy(int(np.random.SeedSequence([settings.seed, TAG_INIT]).generate_state(1)[0]))
    learner = GCSLLearner(policy, settings.gamma, settings.learning_rate, settings.batch_size)
    rng = np.random.default_rng(np.random.SeedSequence([settings.seed, TAG_WARMUP]))
    exposure = _Exposure(settings.maze)
    log = {"rule": settings.warmup_rule, "checks": []}
    if settings.warmup_rule == "fixed":
        for _ in range(settings.warmup_updates):
            _step(learner, d0_buffer, rng, exposure)
    else:
        validation = build_demonstrations(settings.maze, settings.validation_per_pair, settings.seed,
                                          seed_tag=TAG_VALIDATION)
        batch = _buffer_of(validation, "validation").sample(
            settings.validation_examples,
            np.random.default_rng(np.random.SeedSequence([settings.seed, TAG_VALIDATION_BATCH])),
            settings.gamma)
        lag = settings.validation_window // settings.validation_check_every
        while learner.gradient_steps < settings.warmup_max_updates:
            for _ in range(settings.validation_check_every):
                _step(learner, d0_buffer, rng, exposure)
            log["checks"].append({"updates": learner.gradient_steps, "loss": validation_loss(policy, batch)})
            if learner.gradient_steps < settings.warmup_min_updates or len(log["checks"]) <= lag:
                continue
            losses = [c["loss"] for c in log["checks"]]
            best_before, best_now = min(losses[:-lag]), min(losses)
            if best_now > (1 - settings.validation_min_improvement) * best_before:
                break
    log.update({"updates": learner.gradient_steps, "optimizer_steps": _adam_steps(learner.optimizer),
                **{f"batch_{k}": v for k, v in exposure.summary().items() if k != "updates"}})
    return policy, log


def _adam_steps(optimizer):
    steps = {int(s["step"]) for s in optimizer.state.values() if "step" in s}
    if len(steps) > 1:
        raise AssertionError(f"Adam's parameters disagree on step count: {sorted(steps)}")
    return steps.pop() if steps else 0


# --- Practice ------------------------------------------------------------------------

def run_training(settings, on_episode=None):
    """One run: D0, warm-up, Adam reset, practice for N. See the module docstring.

    `on_episode(row)` receives each practice log row as it is produced (the
    script writes them to disk). Returns a dict with the pretrained and final
    policies' weights, the summary and the episode log.
    """
    started = time.perf_counter()
    counted = [0]
    handle = register_optimizer_step_post_hook(lambda *_: counted.__setitem__(0, counted[0] + 1))
    try:
        return _run(settings, on_episode, started, counted)
    finally:
        handle.remove()


def _run(settings, on_episode, started, counted):
    s = settings
    demonstrations = build_demonstrations(s.maze, s.demos_per_pair, s.seed)
    buffer = _buffer_of(demonstrations, "D0")
    diversity = DataDiversityLog(s.maze)
    diversity.add_demonstrations([d["observations"] for d in demonstrations])

    policy, warmup_log = warm_up(s, buffer)
    if counted[0] != warmup_log["updates"] or warmup_log["optimizer_steps"] != warmup_log["updates"]:
        raise AssertionError("warm-up took a different number of optimiser steps than it counted")
    pretrained_state = {k: v.clone() for k, v in policy.network.state_dict().items()}
    pretrained_digest = policy.parameter_digest()
    warmup_seconds = time.perf_counter() - started

    # Adam reset (spec v0.12): a new learner, so a new optimiser; weights carry over.
    learner = GCSLLearner(policy, s.gamma, s.learning_rate, s.batch_size, s.lr_warmup_steps)
    rng = np.random.default_rng(np.random.SeedSequence([s.seed, TAG_PRACTICE]))
    per_episode = updates_per_episode(s.updates_per_env_step)
    exposure = _Exposure(s.maze)
    forks = fork_cells(s.maze)
    grid = RINGS[s.maze]["grid"]
    held_out = held_out_cells(s.maze)

    selector = None
    if s.condition == FRONTIER:
        selector = FrontierSelector(s.frontier_candidates, s.frontier_bandwidth, s.frontier_fit_points,
                                    s.frontier_position_scale)
        for d in demonstrations:
            selector.add_positions(d["observations"])
    collector = None
    if s.condition in COLLECTING:
        acting = policy if s.condition not in (EXPERT, RANDOM) else None
        collector = Collector(s.condition, s.maze, s.seed, policy=acting, selector=selector,
                              start_distribution=s.start_distribution, random_actions=s.random_actions,
                              topup_routes=s.topup_routes)

    log, env_steps, collected_goal_cells_held_out = [], 0, 0
    try:
        for index in range(s.practice_episodes):
            exposure.reset()
            row = {"episode": index}
            if collector is None:  # demonstrations only: the same updates, on D0 alone
                for _ in range(per_episode):
                    _step(learner, buffer, rng, exposure)
            else:
                due = _interleaved_due(s.updates_per_env_step) if s.schedule == "interleaved" else None
                after_step = None
                if due is not None:
                    after_step = lambda step: [_step(learner, buffer, rng, exposure) for _ in range(due[step])]
                trajectory, record = collector.run_episode(index, after_step)
                env_steps += len(trajectory["actions"])
                buffer.add(trajectory, s.condition)
                positions = trajectory["observations"][:, :2]
                if selector is not None:
                    selector.add_positions(positions)
                row.update(record)
                row.update(diversity.add_episode(positions))
                cells = [tuple(c) for c in _cells(grid, positions)]
                row["fork_visits"] = {str(r): sum(c == cell for c in cells) for r, cell in forks.items()}
                row["held_out_states"] = sum(c in held_out for c in cells)
                collected_goal_cells_held_out += int(record["goal_in_held_out"])
                if due is None:
                    for _ in range(per_episode):
                        _step(learner, buffer, rng, exposure)
            row.update({"environment_steps": env_steps, "gradient_steps": learner.gradient_steps,
                        "buffer_trajectories": len(buffer), "learning_rate": learner.current_learning_rate(),
                        "batches": exposure.summary()})
            if row["batches"]["updates"] != per_episode:
                raise AssertionError(f"episode {index} took {row['batches']['updates']} updates, "
                                     f"not {per_episode}")
            log.append(row)
            if on_episode is not None:
                on_episode(row)
    finally:
        if collector is not None:
            collector.close()

    budget = _budget_audit(s, env_steps, learner, warmup_log, counted[0], len(buffer), len(demonstrations))
    summary = {
        "settings": asdict(s), "gamma": s.gamma,
        "demonstrations": len(demonstrations),
        "pretrained_digest": pretrained_digest, "final_digest": policy.parameter_digest(),
        "warmup": warmup_log, "budget": budget,
        "commanded_held_out_goals": collected_goal_cells_held_out,
        "data_diversity": diversity.summary(),
        "fork_visits_total": {str(r): sum(row.get("fork_visits", {}).get(str(r), 0) for row in log)
                              for r in forks},
        "wall_clock_seconds": {"warmup": warmup_seconds, "total": time.perf_counter() - started},
    }
    return {"summary": summary, "episodes": log, "pretrained_state": pretrained_state,
            "final_state": {k: v.clone() for k, v in policy.network.state_dict().items()}}


def _interleaved_due(ratio):
    """Updates after each step 1..T so that step s has taken floor(s x ratio) in total."""
    return {step: int(np.floor(step * ratio + 1e-9)) - int(np.floor((step - 1) * ratio + 1e-9))
            for step in range(1, EPISODE_STEPS + 1)}


def _cells(grid, positions):
    n_rows, n_cols = len(grid), len(grid[0])
    rows = np.floor(n_rows / 2 - positions[:, 1]).astype(int)
    cols = np.floor(positions[:, 0] + n_cols / 2).astype(int)
    return list(zip(rows.tolist(), cols.tolist()))


def _budget_audit(s, env_steps, learner, warmup_log, all_optimizer_steps, buffer_size, demos):
    """Check environment and gradient steps against the budget; raise on any mismatch."""
    expected_env = s.budget_steps if s.condition in COLLECTING else 0
    expected_practice_updates = s.practice_updates
    audit = {
        "budget_steps_N": s.budget_steps,
        "practice_episodes": s.practice_episodes if s.condition in COLLECTING else 0,
        "practice_environment_steps": env_steps,
        "expected_practice_environment_steps": expected_env,
        "warmup_gradient_steps": warmup_log["updates"],
        "practice_gradient_steps": learner.gradient_steps,
        "practice_optimizer_steps": _adam_steps(learner.optimizer),
        "expected_practice_gradient_steps": expected_practice_updates,
        "all_optimizer_steps_in_process": all_optimizer_steps,
        "updates_per_env_step": s.updates_per_env_step,
        "buffer_trajectories": buffer_size,
        "expected_buffer_trajectories": demos + (s.practice_episodes if s.condition in COLLECTING else 0),
    }
    problems = []
    if env_steps != expected_env:
        problems.append("environment steps differ from N")
    if not (learner.gradient_steps == audit["practice_optimizer_steps"] == expected_practice_updates):
        problems.append("practice gradient steps differ from N x ratio")
    if all_optimizer_steps != warmup_log["updates"] + expected_practice_updates:
        problems.append("optimiser steps were taken outside warm-up and practice")
    if buffer_size != audit["expected_buffer_trajectories"]:
        problems.append("the buffer holds a different number of trajectories than collected")
    if problems:
        raise AssertionError("budget audit failed: " + "; ".join(problems) + f" ({audit})")
    audit["passed"] = True
    return audit


def settings_from_dict(values):
    names = {f.name for f in fields(TrainingSettings)}
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")
    return TrainingSettings(**values)
