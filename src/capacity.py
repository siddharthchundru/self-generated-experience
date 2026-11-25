"""The learner-capacity check (spec section 7): can the learner represent the task at all?

What this file does, in one sentence: `run_capacity` trains a fresh learner on
plentiful scripted-expert data covering the whole maze, then measures its
greedy success on M1's task pairs and on random pairs of free cells, against
the agreed pass thresholds.

Why it matters. Every later result assumes the learner (the network, its
9 x 9 action head and the GCSL loss) can solve the maze when it has good data.
If it cannot, a condition's poor result could be a learner failure rather than
an effect of what it practised. The check rules that out first.

What is fixed (spec v0.13, agreed 30 Oct) and what is a planning value:
- Pass thresholds (agreed): greedy success of at least 0.9 on M1's seven task
  pairs, and at least 0.8 on random pairs of free cells, on M1 and on M2.
- Data (code default C11, decision 12 open): one expert episode for every
  ordered pair of distinct free cells, by the shortest route, on the whole
  maze (the held-out region included: this check is about the learner, not
  about generalisation). The expert reaches every such pair within T steps.
- Training (C11): 100,000 updates, batch 256, learning rate 0.001, both
  learners (gamma 0.962 and 1), the same relabelling as everywhere else.
- Evaluation (C11): 200 greedy episodes on random ordered pairs of free cells,
  drawn with fixed seeds identical for every learner; on M1, also 20 greedy
  rollouts per task pair (C7), as S_in does.
- If it fails (decision 15, open; Claude's recommendation is the default):
  train the same learner for as many updates again (100,000 more, 200,000 in
  all) and re-evaluate once; if it still fails, stop before tuning and bring
  it to Siddharth as a design question (network size). Both evaluations are
  reported.

Random streams have tags of their own, so nothing here shares a stream with a
training run.
"""

from dataclasses import asdict, dataclass
import time

import numpy as np

from envs.mazes import free_cells
from envs.pointmaze import MAZES, make_env
from episode import EPISODE_STEPS
from eval.protocol import episode_seed, task_pairs
from eval.suite import policy_view
from expert import run_episode
from gcsl import GCSLLearner, GCSLPolicy, TrajectoryBuffer
from gcsl import GAMMA_PLAIN, GAMMA_WEIGHTED

TAG_DATA = 0xCA1
TAG_INIT = 0xCA2
TAG_BATCH = 0xCA3
TAG_EVAL = 0xCA4
THRESHOLDS = {"task_pairs": 0.9, "random_pairs": 0.8}
MAZES_CHECKED = ("M1", "M2")
ON_FAIL = ("double_once", "stop")


@dataclass(frozen=True)
class CapacitySettings:
    maze: str = "M1"
    learner: str = "weighted"
    seed: int = 0
    updates: int = 100_000
    learning_rate: float = 1e-3
    batch_size: int = 256
    random_pair_episodes: int = 200
    task_pair_rollouts: int = 20
    on_fail: str = "double_once"

    def __post_init__(self):
        if self.maze not in MAZES_CHECKED:
            raise ValueError("the capacity check runs on M1 and M2")
        if self.learner not in ("weighted", "plain"):
            raise ValueError("learner must be 'weighted' or 'plain'")
        if self.on_fail not in ON_FAIL:
            raise ValueError(f"on_fail must be one of {ON_FAIL}")
        if min(self.updates, self.batch_size, self.random_pair_episodes, self.task_pair_rollouts) <= 0:
            raise ValueError("counts must be positive")

    @property
    def gamma(self):
        return GAMMA_WEIGHTED if self.learner == "weighted" else GAMMA_PLAIN


def ordered_pairs(maze):
    """Every ordered pair (start, goal) of distinct free cells, in a fixed order."""
    cells = sorted(free_cells(MAZES[maze]))
    return [(a, b) for a in cells for b in cells if a != b]


def expert_buffer(maze, seed):
    """One expert episode per ordered pair, shortest route; every one must succeed."""
    grid = MAZES[maze]
    env = make_env(maze, allowed_goals=frozenset(free_cells(grid)))
    buffer, failures = TrajectoryBuffer(), []
    try:
        for i, (start, goal) in enumerate(ordered_pairs(maze)):
            episode = run_episode(env, grid, start, goal, "shortest",
                                  seed=episode_seed(seed, TAG_DATA, 0, i))
            if not episode["success"]:
                failures.append((start, goal))
                continue
            buffer.add({"observations": episode["observations"], "actions": episode["actions"],
                        "goal": episode["goal"]}, "capacity_expert")
    finally:
        env.close()
    if failures:
        raise AssertionError(f"the expert failed {len(failures)} pairs on {maze}, e.g. {failures[:3]}")
    return buffer


def _greedy_success(env, policy, start, goal, seed):
    obs, _ = env.reset(seed=seed, options={"reset_cell": list(start), "goal_cell": list(goal)})
    act = policy.start_episode(policy_view(obs), np.random.default_rng(0), sampled=False)
    info, steps = {"success": False}, 0
    for steps in range(1, EPISODE_STEPS + 1):
        obs, _, terminated, truncated, info = env.step(np.asarray(act(policy_view(obs)), dtype=np.float32))
        if terminated or (truncated != (steps == EPISODE_STEPS)):
            raise AssertionError("capacity episode violated the fixed horizon")
    return bool(info["success"])


def evaluate(policy, s):
    """Greedy success on random ordered pairs of free cells (and on M1's task pairs)."""
    grid = MAZES[s.maze]
    pairs = ordered_pairs(s.maze)
    draw = np.random.default_rng(np.random.SeedSequence([TAG_EVAL, 0]))  # the same pairs for every learner
    chosen = [pairs[i] for i in draw.integers(len(pairs), size=s.random_pair_episodes)]
    env = make_env(s.maze, allowed_goals=frozenset(free_cells(grid)))
    try:
        random_hits = [_greedy_success(env, policy, a, b, episode_seed(TAG_EVAL, 1, 0, i))
                       for i, (a, b) in enumerate(chosen)]
        task = {}
        if s.maze == "M1":
            for pair in task_pairs("M1"):
                hits = [_greedy_success(env, policy, pair.start, pair.goal, episode_seed(TAG_EVAL, 2, pair.row, i))
                        for i in range(s.task_pair_rollouts)]
                task[pair.row] = sum(hits) / len(hits)
    finally:
        env.close()
    result = {"random_pairs": sum(random_hits) / len(random_hits), "random_pair_episodes": len(random_hits)}
    if task:
        result["task_pairs"] = sum(task.values()) / len(task)
        result["task_pairs_by_row"] = task
        # Recorded, not used for the verdict: the agreed threshold is read as the mean over the seven pairs
        # (audit of 26 Nov, MI3: whether every pair must also reach 0.9 is open).
        result["task_pairs_worst"] = min(task.values())
    result["passed"] = passes(result)
    return result


def passes(result):
    """The agreed pass rule on one evaluation: random pairs >= 0.8 and, on M1, the mean over the
    seven task pairs >= 0.9."""
    return (result["random_pairs"] >= THRESHOLDS["random_pairs"]
            and result.get("task_pairs", 1.0) >= THRESHOLDS["task_pairs"])


def run_capacity(s, buffer=None):
    """Train and evaluate one learner on one maze. Returns the result dict and the policy."""
    started = time.perf_counter()
    buffer = expert_buffer(s.maze, s.seed) if buffer is None else buffer
    policy = GCSLPolicy(int(np.random.SeedSequence([s.seed, TAG_INIT]).generate_state(1)[0]), name="capacity")
    learner = GCSLLearner(policy, s.gamma, s.learning_rate, s.batch_size)
    rng = np.random.default_rng(np.random.SeedSequence([s.seed, TAG_BATCH]))
    learner.train(buffer, s.updates, rng)
    attempts = [{"updates": learner.gradient_steps, **evaluate(policy, s)}]
    if not attempts[0]["passed"] and s.on_fail == "double_once":
        learner.train(buffer, s.updates, rng)  # the same learner, as many updates again
        attempts.append({"updates": learner.gradient_steps, **evaluate(policy, s)})
    passed = attempts[-1]["passed"]
    return {
        "settings": asdict(s), "gamma": s.gamma, "thresholds": THRESHOLDS,
        "trajectories": len(buffer), "ordered_pairs": len(ordered_pairs(s.maze)),
        "attempts": attempts, "passed": passed,
        "next": "none" if passed else (
            "stop before tuning: bring the learner's capacity to Siddharth as a design question (network size)"),
        "final_digest": policy.parameter_digest(),
        "wall_clock_seconds": time.perf_counter() - started,
    }, policy
