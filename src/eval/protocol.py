"""Concrete ring-task evaluation settings; planning defaults are not registration.

The primary P1 wall is always the middle cell of the shorter arm. Its location
depends on the task geometry, never on a policy's observed performance.
"""

from dataclasses import dataclass

from envs.mazes import RINGS


@dataclass(frozen=True)
class TaskPair:
    maze: str
    row: int

    @property
    def start(self):
        return (self.row, RINGS[self.maze]["west_col"])

    @property
    def goal(self):
        return (self.row, RINGS[self.maze]["east_col"])

    @property
    def shorter(self):
        tie = RINGS[self.maze]["tie_row"]
        return "north" if self.row < tie else "south" if self.row > tie else None

    def wall(self, route):
        if route not in ("north", "south"):
            raise ValueError("a wall requires a north or south route")
        ring = RINGS[self.maze]
        return (ring[f"{route}_row"], ring["midline_col"])


def task_pairs(maze="M1"):
    ring = RINGS[maze]
    return [TaskPair(maze, r) for r in range(ring["north_row"], ring["south_row"] + 1)]


@dataclass(frozen=True)
class EvaluationConfig:
    maze: str = "M1"
    horizon: int = 100  # Stage 1 default, NOT the Stage 2 feasibility/pilot choice.
    sampled_rollouts: int = 100
    greedy_rollouts: int = 20
    minimum_successes: int = 20
    best_of_m: int = 5  # provisional; recorded in every report
    bootstrap_resamples: int = 2000
    seed: int = 20261028
    adaptation_steps: int = 1000  # total across blocked task pairs, not per pair

    def __post_init__(self):
        if self.maze not in RINGS:
            raise ValueError("ring evaluation supports M0 and M1, not M2")
        values = (self.horizon, self.sampled_rollouts, self.greedy_rollouts,
                  self.minimum_successes, self.best_of_m, self.bootstrap_resamples)
        if any(v <= 0 for v in values):
            raise ValueError("counts and horizon must be positive")
        if self.minimum_successes > self.sampled_rollouts:
            raise ValueError("minimum successes exceeds sampled rollouts")
        if self.sampled_rollouts % self.best_of_m:
            raise ValueError("sampled rollouts must form complete best-of-m groups")
        if (self.adaptation_steps < 0
                or self.adaptation_steps % self.horizon):
            raise ValueError("adaptation budget must contain whole episodes")


def episode_seed(base, row, phase, index):
    """Stable phase-separated seeds; independent of policy name and loop order."""
    import numpy as np
    return int(np.random.SeedSequence([base, row, phase, index]).generate_state(1)[0])
