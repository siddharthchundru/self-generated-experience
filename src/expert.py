"""The scripted expert that produces the demonstrations (D0).

What this file does, in one sentence: given a start cell, a goal cell and a
route ("north" or "south" on M1, or "shortest"), it plans the cell sequence with
the grid planner in `mazes.py` and drives the ball along it with a simple
proportional-derivative (PD) controller.

Why it matters: the spec (section 6) requires the demonstrations to show *both*
routes for every two-route pair, in fixed equal counts, because mode collapse
presupposes that both modes were present to begin with. So the expert must be
able to take the route it is told to take, reliably. The Stage 1 gate is that it
reaches about 100% success on M1 on both routes.

How the controller works:
- Waypoints are the centres of the planned cells after the start cell; the last
  waypoint is the exact goal position (not its cell centre, since the goal is
  placed with noise inside its cell).
- Each step the ball is pushed toward the current waypoint with force
  KP * (waypoint - position) - KD * velocity, clipped to the action range [-1, 1].
  The KD term damps the approach so the ball does not overshoot into walls.
- The expert moves on to the next waypoint once it is within SWITCH_RADIUS of the
  current one. On the final waypoint it keeps holding position, because episodes
  have a fixed length and success is judged at the final step.

The expert never sees the learner and never uses randomness of its own; the only
variation between its demonstrations comes from the start and goal noise.
"""

import numpy as np

from envs.mazes import NORTH_ARM, SOUTH_ARM, shortest_path
from envs.pointmaze import cell_to_xy

# Controller settings. Chosen by hand on M1 so the expert succeeds on every pair
# and both routes (see tests/test_expert.py); they are not tuned on any result.
KP = 10.0
KD = 1.5
SWITCH_RADIUS = 0.4

ROUTE_BLOCKS = {
    # To force a route, the planner treats the *other* arm as blocked.
    "north": SOUTH_ARM,
    "south": NORTH_ARM,
    "shortest": frozenset(),
}


def plan_cells(grid, start, goal, route="shortest", blocked=frozenset()):
    """Cell sequence from start to goal along `route`, also avoiding `blocked`
    (for example the P1 wall cell). Raises if no such route exists."""
    avoid = frozenset(ROUTE_BLOCKS[route]) | frozenset(blocked)
    path = shortest_path(grid, tuple(start), tuple(goal), blocked=avoid)
    if path is None:
        raise ValueError(f"no {route} route from {start} to {goal} avoiding {sorted(blocked)}")
    return path


class ScriptedExpert:
    """Follows a planned cell path with a PD controller.

    Usage, once per episode:
        expert = ScriptedExpert(grid, start_cell, goal_cell, goal_xy, route="north")
        action = expert.act(obs["observation"])   # every step
    """

    def __init__(self, grid, start, goal, goal_xy, route="shortest", blocked=frozenset()):
        self.cells = plan_cells(grid, start, goal, route, blocked)
        waypoints = [cell_to_xy(grid, c) for c in self.cells[1:-1]]
        waypoints.append(np.asarray(goal_xy, dtype=np.float64))
        self.waypoints = waypoints
        self.index = 0

    def act(self, observation):
        pos = np.asarray(observation[:2], dtype=np.float64)
        vel = np.asarray(observation[2:4], dtype=np.float64)
        last = len(self.waypoints) - 1
        while (self.index < last
               and np.linalg.norm(self.waypoints[self.index] - pos) < SWITCH_RADIUS):
            self.index += 1
        target = self.waypoints[self.index]
        action = KP * (target - pos) - KD * vel
        return np.clip(action, -1.0, 1.0).astype(np.float32)


def run_episode(env, grid, start, goal, route="shortest", blocked=frozenset(), seed=None):
    """Run one fixed-length expert episode on `env`.

    Returns a dict with the observations (T+1 x 4), actions (T x 2), the goal
    position, `success` (within the goal radius at the final step, as reported
    by the environment on the last step) and `first_success_step` (the number of
    actions taken when the ball was first within the radius, or None).
    """
    obs, _ = env.reset(seed=seed, options={"reset_cell": list(start), "goal_cell": list(goal)})
    expert = ScriptedExpert(grid, start, goal, obs["desired_goal"], route, blocked)
    observations, actions = [obs["observation"].copy()], []
    first_success, info, done = None, {"success": False}, False
    t = 0
    while not done:
        action = expert.act(obs["observation"])
        obs, _, terminated, truncated, info = env.step(action)
        observations.append(obs["observation"].copy())
        actions.append(action)
        if info["success"] and first_success is None:
            first_success = t + 1  # actions taken so far
        done = terminated or truncated
        t += 1
    return {
        "observations": np.array(observations),
        "actions": np.array(actions),
        "goal": obs["desired_goal"].copy(),
        "planned_cells": expert.cells,
        "success": bool(info["success"]),
        "first_success_step": first_success,
    }
