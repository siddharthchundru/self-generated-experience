"""Build the Gymnasium-Robotics PointMaze simulator on this project's maze grids.

What this file does, in one sentence: `make_env("M1")` returns a PointMaze
environment whose walls are exactly the grid in `mazes.py`, with every task
setting fixed explicitly, plus two helpers that convert between grid cells
(row, col) and simulator coordinates (x, y).

Why it matters: the planner checks in `mazes.py` prove the two-route structure on
the grid. This file is the bridge that makes the simulated ball move in that same
grid. Cell conversion is also what the route classifier will use to turn a
trajectory into a sequence of cells.

Task settings (each set here, none inherited silently from the library):
- Control rate 10 Hz. The library advances physics one 0.01 s substep per
  action (100 Hz) and gives no argument to change it. Here each action is held
  for 10 substeps (0.1 s). The library caps the ball's speed at 5 per axis before
  each of its steps; holding an action for 10 substeps with one cap would let
  the ball go faster (about 6.9). So the cap is applied before every substep,
  and one 10 Hz step is exactly ten of the library's own steps.
- An episode ends when the ball comes within 0.45 of the goal (the library's
  fixed success radius), or at `max_episode_steps`.
- Sparse reward: 1 on the step that reaches the goal, otherwise 0. (The
  learner, GCSL, does not use reward; it is logged for checks.)
- Start and goal positions are the centre of the requested cell plus uniform
  noise of up to ±POSITION_NOISE in x and y. The library's constructor rejects
  this argument, so it too is set after construction.

Coordinate convention (from Gymnasium-Robotics): the maze is centred on (0, 0),
each cell is 1 unit wide, x grows with column and y grows *upward*, so row 1 has
a larger y than row 7.
"""

import math
import types

import gymnasium as gym
import gymnasium_robotics
import numpy as np

from envs.mazes import M1, M1_DEMO_REGION, M2, free_cells

gym.register_envs(gymnasium_robotics)

MAZES = {"M1": M1, "M2": M2}
DEFAULT_ALLOWED_GOALS = {"M1": M1_DEMO_REGION, "M2": frozenset(free_cells(M2))}

CONTROL_HZ = 10
PHYSICS_DT = 0.01                                # library's fixed MuJoCo substep (s)
FRAME_SKIP = round(1 / (CONTROL_HZ * PHYSICS_DT))  # = 10 substeps per action
POSITION_NOISE = 0.25                            # cell widths; planning default
SUCCESS_RADIUS = 0.45                            # fixed in the library; recorded here
MAX_EPISODE_STEPS = 100                          # 10 s; planning default, set in the pilot

# Any registered PointMaze id works as a base; `maze_map` replaces its layout,
# and every setting that matters is passed or set below.
_BASE_ID = "PointMaze_UMaze-v3"


def make_env(
    maze: str = "M1",
    max_episode_steps: int = MAX_EPISODE_STEPS,
    allowed_goals: frozenset | None = None,
) -> gym.Env:
    """Return a PointMaze environment on one of this project's layouts.

    Start and goal cells must be named on every reset:
        env.reset(seed=0, options={"reset_cell": [r, c], "goal_cell": [r, c]})
    and the goal cell must be in `allowed_goals`. On M1 the default is the demo
    region R_d; the S_out evaluation passes M1_HELD_OUT_REGION explicitly. On M2,
    which has no held-out region, every free cell is allowed. See RequireCells.
    """
    env = gym.make(
        _BASE_ID,
        maze_map=MAZES[maze],
        reward_type="sparse",
        continuing_task=False,   # success ends the episode
        reset_target=False,      # irrelevant when not continuing; fixed anyway
        max_episode_steps=max_episode_steps,
    )
    inner = env.unwrapped
    point = inner.point_env
    point.frame_skip = FRAME_SKIP  # so point.dt reports 0.1 s; stepping is below
    point.step = types.MethodType(_step_with_cap_per_substep, point)
    for e in (inner, point):
        e.metadata = {**e.metadata, "render_fps": CONTROL_HZ}
    inner.position_noise_range = POSITION_NOISE
    assert abs(point.dt - 1 / CONTROL_HZ) < 1e-9
    if allowed_goals is None:
        allowed_goals = DEFAULT_ALLOWED_GOALS[maze]
    return RequireCells(env, MAZES[maze], allowed_goals)


def _step_with_cap_per_substep(self, action):
    """Replacement for the library's PointEnv.step at 10 Hz: the same speed cap
    and physics as ten library steps with the same action."""
    action = np.clip(action, -1.0, 1.0)
    for _ in range(FRAME_SKIP):
        self._clip_velocity()
        self.do_simulation(action, 1)
    obs, info = self._get_obs()
    return obs, 0, False, False, info


class RequireCells(gym.Wrapper):
    """Refuse any reset that does not name a free start cell and an allowed goal cell.

    Why: when no cell is named (or a cell is given as None), the library picks a
    random free cell, and about a third of M1's free cells are in the held-out
    region. A training goal drawn there would let the agent practise on the cells
    reserved for the generalisation test (S_out). Every reset must therefore name
    its cells, and the goal must come from the set this environment was built for.

    Limit: code that calls `env.unwrapped.reset()` skips this wrapper. Nothing in
    this project should; the data-collection audit checks every logged goal.
    """

    def __init__(self, env, grid, allowed_goals):
        super().__init__(env)
        self.free = {(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == 0}
        self.allowed_goals = frozenset(allowed_goals)
        assert self.allowed_goals <= self.free

    def reset(self, *, seed=None, options=None):
        options = options or {}
        start, goal = options.get("reset_cell"), options.get("goal_cell")
        if start is None or goal is None:
            raise ValueError("reset needs options={'reset_cell': [r, c], 'goal_cell': [r, c]}")
        start, goal = tuple(int(v) for v in start), tuple(int(v) for v in goal)
        if start not in self.free:
            raise ValueError(f"start cell {start} is not a free cell")
        if goal not in self.allowed_goals:
            raise ValueError(f"goal cell {goal} is not allowed in this environment")
        return self.env.reset(seed=seed, options=options)


def cell_to_xy(grid: list[list[int]], cell: tuple[int, int]) -> np.ndarray:
    """Centre of grid cell (row, col) in simulator coordinates.

    Computed from the grid's size alone, not by asking the simulator, so that the
    tests compare two independent sources. Cells are 1 unit wide and the maze is
    centred on (0, 0); y grows upward, so row 0 is at the top.
    """
    n_rows, n_cols = len(grid), len(grid[0])
    r, c = cell
    return np.array([c + 0.5 - n_cols / 2, n_rows / 2 - r - 0.5])


def xy_to_cell(grid: list[list[int]], xy) -> tuple[int, int]:
    """Grid cell (row, col) containing simulator point (x, y).

    Cells include their west and north edges and exclude their east and south
    edges, so a point exactly on a shared edge goes to the cell east of it, or to
    the cell south of it. (This matches the library.) Points outside the maze
    raise an error instead of returning an invalid cell.
    """
    n_rows, n_cols = len(grid), len(grid[0])
    x, y = float(xy[0]), float(xy[1])
    r, c = math.floor(n_rows / 2 - y), math.floor(x + n_cols / 2)
    if not (0 <= r < n_rows and 0 <= c < n_cols):
        raise ValueError(f"point {(x, y)} is outside the {n_rows}x{n_cols} maze")
    return r, c
