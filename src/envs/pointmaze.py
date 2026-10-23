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
  action (100 Hz) and gives no argument to change it, so `frame_skip` is set to
  10 after construction: each action is held for 10 substeps (0.1 s).
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

import gymnasium as gym
import gymnasium_robotics
import numpy as np

from envs.mazes import M1, M2

gym.register_envs(gymnasium_robotics)

MAZES = {"M1": M1, "M2": M2}

CONTROL_HZ = 10
PHYSICS_DT = 0.01                                # library's fixed MuJoCo substep (s)
FRAME_SKIP = round(1 / (CONTROL_HZ * PHYSICS_DT))  # = 10 substeps per action
POSITION_NOISE = 0.25                            # cell widths; planning default
SUCCESS_RADIUS = 0.45                            # fixed in the library; recorded here
MAX_EPISODE_STEPS = 100                          # 10 s; planning default, set in the pilot

# Any registered PointMaze id works as a base; `maze_map` replaces its layout,
# and every setting that matters is passed or set below.
_BASE_ID = "PointMaze_UMaze-v3"


def make_env(maze: str = "M1", max_episode_steps: int = MAX_EPISODE_STEPS) -> gym.Env:
    """Return a PointMaze environment on one of this project's layouts.

    Start and goal cells must be named on every reset:
        env.reset(seed=0, options={"reset_cell": [r, c], "goal_cell": [r, c]})
    A reset without them raises an error (see RequireCells).
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
    inner.point_env.frame_skip = FRAME_SKIP
    inner.position_noise_range = POSITION_NOISE
    assert abs(inner.point_env.dt - 1 / CONTROL_HZ) < 1e-9
    return RequireCells(env)


class RequireCells(gym.Wrapper):
    """Refuse any reset that does not name both the start and the goal cell.

    Why: when no cell is named, the library picks a random free cell, and about a
    third of M1's free cells are in the held-out region. A training goal drawn
    there would let the agent practise on the cells reserved for the
    generalisation test (S_out). Forcing every caller to name its cells means
    goals come only from each condition's own, checkable rule.
    """

    def reset(self, *, seed=None, options=None):
        options = options or {}
        if "reset_cell" not in options or "goal_cell" not in options:
            raise ValueError("reset needs options={'reset_cell': [r, c], 'goal_cell': [r, c]}")
        return self.env.reset(seed=seed, options=options)


def cell_to_xy(env: gym.Env, cell: tuple[int, int]) -> np.ndarray:
    """Centre of grid cell (row, col) in simulator coordinates."""
    return env.unwrapped.maze.cell_rowcol_to_xy(np.asarray(cell))


def xy_to_cell(env: gym.Env, xy) -> tuple[int, int]:
    """Grid cell (row, col) containing simulator point (x, y)."""
    r, c = env.unwrapped.maze.cell_xy_to_rowcol(np.asarray(xy))
    return int(r), int(c)
