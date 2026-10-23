"""Build the Gymnasium-Robotics PointMaze simulator on this project's maze grids.

What this file does, in one sentence: `make_env("M1")` returns a PointMaze
environment whose walls are exactly the grid in `mazes.py`, plus two helpers that
convert between grid cells (row, col) and simulator coordinates (x, y).

Why it matters: the planner checks in `mazes.py` prove the two-route structure on
the grid. This file is the bridge that makes the simulated ball move in that same
grid, so the checks carry over to the experiment. Cell conversion is also what the
route classifier will use to turn a trajectory into a sequence of cells.

Coordinate convention (from Gymnasium-Robotics): the maze is centred on (0, 0),
each cell is 1 unit wide, x grows with column and y grows *upward*, so row 1 has
a larger y than row 7. Resets add up to ±0.25 of uniform noise to start and goal.
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np

from envs.mazes import M1, M2

gym.register_envs(gymnasium_robotics)

MAZES = {"M1": M1, "M2": M2}

# Any registered PointMaze id works as a base; `maze_map` replaces its layout.
_BASE_ID = "PointMaze_UMaze-v3"


def make_env(maze: str = "M1", max_episode_steps: int = 300, **kwargs) -> gym.Env:
    """Return a PointMaze environment on one of this project's layouts.

    Start and goal cells are chosen per episode through reset options:
        env.reset(seed=0, options={"reset_cell": [r, c], "goal_cell": [r, c]})
    Episode length is a planning default; the pilot sets the final value.
    """
    return gym.make(_BASE_ID, maze_map=MAZES[maze], max_episode_steps=max_episode_steps, **kwargs)


def cell_to_xy(env: gym.Env, cell: tuple[int, int]) -> np.ndarray:
    """Centre of grid cell (row, col) in simulator coordinates."""
    return env.unwrapped.maze.cell_rowcol_to_xy(np.asarray(cell))


def xy_to_cell(env: gym.Env, xy) -> tuple[int, int]:
    """Grid cell (row, col) containing simulator point (x, y)."""
    r, c = env.unwrapped.maze.cell_xy_to_rowcol(np.asarray(xy))
    return int(r), int(c)
