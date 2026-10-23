"""Checks that the simulator really uses the M1 grid (Stage 1, simulator integration).

Run with:  python -m unittest discover -s tests

These tests need MuJoCo and Gymnasium-Robotics. They confirm three things the
rest of the project relies on: the environment builds and steps on M1; grid cells
and simulator coordinates convert back and forth exactly; and the simulator's
walls match the grid, so the ball cannot pass through a wall cell.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from envs.mazes import M1, free_cells  # noqa: E402
from envs.pointmaze import make_env, cell_to_xy, xy_to_cell  # noqa: E402


class TestPointMazeM1(unittest.TestCase):
    def setUp(self):
        self.env = make_env("M1")

    def tearDown(self):
        self.env.close()

    def test_reset_and_step(self):
        obs, _ = self.env.reset(seed=0, options={"reset_cell": [1, 1], "goal_cell": [1, 7]})
        self.assertEqual(obs["observation"].shape, (4,))  # x, y, vx, vy
        self.assertEqual(obs["desired_goal"].shape, (2,))
        obs, *_ = self.env.step(self.env.action_space.sample())
        self.assertTrue(np.all(np.isfinite(obs["observation"])))

    def test_cells_round_trip(self):
        self.env.reset(seed=0)
        for cell in free_cells(M1):
            self.assertEqual(xy_to_cell(self.env, cell_to_xy(self.env, cell)), cell)

    def test_start_and_goal_land_in_requested_cells(self):
        obs, _ = self.env.reset(seed=1, options={"reset_cell": [7, 1], "goal_cell": [7, 7]})
        self.assertEqual(xy_to_cell(self.env, obs["observation"][:2]), (7, 1))
        self.assertEqual(xy_to_cell(self.env, obs["desired_goal"]), (7, 7))

    def test_ball_never_enters_a_wall_cell(self):
        # Push the ball hard in random directions; it must stay in free cells.
        free = free_cells(M1)
        rng = np.random.default_rng(0)
        self.env.reset(seed=2, options={"reset_cell": [4, 1], "goal_cell": [4, 7]})
        for _ in range(2000):
            obs, _, term, trunc, _ = self.env.step(rng.uniform(-1, 1, size=2).astype(np.float32))
            self.assertIn(xy_to_cell(self.env, obs["observation"][:2]), free)
            if term or trunc:
                self.env.reset()

    def test_m2_builds(self):
        env = make_env("M2")
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs["observation"].shape, (4,))
        env.close()


if __name__ == "__main__":
    unittest.main()
