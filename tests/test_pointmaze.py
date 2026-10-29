"""Checks that the simulator matches the maze grids and behaves as the spec says.

Run with:  python -m unittest discover -s tests

These tests need MuJoCo and Gymnasium-Robotics. They rely on a few library
internals (`point_env`, `position_noise_range`, wall geoms named "block..."),
which is safe while the versions in requirements.txt are pinned. They replace an earlier set that
mostly checked the library against itself. Each test here compares the
simulator with an independent source: the grid in `mazes.py`, a number worked
out by hand, or a controlled push of the ball.

What they establish:
1. The walls MuJoCo builds are exactly the wall cells of the grid (M0, M1, M2).
2. Cell/coordinate conversion matches hand-computed anchors, off-centre points
   and cell edges, and rejects points outside the maze.
3. The task settings are what the spec says: 10 Hz control with exactly the
   library's dynamics (including its speed cap), ±0.25 start and goal noise,
   and fixed-length episodes: reaching the goal neither ends the episode nor
   moves the goal.
4. Pushing the ball into a known wall leaves it in its cell.
5. A reset must name a free start cell and an allowed goal cell. On M1 goals
   are limited to the demo region unless the held-out region is chosen
   explicitly (as the S_out evaluation will).
6. The same seed and the same actions give the same trajectory.
"""

import sys
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from envs.mazes import M0, M1, M1_HELD_OUT_REGION, M2  # noqa: E402
from envs.pointmaze import (  # noqa: E402
    FRAME_SKIP, POSITION_NOISE, cell_to_xy, make_env, xy_to_cell,
)

M1_OPTS = {"reset_cell": [4, 1], "goal_cell": [4, 7]}


def wall_cells(grid):
    return {(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == 1}


class TestWallsMatchGrid(unittest.TestCase):
    """The simulator's wall blocks, read from the MuJoCo model, are the grid's walls."""

    def check(self, name, grid):
        env = make_env(name)
        model = env.unwrapped.point_env.model
        data = env.unwrapped.point_env.data
        env.reset(seed=0, options={"reset_cell": [1, 1], "goal_cell": [1, 2]})
        blocks = set()
        for g in range(model.ngeom):
            geom_name = model.geom(g).name
            if geom_name.startswith("block"):
                blocks.add(xy_to_cell(grid, data.geom_xpos[g][:2]))
        env.close()
        self.assertEqual(blocks, wall_cells(grid))

    def test_m1(self):
        self.check("M1", M1)

    def test_m2(self):
        self.check("M2", M2)

    def test_m0(self):
        self.check("M0", M0)


class TestCellConversion(unittest.TestCase):
    def test_hand_computed_anchors(self):
        # M1 is 9 rows x 14 columns, so the centre (0, 0) lies between them.
        self.assertTrue(np.allclose(cell_to_xy(M1, (1, 1)), [-5.5, 3.0]))   # north-west corner
        self.assertTrue(np.allclose(cell_to_xy(M1, (7, 1)), [-5.5, -3.0]))  # south-west corner
        self.assertTrue(np.allclose(cell_to_xy(M1, (4, 12)), [5.5, 0.0]))   # held-out, far east

    def test_off_centre_points(self):
        rng = np.random.default_rng(0)
        for r in range(len(M1)):
            for c in range(len(M1[0])):
                for offset in rng.uniform(-0.499, 0.499, size=(20, 2)):
                    self.assertEqual(xy_to_cell(M1, cell_to_xy(M1, (r, c)) + offset), (r, c))

    def test_edges_go_east_and_south(self):
        # x = -6.0 is the edge between columns 0 and 1; y = 3.5 between rows 0 and 1.
        self.assertEqual(xy_to_cell(M1, (-6.0, 3.0)), (1, 1))
        self.assertEqual(xy_to_cell(M1, (-5.5, 3.5)), (1, 1))

    def test_outside_the_maze_is_an_error(self):
        for point in [(-7.01, 0.0), (7.0, 0.0), (0.0, 4.51), (0.0, -4.5)]:
            with self.assertRaises(ValueError):
                xy_to_cell(M1, point)


class TestTaskSettings(unittest.TestCase):
    def setUp(self):
        self.env = make_env("M1")

    def tearDown(self):
        self.env.close()

    def test_control_rate_is_10_hz(self):
        self.assertAlmostEqual(self.env.unwrapped.point_env.dt, 0.1)

    def test_one_step_equals_ten_library_steps(self):
        # Reference: the library's own environment at 100 Hz, with no noise. Full
        # force for 1 s, so the ball reaches the speed cap and the cap is exercised.
        ref = gym.make("PointMaze_UMaze-v3", maze_map=M1, continuing_task=False)
        ref.unwrapped.position_noise_range = 0.0
        self.env.unwrapped.position_noise_range = 0.0
        options = {"reset_cell": [7, 1], "goal_cell": [1, 7]}
        ref.reset(seed=0, options=options)
        self.env.reset(seed=0, options=options)
        action = np.array([1.0, 0.0], dtype=np.float32)  # east along the south arm
        for _ in range(10):
            for _ in range(FRAME_SKIP):
                ref_obs, *_ = ref.step(action)
            obs, *_ = self.env.step(action)
            self.assertTrue(np.allclose(obs["observation"], ref_obs["observation"]))
        ref.close()

    def test_speed_stays_within_the_cap(self):
        # The library caps speed at 5 per axis before each substep, and one substep
        # of full force then adds about 0.2. With a single cap per 10 Hz step the
        # ball reaches about 7, so 5.5 separates the two.
        self.env.reset(seed=0, options={"reset_cell": [7, 1], "goal_cell": [1, 7]})
        for _ in range(10):
            obs, *_ = self.env.step(np.array([1.0, 0.0], dtype=np.float32))
            self.assertLessEqual(np.abs(obs["observation"][2:]).max(), 5.5)

    def test_start_and_goal_noise(self):
        start_centre, goal_centre = cell_to_xy(M1, (4, 1)), cell_to_xy(M1, (4, 7))
        offsets = []
        for seed in range(200):
            obs, _ = self.env.reset(seed=seed, options=M1_OPTS)
            offsets.append(obs["observation"][:2] - start_centre)
            offsets.append(obs["desired_goal"] - goal_centre)
        offsets = np.abs(np.array(offsets))
        self.assertLessEqual(offsets.max(), POSITION_NOISE)
        self.assertGreater(offsets.max(), 0.9 * POSITION_NOISE)  # noise is really on

    def test_reaching_the_goal_does_not_end_the_episode(self):
        # Steer straight at a goal three cells along the north arm and hold there
        # for the whole episode. Every step must run, the goal must not move, and
        # the final step must report success (how episode success is judged).
        env = make_env("M1", max_episode_steps=100)
        obs, _ = env.reset(seed=0, options={"reset_cell": [1, 1], "goal_cell": [1, 4]})
        goal = obs["desired_goal"].copy()
        reached_at = None
        for t in range(100):
            pos, vel = obs["observation"][:2], obs["observation"][2:]
            action = np.clip(2 * (goal - pos) - 0.5 * vel, -1, 1)
            obs, reward, terminated, truncated, info = env.step(action.astype(np.float32))
            if info["success"] and reached_at is None:
                reached_at = t
            self.assertFalse(terminated)
            self.assertTrue(np.array_equal(obs["desired_goal"], goal))
            if t < 99:
                self.assertFalse(truncated)
        env.close()
        self.assertIsNotNone(reached_at)
        self.assertLess(reached_at, 50)  # reached well before the end ...
        self.assertTrue(truncated)       # ... yet the episode ran to its full length
        self.assertTrue(info["success"])
        self.assertEqual(reward, 1.0)

    def test_episode_truncates_at_the_step_limit(self):
        env = make_env("M1", max_episode_steps=5)
        env.reset(seed=0, options=M1_OPTS)
        for _ in range(5):
            *_, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
        env.close()
        self.assertFalse(terminated)
        self.assertTrue(truncated)


class TestWallsStopTheBall(unittest.TestCase):
    """Push the ball at full force into a wall for 5 s; it must stay in its cell."""

    def push(self, cell, action):
        env = make_env("M1", max_episode_steps=1000)
        env.reset(seed=0, options={"reset_cell": list(cell), "goal_cell": [4, 7]})
        for _ in range(50):
            obs, *_ = env.step(np.array(action, dtype=np.float32))
            self.assertEqual(xy_to_cell(M1, obs["observation"][:2]), cell)
        env.close()

    def test_east_into_the_central_block(self):
        self.push((4, 1), [1.0, 0.0])

    def test_north_into_the_outer_wall(self):
        self.push((1, 3), [0.0, 1.0])

    def test_south_from_the_corridor(self):
        self.push((4, 10), [0.0, -1.0])


class TestResetGuard(unittest.TestCase):
    def test_reset_without_cells_is_refused(self):
        env = make_env("M1")
        for options in [None, {}, {"reset_cell": [4, 1]}, {"goal_cell": [4, 7]},
                        {"reset_cell": None, "goal_cell": None},
                        {"reset_cell": [4, 1], "goal_cell": None}]:
            with self.assertRaises(ValueError):
                env.reset(seed=0, options=options)
        env.close()

    def test_held_out_goals_are_refused_by_default(self):
        env = make_env("M1")
        for goal in M1_HELD_OUT_REGION:
            with self.assertRaises(ValueError):
                env.reset(seed=0, options={"reset_cell": [4, 1], "goal_cell": list(goal)})
        env.close()

    def test_held_out_goals_need_an_explicit_choice(self):
        env = make_env("M1", allowed_goals=M1_HELD_OUT_REGION)
        obs, _ = env.reset(seed=0, options={"reset_cell": [4, 1], "goal_cell": [4, 12]})
        self.assertEqual(xy_to_cell(M1, obs["desired_goal"]), (4, 12))
        with self.assertRaises(ValueError):  # and then demo goals are refused
            env.reset(seed=0, options=M1_OPTS)
        env.close()

    def test_start_in_a_wall_is_refused(self):
        env = make_env("M1")
        with self.assertRaises(ValueError):
            env.reset(seed=0, options={"reset_cell": [2, 2], "goal_cell": [4, 7]})
        env.close()


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_trajectory(self):
        actions = np.random.default_rng(0).uniform(-1, 1, size=(100, 2)).astype(np.float32)

        def rollout():
            env = make_env("M1", max_episode_steps=1000)
            obs, _ = env.reset(seed=7, options=M1_OPTS)
            trajectory = [obs["observation"].copy()]
            for a in actions:
                obs, *_ = env.step(a)
                trajectory.append(obs["observation"].copy())
            env.close()
            return np.array(trajectory)

        self.assertTrue(np.array_equal(rollout(), rollout()))


class TestBlockedCells(unittest.TestCase):
    """Perturbation P1 turns one free cell into a wall at evaluation time."""

    def test_blocked_cell_becomes_a_wall_in_the_simulator(self):
        env = make_env("M1", blocked_cells=frozenset({(1, 4)}))
        model = env.unwrapped.point_env.model
        data = env.unwrapped.point_env.data
        env.reset(seed=0, options={"reset_cell": [1, 1], "goal_cell": [1, 2]})
        blocks = {xy_to_cell(M1, data.geom_xpos[g][:2]) for g in range(model.ngeom)
                  if model.geom(g).name.startswith("block")}
        env.close()
        self.assertEqual(blocks, wall_cells(M1) | {(1, 4)})

    def test_ball_cannot_pass_the_block(self):
        # Full force east along the north arm from (1, 3): the ball stays west of (1, 4).
        env = make_env("M1", max_episode_steps=1000, blocked_cells=frozenset({(1, 4)}))
        obs, _ = env.reset(seed=0, options={"reset_cell": [1, 3], "goal_cell": [1, 7]})
        for _ in range(50):
            obs, *_ = env.step(np.array([1.0, 0.0], dtype=np.float32))
            self.assertEqual(xy_to_cell(M1, obs["observation"][:2]), (1, 3))
        env.close()

    def test_block_must_be_a_free_cell_and_is_no_longer_a_goal(self):
        with self.assertRaises(ValueError):
            make_env("M1", blocked_cells=frozenset({(2, 2)}))  # already a wall
        env = make_env("M1", blocked_cells=frozenset({(1, 4)}))
        with self.assertRaises(ValueError):
            env.reset(seed=0, options={"reset_cell": [1, 1], "goal_cell": [1, 4]})
        env.close()

    def test_unblocked_env_is_unchanged(self):
        # Building a blocked env must not alter the shared grid.
        make_env("M1", blocked_cells=frozenset({(1, 4)})).close()
        self.assertEqual(M1[1][4], 0)


if __name__ == "__main__":
    unittest.main()
