"""Check the collection loops (spec section 6) and frontier novelty's goal rule.

What these tests protect:
- every collecting condition starts episode i at the same task pair with the
  same reset noise, so the conditions differ only in the goal and who acts;
- every episode is exactly T steps and stores observations, executed actions
  and the commanded goal, nothing else;
- goal leakage (spec section 7): only frontier novelty may command a held-out
  goal; every other condition's environment refuses one, including an exact
  goal point;
- frontier novelty commands the exact position its selector chose, and the
  selector picks the lowest-density candidate;
- expert top-up's "alternate" rule splits each pair's episodes evenly between
  routes (to within one), and the expert drives the route it was told to;
- random actions on the grid rule are bin centres;
- the fork cells are the start cells of the shorter-route pairs.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from collection import (COLLECTING, EXPERT, FRONTIER, RANDOM, TASK_DIRECTED, Collector, fork_cells,
                        held_out_cells, start_draw)
from envs.mazes import M1_DEMO_REGION, M1_HELD_OUT_REGION, RINGS
from envs.pointmaze import cell_to_xy, xy_to_cell
from episode import EPISODE_STEPS
from frontier import FrontierSelector
from gcsl import BIN_CENTRES, GCSLPolicy

T = EPISODE_STEPS
GRID = RINGS["M1"]["grid"]


class FixedSelector:
    """Stands in for the frontier selector: always commands the same point."""

    def __init__(self, xy):
        self.xy = np.asarray(xy, dtype=np.float64)

    def choose(self, rng):
        return self.xy.copy(), {"fixed": True}


def make(condition, selector=None, **kwargs):
    policy = GCSLPolicy(0) if condition in (TASK_DIRECTED, FRONTIER) else None
    if condition == FRONTIER and selector is None:
        selector = FixedSelector(cell_to_xy(GRID, (1, 3)))
    return Collector(condition, "M1", 0, policy=policy, selector=selector, **kwargs)


class TestStarts(unittest.TestCase):
    def test_same_start_in_every_condition(self):
        firsts = {}
        for condition in COLLECTING:
            collector = make(condition)
            try:
                firsts[condition] = [collector.run_episode(i)[0]["observations"][0] for i in range(4)]
            finally:
                collector.close()
        for condition in COLLECTING:
            np.testing.assert_array_equal(firsts[condition], firsts[TASK_DIRECTED])

    def test_start_draw_depends_on_seed_and_index_only(self):
        draws = [start_draw("M1", 3, i) for i in range(700)]
        self.assertEqual(draws[5], start_draw("M1", 3, 5))
        rows = np.bincount([d.pair.row for d in draws], minlength=8)[1:]
        self.assertTrue(all(70 <= n <= 130 for n in rows), rows)  # uniform over the seven pairs
        self.assertNotEqual([d.reset_seed for d in draws[:5]], [start_draw("M1", 4, i).reset_seed for i in range(5)])
        with self.assertRaises(ValueError):
            start_draw("M1", 0, 0, distribution="random_free_cells")


class TestEpisodes(unittest.TestCase):
    def test_fixed_length_and_what_is_stored(self):
        for condition in COLLECTING:
            collector = make(condition)
            try:
                trajectory, record = collector.run_episode(0)
            finally:
                collector.close()
            self.assertEqual(set(trajectory), {"observations", "actions", "goal"})
            self.assertEqual(trajectory["observations"].shape, (T + 1, 4))
            self.assertEqual(trajectory["actions"].shape, (T, 2))
            self.assertEqual(record["condition"], condition)

    def test_task_directed_commands_the_drawn_pair(self):
        collector = make(TASK_DIRECTED)
        try:
            for i in range(5):
                _, record = collector.run_episode(i)
                pair = start_draw("M1", 0, i).pair
                self.assertEqual(tuple(record["goal_cell"]), pair.goal)
                self.assertFalse(record["goal_in_held_out"])
        finally:
            collector.close()

    def test_after_step_is_called_once_per_step(self):
        collector = make(RANDOM)
        steps = []
        try:
            collector.run_episode(0, after_step=steps.append)
        finally:
            collector.close()
        self.assertEqual(steps, list(range(1, T + 1)))

    def test_random_grid_actions_are_bin_centres(self):
        collector = make(RANDOM, random_actions="grid")
        try:
            trajectory, _ = collector.run_episode(0)
        finally:
            collector.close()
        self.assertTrue(np.isin(trajectory["actions"], BIN_CENTRES.astype(np.float32)).all())


class TestGoalLeakage(unittest.TestCase):
    def test_held_out_cells(self):
        self.assertEqual(held_out_cells("M1"), M1_HELD_OUT_REGION)
        self.assertTrue(M1_HELD_OUT_REGION)

    def test_only_frontier_environment_allows_held_out_goals(self):
        for condition in COLLECTING:
            collector = make(condition)
            try:
                allows = bool(collector.env.allowed_goals & M1_HELD_OUT_REGION)
                self.assertEqual(allows, condition == FRONTIER, condition)
                self.assertTrue(M1_DEMO_REGION <= collector.env.allowed_goals)
            finally:
                collector.close()

    def test_exact_held_out_goal_refused_outside_frontier(self):
        held = sorted(M1_HELD_OUT_REGION)[0]
        collector = make(TASK_DIRECTED)
        try:
            obs, _ = collector.env.reset(seed=0, options={"reset_cell": [1, 1], "goal_cell": [1, 7]})
            with self.assertRaises(ValueError):
                collector.env.set_goal_position(obs, cell_to_xy(GRID, held))
        finally:
            collector.close()

    def test_frontier_commands_its_exact_point_even_held_out(self):
        point = cell_to_xy(GRID, sorted(M1_HELD_OUT_REGION)[0]) + np.array([0.1, -0.2])
        collector = make(FRONTIER, selector=FixedSelector(point))
        try:
            trajectory, record = collector.run_episode(0)
        finally:
            collector.close()
        np.testing.assert_allclose(trajectory["goal"], point)
        self.assertTrue(record["goal_in_held_out"])
        self.assertEqual(tuple(record["goal_cell"]), xy_to_cell(GRID, point))

    def test_per_episode_assertion_catches_a_held_out_goal(self):  # audit of 4 Nov, MI4
        held = cell_to_xy(GRID, sorted(M1_HELD_OUT_REGION)[0])
        collector = make(TASK_DIRECTED)
        original = collector.env.reset

        def reset(**kwargs):
            obs, info = original(**kwargs)
            return {**obs, "desired_goal": held.copy()}, info

        collector.env.reset = reset
        try:
            with self.assertRaisesRegex(AssertionError, "held-out goal"):
                collector.run_episode(0)
        finally:
            collector.close()

    def test_constructor_pairs_conditions_with_actors(self):
        with self.assertRaises(ValueError):
            Collector(TASK_DIRECTED, "M1", 0)                                   # no policy
        with self.assertRaises(ValueError):
            Collector(RANDOM, "M1", 0, policy=GCSLPolicy(0))                    # random does not act with it
        with self.assertRaises(ValueError):
            Collector(TASK_DIRECTED, "M1", 0, policy=GCSLPolicy(0), selector=FixedSelector([0, 0]))
        with self.assertRaises(ValueError):
            Collector("demonstrations_only", "M1", 0)


class TestExpertTopUp(unittest.TestCase):
    def test_alternating_routes_split_each_pair_evenly_and_are_driven(self):
        collector = make(EXPERT)
        per_row = {}
        try:
            for i in range(42):
                trajectory, record = collector.run_episode(i)
                self.assertTrue(record["success"])
                self.assertEqual(record["route"], record["expert_route"])
                per_row.setdefault(record["start_row"], []).append(record["expert_route"])
        finally:
            collector.close()
        for row, routes in per_row.items():
            self.assertLessEqual(abs(routes.count("north") - routes.count("south")), 1, (row, routes))


class TestForks(unittest.TestCase):
    def test_fork_cells_are_shorter_route_start_cells(self):
        self.assertEqual(fork_cells("M1"), {r: (r, 1) for r in (1, 2, 3, 5, 6, 7)})
        self.assertEqual(fork_cells("M0"), {r: (r, 1) for r in (1, 2, 4, 5)})


class TestFrontierSelector(unittest.TestCase):
    def test_log_density_matches_brute_force(self):
        rng = np.random.default_rng(0)
        fitted = rng.normal(loc=[1.0, -2.0], scale=[2.0, 0.5], size=(50, 2))
        points = fitted[:7] + rng.normal(scale=0.05, size=(7, 2))  # near the data, so the brute force stays finite
        fixed = FrontierSelector(bandwidth=0.1, position_scale=5.0, normalisation="fixed")
        mega = FrontierSelector(bandwidth=0.1)  # default: MEGA's per-axis standardisation
        mean, std = fitted.mean(0), fitted.std(0) + 1e-4
        for selector, (a, b) in ((fixed, (points / 5.0, fitted / 5.0)),
                                 (mega, ((points - mean) / std, (fitted - mean) / std))):
            brute = np.log(np.mean(np.exp(-((a[:, None] - b[None]) ** 2).sum(-1) / (2 * 0.01)), axis=1))
            np.testing.assert_allclose(selector.log_density(points, fitted), brute, rtol=1e-9, atol=1e-9)
        with self.assertRaises(ValueError):
            FrontierSelector(normalisation="minmax")

    def test_candidates_are_drawn_from_every_achieved_position(self):  # audit of 4 Nov, M21
        positions = np.stack([np.arange(20_000.0), np.zeros(20_000)], axis=1)
        selector = FrontierSelector(candidates=1, fit_points=10)
        selector.add_positions(positions)
        for seed in range(20):
            expected = positions[np.random.default_rng(seed).integers(20_000)]
            goal, _ = selector.choose(np.random.default_rng(seed))
            np.testing.assert_array_equal(goal, expected)

    def test_chooses_the_isolated_point(self):
        selector = FrontierSelector(candidates=2000, fit_points=10_000)
        cluster = np.random.default_rng(1).normal(scale=0.05, size=(999, 2))
        selector.add_positions(cluster)
        selector.add_positions(np.array([[3.0, 3.0]]))
        goal, info = selector.choose(np.random.default_rng(2))
        np.testing.assert_array_equal(goal, [3.0, 3.0])
        self.assertEqual(info["fit_points"], 1000)

    def test_candidates_come_from_achieved_positions_and_fit_is_capped(self):
        selector = FrontierSelector(candidates=100, fit_points=10_000)
        positions = np.random.default_rng(3).uniform(-3, 3, size=(25_000, 4))
        for chunk in np.split(positions, 5):
            selector.add_positions(chunk)
        self.assertEqual(len(selector.achieved), 25_000)
        goal, info = selector.choose(np.random.default_rng(4))
        self.assertEqual(info["fit_points"], 10_000)
        self.assertTrue((np.abs(positions[:, :2] - goal).sum(1) == 0).any())

    def test_empty_selector_refuses(self):
        with self.assertRaises(ValueError):
            FrontierSelector().choose(np.random.default_rng(0))


if __name__ == "__main__":
    unittest.main()
