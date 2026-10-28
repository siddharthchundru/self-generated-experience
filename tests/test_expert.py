"""Checks that the scripted expert does what the demonstrations need (Stage 1 gate).

Run with:  python -m unittest discover -s tests

What they establish:
1. On every west-east pair of M1's ring (rows 1-7), the expert reaches the goal
   by the final step on BOTH routes, and the route it actually drove (read from
   the simulated trajectory, not from its plan) is the one it was told to take.
   This is the gate "expert reaches ~100% on M1 on both routes".
2. With a cell of the shorter arm turned into a real wall in the simulator (the
   P1 perturbation), the expert still succeeds on the longer route. Both arm
   ends are tested as well as the middle, because the ball passes closest to
   them; the exact P1 cell is fixed with the evaluation suite.
3. Taking the planner's shortest route, the expert reaches every free cell from
   every other free cell on M1 and on M2. The learner-capacity check (spec
   section 7) needs plentiful whole-maze expert data, held-out region included.
4. Asking for a route that cannot exist raises an error instead of silently
   driving a different route.

These run the simulator, so they take about a minute.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from envs.mazes import M1, M2, free_cells, route_class  # noqa: E402
from envs.pointmaze import make_env, xy_to_cell  # noqa: E402
from expert import plan_cells, run_episode  # noqa: E402

T = 70  # long enough for the longest M1 pair (about 53 steps); not the experiment's T
ROWS = range(1, 8)
SEEDS = range(20)


def driven_route(grid, observations):
    return route_class([xy_to_cell(grid, o[:2]) for o in observations])


class TestBothRoutesOnM1(unittest.TestCase):
    def test_every_pair_every_route(self):
        env = make_env("M1", max_episode_steps=T)
        for row in ROWS:
            for route in ("north", "south"):
                for seed in SEEDS:
                    with self.subTest(row=row, route=route, seed=seed):
                        ep = run_episode(env, M1, (row, 1), (row, 7), route, seed=seed)
                        self.assertTrue(ep["success"])
                        self.assertEqual(len(ep["actions"]), T)  # fixed length
                        self.assertEqual(driven_route(M1, ep["observations"]), route)
        env.close()


class TestP1Block(unittest.TestCase):
    def test_longer_route_succeeds_with_a_real_wall(self):
        for row in ROWS:
            if row == 4:
                continue  # tie pairs have no shorter route
            shorter, longer = ("north", "south") if row < 4 else ("south", "north")
            arm_row = 1 if shorter == "north" else 7
            for col in (2, 4, 6):
                block = frozenset({(arm_row, col)})
                with self.assertRaises(ValueError):  # the shorter route is gone
                    plan_cells(M1, (row, 1), (row, 7), shorter, blocked=block)
                env = make_env("M1", max_episode_steps=T, blocked_cells=block)
                for seed in range(5):
                    with self.subTest(row=row, block=(arm_row, col), seed=seed):
                        ep = run_episode(env, M1, (row, 1), (row, 7), longer,
                                         blocked=block, seed=seed)
                        self.assertTrue(ep["success"])
                        self.assertEqual(driven_route(M1, ep["observations"]), longer)
                env.close()


class TestAllPairs(unittest.TestCase):
    def check(self, name, grid):
        free = sorted(free_cells(grid))
        env = make_env(name, max_episode_steps=T, allowed_goals=frozenset(free))
        failures = []
        for start in free:
            for goal in free:
                if start != goal:
                    ep = run_episode(env, grid, start, goal, seed=0)
                    if not ep["success"]:
                        failures.append((start, goal))
        env.close()
        self.assertEqual(failures, [])

    def test_m1(self):
        self.check("M1", M1)

    def test_m2(self):
        self.check("M2", M2)


class TestImpossibleRoute(unittest.TestCase):
    def test_route_through_both_blocked_arms_is_refused(self):
        with self.assertRaises(ValueError):
            plan_cells(M1, (2, 1), (2, 7), "north", blocked=frozenset({(1, 4)}))


if __name__ == "__main__":
    unittest.main()
