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
5. Demonstrations on M1's ring pairs never enter the held-out region, so goals
   relabelled from their states cannot leak into it.
6. The same holds on the tuning maze M0: both routes on every west-east pair,
   classified correctly by the trajectory route classifier, inside its ring.
7. The episode's `success` is the final step only: a ball that reaches the goal
   and is then pushed away counts as a failure. The expert ends close to the
   exact (noisy) goal, not merely inside its cell.

These run the simulator, so they take about a minute.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from envs.mazes import M0, M0_DEMO_REGION, M1, M1_DEMO_REGION, M2, free_cells, route_class  # noqa: E402
from envs.pointmaze import make_env, xy_to_cell  # noqa: E402
from expert import plan_cells, run_episode  # noqa: E402
from routes import trajectory_route  # noqa: E402

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
                        visited = {xy_to_cell(M1, o[:2]) for o in ep["observations"]}
                        self.assertLessEqual(visited, M1_DEMO_REGION)
                        final_distance = np.linalg.norm(ep["observations"][-1][:2] - ep["goal"])
                        self.assertLess(final_distance, 0.2)
        env.close()


class TestBothRoutesOnM0(unittest.TestCase):
    def test_every_pair_every_route(self):
        env = make_env("M0", max_episode_steps=T)
        for row in range(1, 6):
            for route in ("north", "south"):
                for seed in range(5):
                    with self.subTest(row=row, route=route, seed=seed):
                        ep = run_episode(env, M0, (row, 1), (row, 9), route, seed=seed)
                        self.assertTrue(ep["success"])
                        self.assertEqual(trajectory_route(ep["observations"], "M0"), route)
                        visited = {xy_to_cell(M0, o[:2]) for o in ep["observations"]}
                        self.assertLessEqual(visited, M0_DEMO_REGION)
        env.close()


class TestSuccessIsJudgedAtTheFinalStep(unittest.TestCase):
    def test_reach_then_leave_is_a_failure(self):
        # Wrap the env so that, after the expert has held the goal for a while,
        # its actions are replaced by a full push away from the goal.
        import expert as expert_module

        class PushAwayLate:
            def __init__(self, env, after):
                self.env, self.after, self.t = env, after, 0

            def reset(self, **kw):
                self.t = 0
                return self.env.reset(**kw)

            def step(self, action):
                self.t += 1
                if self.t > self.after:
                    action = np.array([-1.0, 0.0], dtype=np.float32)  # west, away from (1, 7)
                return self.env.step(action)

        env = make_env("M1", max_episode_steps=T)
        ep = expert_module.run_episode(PushAwayLate(env, 40), M1, (1, 1), (1, 7), "north", seed=0)
        env.close()
        self.assertIsNotNone(ep["first_success_step"])  # it did reach the goal ...
        self.assertFalse(ep["success"])                  # ... but was not there at the end


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
    def test_named_route_on_a_maze_without_a_ring_is_refused(self):
        with self.assertRaises(ValueError):
            plan_cells(M2, (1, 1), (6, 6), "north")

    def test_route_through_both_blocked_arms_is_refused(self):
        with self.assertRaises(ValueError):
            plan_cells(M1, (2, 1), (2, 7), "north", blocked=frozenset({(1, 4)}))


if __name__ == "__main__":
    unittest.main()
