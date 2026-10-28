"""Checks for the trajectory route classifier (src/routes.py).

Run with:  python -m unittest discover -s tests

What they establish:
1. On expert trajectories (both routes, every west-east ring pair), the
   classifier returns the route the expert was told to take, and agrees with
   the cell-set rule used for planner paths.
2. Hand-built trajectories cover the cases a learned policy can produce and the
   expert never does: a dip into one arm before going round the other way, a
   full crossing followed by a return and a crossing on the other arm, never
   getting across, and touching the midline exactly and going back.
3. On the tuning maze M0 the rule uses M0's own midline and arms.
4. A crossing outside both arms, which the walls make impossible, raises an
   error instead of being silently classified.

The hand-labelled check on messy simulated trajectories (the Stage 1 gate) is
separate: see scripts/route_handcheck.py.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from envs.mazes import M0, M1, route_class  # noqa: E402
from envs.pointmaze import cell_to_xy, make_env, xy_to_cell  # noqa: E402
from expert import run_episode  # noqa: E402
from routes import MIDLINE_X, trajectory_route  # noqa: E402


def through(*cells, grid=M1):
    """Positions along straight segments joining cell centres, 5 points per segment."""
    pts = [cell_to_xy(grid, c) for c in cells]
    out = []
    for a, b in zip(pts, pts[1:]):
        for f in np.linspace(0, 1, 5, endpoint=False):
            out.append(a + f * (b - a))
    out.append(pts[-1])
    return np.array(out)


class TestExpertTrajectories(unittest.TestCase):
    def test_matches_commanded_route_and_cell_rule(self):
        env = make_env("M1", max_episode_steps=70)
        for row in range(1, 8):
            for route in ("north", "south"):
                for seed in range(5):
                    ep = run_episode(env, M1, (row, 1), (row, 7), route, seed=seed)
                    obs = ep["observations"]
                    self.assertEqual(trajectory_route(obs), route)
                    self.assertEqual(route_class([xy_to_cell(M1, o[:2]) for o in obs]), route)
        env.close()


class TestHandBuiltTrajectories(unittest.TestCase):
    def test_dip_into_north_arm_then_south(self):
        # Up into the north arm as far as (1, 3), back down the west side, round the south.
        path = through((2, 1), (1, 1), (1, 3), (1, 1), (7, 1), (7, 7), (2, 7))
        self.assertEqual(trajectory_route(path), "south")
        # The cell-set rule calls this "both"; that is why it is not used here.
        self.assertEqual(route_class([xy_to_cell(M1, p) for p in path]), "both")

    def test_cross_north_return_then_cross_south(self):
        path = through((2, 1), (1, 1), (1, 6), (1, 1), (7, 1), (7, 7), (2, 7))
        self.assertEqual(trajectory_route(path), "south")

    def test_never_gets_across(self):
        path = through((3, 1), (1, 1), (1, 3), (7, 1), (7, 3))
        self.assertEqual(trajectory_route(path), "neither")

    def test_touching_the_midline_is_not_a_crossing(self):
        y = cell_to_xy(M1, (1, 4))[1]
        path = np.array([[MIDLINE_X - 0.5, y], [MIDLINE_X, y], [MIDLINE_X - 0.3, y]])
        self.assertEqual(trajectory_route(path), "neither")

    def test_crossing_via_a_point_on_the_line(self):
        y = cell_to_xy(M1, (7, 4))[1]
        path = np.array([[MIDLINE_X - 0.5, y], [MIDLINE_X, y], [MIDLINE_X + 0.3, y]])
        self.assertEqual(trajectory_route(path), "south")

    def test_impossible_crossing_raises(self):
        path = np.array([[MIDLINE_X - 0.5, 0.0], [MIDLINE_X + 0.5, 0.0]])  # through the block
        with self.assertRaises(ValueError):
            trajectory_route(path)

    def test_m0_uses_its_own_midline(self):
        # On M0 the midline is column 5 and the arms are rows 1 and 5.
        dip = through((2, 1), (1, 1), (1, 4), (1, 1), (5, 1), (5, 9), (2, 9), grid=M0)
        self.assertEqual(trajectory_route(dip, "M0"), "south")
        across = through((2, 1), (1, 1), (1, 9), (2, 9), grid=M0)
        self.assertEqual(trajectory_route(across, "M0"), "north")
        with self.assertRaises(KeyError):
            trajectory_route(across, "M2")  # no ring

    def test_velocity_columns_are_ignored(self):
        path = through((2, 1), (1, 1), (1, 7), (2, 7))
        with_vel = np.hstack([path, np.full((len(path), 2), 99.0)])
        self.assertEqual(trajectory_route(with_vel), "north")


if __name__ == "__main__":
    unittest.main()
