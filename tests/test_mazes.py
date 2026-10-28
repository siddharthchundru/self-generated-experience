"""Planner checks for the maze layouts (the Stage 1 gate for M1).

Run with:  python -m unittest discover -s tests

These tests need no simulator. They prove the property the repertoire endpoint
depends on: in M1, west-east pairs in the demo region are joined by exactly two
routes (north arm, south arm), and which one is shorter depends on the row.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from envs.mazes import (  # noqa: E402
    M0, M1, M2, DEMO_COLS, NORTH_ARM, SOUTH_ARM, M1_DEMO_REGION, M1_HELD_OUT_REGION,
    M0_DEMO_REGION, M0_HELD_OUT_REGION, M0_NORTH_ARM, M0_SOUTH_ARM,
    free_cells, shortest_path, cycle_rank, route_class,
)


class TestM1(unittest.TestCase):
    def test_exactly_one_loop(self):
        # One loop means exactly two topologically distinct routes around it.
        free = free_cells(M1)
        ring = {c for c in free if c[1] in DEMO_COLS}
        self.assertEqual(cycle_rank(ring), 1)
        self.assertEqual(cycle_rank(free), 1)  # the held-out region adds no loops

    def test_upper_pairs_prefer_north(self):
        for row in (1, 2, 3):
            path = shortest_path(M1, (row, 1), (row, 7))
            self.assertEqual(route_class(path), "north")

    def test_lower_pairs_prefer_south(self):
        for row in (5, 6, 7):
            path = shortest_path(M1, (row, 1), (row, 7))
            self.assertEqual(route_class(path), "south")

    def test_middle_row_is_a_tie(self):
        # Row 4 is equidistant by design; it is excluded from "shorter route" pairs.
        north = shortest_path(M1, (4, 1), (4, 7), blocked=SOUTH_ARM)
        south = shortest_path(M1, (4, 1), (4, 7), blocked=NORTH_ARM)
        self.assertEqual(len(north), len(south))

    def test_both_routes_survive_one_block(self):
        # Blocking either arm (perturbation P1) must leave the other route open.
        for row in range(1, 8):
            self.assertIsNotNone(shortest_path(M1, (row, 1), (row, 7), blocked=NORTH_ARM))
            self.assertIsNotNone(shortest_path(M1, (row, 1), (row, 7), blocked=SOUTH_ARM))

    def test_held_out_region_reached_only_through_corridor(self):
        # Every path from the ring to the held-out region passes (4, 8).
        path = shortest_path(M1, (1, 1), (7, 9), blocked=frozenset({(4, 8)}))
        self.assertIsNone(path)


class TestM1Pinned(unittest.TestCase):
    """Pin M1's layout so an accidental edit (a flip, a moved corridor) fails.

    The ring is symmetric top to bottom, so the structural tests above would
    still pass if M1 were flipped; the held-out region is not symmetric.
    """

    def test_held_out_region_cells(self):
        expected = {(4, 8), (4, 9), (4, 10), (4, 11), (4, 12), (5, 12),
                    (6, 9), (6, 10), (6, 11), (6, 12), (7, 9)}
        self.assertEqual(set(M1_HELD_OUT_REGION), expected)
        self.assertEqual(len(M1_DEMO_REGION), 24)


class TestM0(unittest.TestCase):
    """The tuning maze has M1's two-route structure with a different shape."""

    def route_lengths(self, row):
        north = shortest_path(M0, (row, 1), (row, 9), blocked=M0_SOUTH_ARM)
        south = shortest_path(M0, (row, 1), (row, 9), blocked=M0_NORTH_ARM)
        return len(north) - 1, len(south) - 1

    def test_exactly_one_loop(self):
        self.assertEqual(cycle_rank(set(M0_DEMO_REGION)), 1)
        self.assertEqual(cycle_rank(free_cells(M0)), 1)  # held-out region adds no loops

    def test_which_route_is_shorter_depends_on_the_row(self):
        # (north, south) lengths in cells; row 3 is the tie.
        expected = {1: (8, 16), 2: (10, 14), 3: (12, 12), 4: (14, 10), 5: (16, 8)}
        for row, lengths in expected.items():
            self.assertEqual(self.route_lengths(row), lengths)

    def test_geometry_differs_from_m1(self):
        # Not a copy or flip of M1: different shape and different route gaps.
        self.assertNotEqual((len(M0), len(M0[0])), (len(M1), len(M1[0])))
        m1_gaps = {abs(len(shortest_path(M1, (r, 1), (r, 7), blocked=NORTH_ARM))
                       - len(shortest_path(M1, (r, 1), (r, 7), blocked=SOUTH_ARM)))
                   for r in range(1, 8)}
        m0_gaps = {abs(a - b) for a, b in map(self.route_lengths, range(1, 6))}
        self.assertNotEqual(m0_gaps, m1_gaps)

    def test_regions(self):
        self.assertEqual(len(M0_DEMO_REGION), 24)  # same size as M1's ring
        expected = {(3, 10), (3, 11), (2, 11), (1, 11), (1, 12), (1, 13),
                    (2, 13), (3, 13), (4, 13), (5, 13), (5, 12), (5, 11)}
        self.assertEqual(set(M0_HELD_OUT_REGION), expected)
        # Reached only through the corridor cell (3, 10).
        self.assertIsNone(shortest_path(M0, (1, 1), (5, 11), blocked=frozenset({(3, 10)})))

    def test_both_routes_survive_one_block(self):
        for row in range(1, 6):
            for arm in (M0_NORTH_ARM, M0_SOUTH_ARM):
                self.assertIsNotNone(shortest_path(M0, (row, 1), (row, 9), blocked=arm))


class TestM2(unittest.TestCase):
    def test_is_the_library_medium_maze(self):
        # M2 must be the standard layout, unmodified, for comparability.
        from gymnasium_robotics.envs.maze.maps import MEDIUM_MAZE
        self.assertEqual(M2, MEDIUM_MAZE)

    def test_standard_medium_has_loops(self):
        # Sanity check that M2 is the multi-loop standard layout, not a corridor.
        self.assertGreater(cycle_rank(free_cells(M2)), 1)


if __name__ == "__main__":
    unittest.main()
