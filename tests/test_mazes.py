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
    M1, M2, DEMO_COLS, NORTH_ARM, SOUTH_ARM,
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


class TestM2(unittest.TestCase):
    def test_standard_medium_has_loops(self):
        # Sanity check that M2 is the multi-loop standard layout, not a corridor.
        self.assertGreater(cycle_rank(free_cells(M2)), 1)


if __name__ == "__main__":
    unittest.main()
