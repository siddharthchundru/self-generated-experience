"""Checks for the data-diversity log (src/data_diversity.py).

Run with:  python -m unittest discover -s tests

What they establish:
1. Binning: only free space counts, a position in a wall raises, and coverage
   of a single bin is 1 / free_bins.
2. Spread: spreading the same number of states over more bins raises both
   coverage and normalised entropy; D0 counts in the buffer but not in the
   collected-only scores.
3. Information added: repeating an episode adds no new bins and is less
   surprising the second time; an episode somewhere new adds bins and is
   more surprising than one where the buffer is already dense.
4. Route mix: hand-built crossings are counted by arm and direction, and the
   minority-arm share is computed per direction. On real expert episodes the
   arm tallies match the route the expert was told to take.
5. Order: demonstrations cannot be added after collection has started.
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_diversity import DataDiversityLog  # noqa: E402
from envs.mazes import M1  # noqa: E402
from envs.pointmaze import cell_to_xy, make_env  # noqa: E402
from expert import run_episode  # noqa: E402


def through(*cells, points=5):
    """Positions along straight segments joining M1 cell centres."""
    pts = [cell_to_xy(M1, c) for c in cells]
    out = []
    for a, b in zip(pts, pts[1:]):
        for f in np.linspace(0, 1, points, endpoint=False):
            out.append(a + f * (b - a))
    out.append(pts[-1])
    return np.array(out)


NORTH_W_TO_E = [(4, 1), (3, 1), (2, 1), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
                (1, 6), (1, 7), (2, 7), (3, 7), (4, 7)]
SOUTH_W_TO_E = [(4, 1), (5, 1), (6, 1), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5),
                (7, 6), (7, 7), (6, 7), (5, 7), (4, 7)]


class TestBinning(unittest.TestCase):
    def test_free_bins_and_single_bin_coverage(self):
        log = DataDiversityLog("M1")
        self.assertEqual(log.n_free, 35 * 16)       # 35 free cells on M1, 4 x 4 bins each
        log.add_episode(np.tile(cell_to_xy(M1, (1, 1)), (10, 1)))
        self.assertAlmostEqual(log.summary()["collected"]["coverage"], 1 / log.n_free)
        self.assertEqual(log.summary()["collected"]["entropy_normalised"], 0.0)

    def test_wall_position_raises(self):
        log = DataDiversityLog("M1")
        with self.assertRaises(ValueError):
            log.add_episode(np.array([cell_to_xy(M1, (3, 3))]))   # inside the central block


class TestSpread(unittest.TestCase):
    def test_spread_raises_coverage_and_entropy(self):
        narrow, wide = DataDiversityLog("M1"), DataDiversityLog("M1")
        corner = cell_to_xy(M1, (1, 1))
        narrow.add_episode(np.tile(corner, (len(NORTH_W_TO_E) * 5 - 4, 1)))
        wide.add_episode(through(*NORTH_W_TO_E))
        n, w = narrow.summary()["collected"], wide.summary()["collected"]
        self.assertEqual(n["states"], w["states"])
        self.assertGreater(w["coverage"], n["coverage"])
        self.assertGreater(w["entropy_normalised"], n["entropy_normalised"])
        self.assertLessEqual(w["entropy_normalised"], 1.0)

    def test_demonstrations_count_in_buffer_only(self):
        log = DataDiversityLog("M1")
        log.add_demonstrations([through(*NORTH_W_TO_E)])
        log.add_episode(np.tile(cell_to_xy(M1, (4, 1)), (5, 1)))
        s = log.summary()
        self.assertGreater(s["all"]["coverage"], s["collected"]["coverage"])
        self.assertEqual(s["collected"]["states"], 5)
        self.assertEqual(s["collected"]["crossings"]["west_to_east"]["north"], 0)
        self.assertEqual(s["all"]["crossings"]["west_to_east"]["north"], 1)


class TestInformationAdded(unittest.TestCase):
    def test_repeat_adds_nothing_new_and_is_less_surprising(self):
        log = DataDiversityLog("M1")
        first = log.add_episode(through(*NORTH_W_TO_E))
        second = log.add_episode(through(*NORTH_W_TO_E))
        self.assertGreater(first["new_bins"], 0)
        self.assertEqual(second["new_bins"], 0)
        self.assertLess(second["surprise_bits"], first["surprise_bits"])

    def test_new_region_more_surprising_than_dense_region(self):
        log = DataDiversityLog("M1")
        log.add_demonstrations([through(*NORTH_W_TO_E)] * 5)
        old = log.add_episode(through(*NORTH_W_TO_E))
        new = log.add_episode(through(*SOUTH_W_TO_E))
        self.assertGreater(new["new_bins"], old["new_bins"])
        self.assertGreater(new["surprise_bits"], old["surprise_bits"])

    def test_surprise_of_first_episode_is_uniform_cost(self):
        log = DataDiversityLog("M1")
        record = log.add_episode(through(*NORTH_W_TO_E))
        self.assertAlmostEqual(record["surprise_bits"], math.log2(log.n_free))


class TestRouteMix(unittest.TestCase):
    def test_crossings_by_arm_and_direction(self):
        log = DataDiversityLog("M1")
        for _ in range(3):
            log.add_episode(through(*NORTH_W_TO_E))
        log.add_episode(through(*SOUTH_W_TO_E))
        log.add_episode(through(*reversed(SOUTH_W_TO_E)))
        c = log.summary()["collected"]["crossings"]
        self.assertEqual(c["west_to_east"]["north"], 3)
        self.assertEqual(c["west_to_east"]["south"], 1)
        self.assertAlmostEqual(c["west_to_east"]["minority_arm_share"], 0.25)
        self.assertEqual(c["east_to_west"], {"north": 0, "south": 1, "minority_arm_share": 0.0})

    def test_no_crossings_is_undefined_not_zero(self):
        log = DataDiversityLog("M1")
        log.add_episode(np.tile(cell_to_xy(M1, (4, 1)), (5, 1)))
        self.assertIsNone(log.summary()["collected"]["crossings"]["west_to_east"]["minority_arm_share"])

    def test_expert_episodes_tally_the_commanded_arm(self):
        env = make_env("M1")
        log = DataDiversityLog("M1")
        for seed, (row, route) in enumerate([(2, "north"), (2, "south"), (6, "south")]):
            ep = run_episode(env, M1, (row, 1), (row, 7), route=route, seed=seed)
            log.add_episode(ep["observations"])
        env.close()
        c = log.summary()["collected"]["crossings"]["west_to_east"]
        self.assertEqual((c["north"], c["south"]), (1, 2))


class TestOrder(unittest.TestCase):
    def test_demonstrations_after_collection_raise(self):
        log = DataDiversityLog("M1")
        log.add_episode(through(*NORTH_W_TO_E))
        with self.assertRaises(ValueError):
            log.add_demonstrations([through(*NORTH_W_TO_E)])


if __name__ == "__main__":
    unittest.main()
