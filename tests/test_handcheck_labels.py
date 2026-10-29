"""Protect independently supplied labels against incomplete or ambiguous import."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_route_labels import parse_answers


class TestHumanLabels(unittest.TestCase):
    def test_uncertainty_is_preserved(self):
        result = parse_answers("1N 2X? 3S", [1, 2, 3])
        self.assertTrue(result[2]["unsure"])
        self.assertEqual(result[2]["route"], "neither")
        self.assertFalse(result[1]["unsure"])

    def test_incomplete_and_duplicate_lines_are_rejected(self):
        for text in ("1N", "1N 2-", "1N 1S", "1N 2X 3S"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_answers(text, [1, 2])


if __name__ == "__main__":
    unittest.main()
