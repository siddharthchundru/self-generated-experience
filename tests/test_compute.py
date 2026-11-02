"""Check compute accounting, including C0, re-learning and two shared k budgets.

Budgets are whole 60-step episodes: 99,960 steps is 1,666 episodes, and k at
10% of it rounds down to 166 episodes (9,960 steps).
"""

from dataclasses import replace
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eval.compute import ComputeAssumptions, counts, estimate_runtime


class TestComputeAccounting(unittest.TestCase):
    def setUp(self):
        self.a = ComputeAssumptions(pilot_seeds=0, tuning_trials_per_condition=0,
            capacity_demo_episodes=0, capacity_updates_per_learner_maze=0,
            capacity_eval_episodes_per_learner_maze=0)

    def test_main_counts_include_no_interactions_for_c0(self):
        report = counts(self.a, 99960, 1)
        components = {c["name"]: c for c in report["components"]}
        practice = components["main: main practice"]
        self.assertEqual(practice["random_dynamics_steps"]+practice["expert_steps"], 6*99960)
        self.assertEqual(practice["gradient_updates"], 7*99960)
        adaptation = components["main: P1 and P1-own adaptation"]
        self.assertEqual(adaptation["random_dynamics_steps"], 7*2*9960)  # not times six pairs
        relearning = components["main: fresh-buffer re-learning"]
        self.assertEqual(relearning["gradient_updates"], 5*(10000+99960))

    def test_optional_topup_removes_one_run_and_one_relearning(self):
        report = counts(replace(self.a, include_expert_topup=False), 99960, 1)
        self.assertEqual(report["main_conditions"], 6)
        self.assertEqual(report["relearning_conditions"], 4)
        practice = next(c for c in report["components"] if c["name"] == "main: main practice")
        self.assertEqual(practice["expert_steps"], 0)
        self.assertEqual(practice["random_dynamics_steps"], 5*99960)

    def test_runtime_is_sum_of_all_measured_operations(self):
        rate = {"median_per_second": 10, "slowest_per_second": 5}
        measured = {"simulator": {"random_actions": rate, "expert_episodes": rate},
                    "network": {"synthetic_updates": rate, "sampled_batch1_inference": rate},
                    "selector_surrogate": {"selections": rate}}
        operation_counts = {"totals": {k: 100 for k in (
            "random_dynamics_steps", "expert_steps", "gradient_updates", "network_inferences", "frontier_selections")}}
        self.assertAlmostEqual(estimate_runtime(operation_counts, measured)["hours_before_overhead"], 50/3600)
        self.assertAlmostEqual(estimate_runtime(operation_counts, measured, True)["hours_before_overhead"], 100/3600)


if __name__ == "__main__":
    unittest.main()
