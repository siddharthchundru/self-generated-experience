"""Check endpoint denominators, missingness, budgets and real-wall evaluation.

Analytic examples protect scientific definitions independently of the simulator.
Small integration runs check policy isolation and one TOTAL adaptation budget.
TestAuditInterface checks the fixes for the 28 Oct audit's findings S1 to S3:
the suite counts adaptation steps itself, gradient steps must equal the ratio
times k and the optimiser's own count, adaptation starts from an empty buffer
with no data on the policy, evaluation leaves parameters unchanged, and a
learned policy never receives the task pair or the wall.
"""

import copy
import sys
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval.metrics import best_of_m, common_pair_rep, summarise_rollouts
from eval.policies import ExpertPolicy, RandomPolicy
from eval.protocol import EvaluationConfig, TaskPair, episode_seed
from episode import EPISODE_STEPS
from eval.suite import (AdaptationResult, BudgetExhausted, NoUpdateAdapter, check_parameters_only,
                        collect, evaluate_policy, no_update_adaptation)

import numpy as np
import torch


def records(north, south, failures):
    return ([{"route": "north", "success": True, "first_success_step": 10}] * north
            + [{"route": "south", "success": True, "first_success_step": 30}] * south
            + [{"route": "neither", "success": False, "first_success_step": None}] * failures)


class TestEndpointDefinitions(unittest.TestCase):
    def test_competence_is_not_confused_with_conditional_share(self):
        good = summarise_rollouts(records(45, 45, 10))
        poor = summarise_rollouts(records(1, 1, 98))
        self.assertEqual(good["Rep"], 0.5)
        self.assertEqual(good["minority_route_success_rate"], 0.45)
        self.assertIsNone(poor["Rep"])
        self.assertEqual(poor["minority_route_success_rate"], 0.01)
        self.assertEqual(poor["Rep_excluded_as_zero"], 0)

    def test_zero_successes_are_undefined_rep_not_zero(self):
        s = summarise_rollouts(records(0, 0, 100))
        self.assertIsNone(s["Rep"])
        self.assertIsNone(s["Rep_bootstrap_95"])
        self.assertEqual(s["minority_route_success_rate"], 0)

    def test_threshold_boundaries_and_route_quality_include_failures(self):
        data = records(89, 11, 0) + [{"route": "south", "success": False, "first_success_step": None}]*10
        s = summarise_rollouts(data)
        self.assertEqual(s["route_class_counts"], {"0.05": 2, "0.1": 2, "0.2": 1})
        self.assertEqual(s["per_route_quality"]["south"]["success_rate_given_route"], 11/21)
        self.assertEqual(s["per_route_quality"]["south"]["first_success_seconds_mean"], 3)
        self.assertEqual(s["Rep"], 0.11)
        self.assertEqual(s["minority_route_success_rate"], 11/110)

    def test_success_without_a_crossing_is_an_error(self):
        with self.assertRaises(ValueError):
            summarise_rollouts([{"route": "neither", "success": True}])

    def test_common_pair_comparison_has_no_survivor_set_mismatch(self):
        left = {1: {"Rep": 0.3}, 2: {"Rep": None}, 3: {"Rep": 0.2}}
        right = {1: {"Rep": 0.1}, 2: {"Rep": 0.4}, 3: {"Rep": None}}
        s = common_pair_rep(left, right)
        self.assertEqual(s["common_rows"], [1])
        self.assertAlmostEqual(s["left_minus_right"], 0.2)
        self.assertEqual(s["dropped_rows"], [2, 3])
        self.assertIsNone(common_pair_rep({1: {"Rep": None}}, {1: {"Rep": 0.4}})["left_minus_right"])

    def test_best_of_uses_real_disjoint_groups(self):
        data = [{"success": s} for s in (True, False, False, False, True, False)]
        self.assertEqual(best_of_m(data, 2)["success_rate"], 2/3)
        with self.assertRaises(ValueError):
            best_of_m(data, 4)


class TestProtocol(unittest.TestCase):
    def test_fixed_blocks_have_equal_geometry_across_policies(self):
        self.assertEqual(TaskPair("M1", 1).wall("north"), (1, 4))
        self.assertEqual(TaskPair("M1", 7).wall("south"), (7, 4))
        self.assertIsNone(TaskPair("M1", 4).shorter)
        self.assertEqual(TaskPair("M0", 2).wall("north"), (1, 5))

    def test_bad_budgets_and_incomplete_groups_fail(self):
        with self.assertRaises(ValueError):
            EvaluationConfig(adaptation_steps=101)
        with self.assertRaises(ValueError):
            EvaluationConfig(best_of_m=3)
        with self.assertRaises(ValueError):
            EvaluationConfig(horizon=100)  # T is project-wide, not per evaluation
        with self.assertRaises(ValueError):
            EvaluationConfig(adaptation_updates_per_env_step=1/7)  # not whole updates per episode

    def test_phase_and_row_seeds_are_independent_and_repeatable(self):
        self.assertEqual(episode_seed(1, 2, 3, 4), episode_seed(1, 2, 3, 4))
        self.assertNotEqual(episode_seed(1, 2, 3, 4), episode_seed(1, 2, 4, 4))

    def test_long_adaptation_indices_do_not_change_the_seed_phase(self):
        config = EvaluationConfig()
        pair = TaskPair("M1", 2)
        record = collect(RandomPolicy(), pair, 1, config, 1000, True, index_offset=500)[0]
        self.assertEqual(record["seed"], episode_seed(config.seed, 2, 1000, 500))
        self.assertNotEqual(record["seed"], episode_seed(config.seed, 2, 1500, 0))


class TestEndToEnd(unittest.TestCase):
    config = EvaluationConfig(sampled_rollouts=10, greedy_rollouts=2, minimum_successes=2,
                              bootstrap_resamples=50, adaptation_steps=3*EPISODE_STEPS)

    def test_real_wall_expert_and_total_adaptation_budget(self):
        report = evaluate_policy(ExpertPolicy(), self.config, no_update_adaptation)
        self.assertEqual(report["aggregate"]["S_in"], 1)
        self.assertEqual(len(report["aggregate"]["Rep_qualifying_rows"]), 6)
        self.assertEqual(report["aggregate"]["primary_rows"], [1, 2, 3, 5, 6, 7])
        for run in report["adaptation_runs"].values():
            self.assertEqual(run["environment_steps"], 3*EPISODE_STEPS)  # NOT 6 times that
            self.assertEqual(run["gradient_steps"], 0)
        for phase in ("P1", "P1_own"):
            collected = [r for r in report["rollouts"] if r["phase"] == phase+"_adaptation_collection"]
            self.assertEqual(sum(r["steps"] for r in collected), 3*EPISODE_STEPS)
        # Expert ignores the wall: greedy insists on its original shorter route.
        self.assertEqual(report["aggregate"]["A_k"], 0)
        self.assertGreater(report["aggregate"]["A_0"], 0)
        self.assertLess(report["aggregate"]["A_0"], 1)

    def test_missing_learner_is_not_silently_replaced_by_learning(self):
        report = evaluate_policy(RandomPolicy(), self.config)
        self.assertIsNone(report["aggregate"]["A_k"])
        self.assertEqual(report["aggregate"]["A_k_status"], "not_run")

    def test_adapter_receives_independent_policy_copies(self):
        calls = []
        policy = ExpertPolicy()

        class Recorder(NoUpdateAdapter):
            def adapt(inner, clone, session, buffer):
                calls.append(clone)
                return NoUpdateAdapter.adapt(inner, clone, session, buffer)

        evaluate_policy(policy, self.config, Recorder())
        self.assertEqual(len(calls), 2)
        self.assertIsNot(calls[0], policy)
        self.assertIsNot(calls[0], calls[1])

    def test_m0_cli_gate_uses_its_four_primary_pairs_not_m1_six(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"m0.json"
            result = subprocess.run([sys.executable, "scripts/evaluate_stage1.py", "--quick",
                                     "--maze", "M0", "--output", str(output)],
                                    cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            report = json.loads(output.read_text())
            self.assertTrue(report["evaluation_pipeline_passes"])
            self.assertEqual(report["reports"][0]["aggregate"]["primary_rows"], [1, 2, 4, 5])


class StubPolicy:
    """A minimal learned policy: one linear layer, records every argument it receives."""

    def __init__(self, name="stub"):
        self.name = name
        torch.manual_seed(0)
        self.network = torch.nn.Linear(6, 2)

    def start_episode(self, view, rng, sampled):
        StubPolicy.seen.append(("start", view))

        def act(current):
            StubPolicy.seen.append(("act", current))
            return rng.uniform(-1, 1, size=2).astype(np.float32)
        return act

    def parameter_digest(self):
        return hash(tuple(p.detach().numpy().tobytes() for p in self.network.parameters()))


class CountingAdapter:
    """A learned-mode adapter that makes `steps` real optimiser steps after collecting `episodes`."""

    def __init__(self, episodes, steps, report=None, ratio=1.0, return_optimizer=True):
        self.episodes, self.steps, self.report = episodes, steps, report
        self.updates_per_env_step, self.return_optimizer = ratio, return_optimizer

    def new_buffer(self):
        return []

    def adapt(self, policy, session, buffer):
        for _ in range(self.episodes):
            buffer.append(session.run_episode())
        optimizer = torch.optim.Adam(policy.network.parameters(), lr=1e-3)
        for _ in range(self.steps):
            optimizer.zero_grad()
            policy.network(torch.ones(1, 6)).sum().backward()
            optimizer.step()
        return AdaptationResult(policy, self.steps if self.report is None else self.report,
                                "test_learned", optimizer if self.return_optimizer else None)


class TestAuditInterface(unittest.TestCase):
    """Audit of 28 Oct: S1 (budgets counted by the suite), S2 (no old data, no hidden
    updates), S3 (observation and goal only)."""

    k = 2*EPISODE_STEPS
    config = EvaluationConfig(sampled_rollouts=5, greedy_rollouts=1, minimum_successes=1,
                              bootstrap_resamples=10, best_of_m=5, adaptation_steps=k)

    def setUp(self):
        StubPolicy.seen = []

    # S3 -------------------------------------------------------------------------
    def test_learned_policy_sees_observation_and_goal_only(self):
        evaluate_policy(StubPolicy(), self.config, CountingAdapter(2, self.k))
        self.assertTrue(StubPolicy.seen)
        for kind, view in StubPolicy.seen:
            self.assertEqual(set(view), {"observation", "desired_goal"}, kind)
            self.assertEqual(view["observation"].shape, (4,))

    def test_session_returns_trajectories_without_task_or_wall(self):
        seen = {}

        class Peek(CountingAdapter):
            def adapt(inner, policy, session, buffer):
                seen["public"] = {n for n in dir(session) if not n.startswith("_")}
                trajectory = session.run_episode()
                seen["keys"] = set(trajectory)
                seen["shapes"] = (trajectory["observations"].shape, trajectory["actions"].shape)
                buffer.append(trajectory)
                buffer.append(session.run_episode())
                return CountingAdapter.adapt(CountingAdapter(0, self.k), policy, session, buffer)

        evaluate_policy(StubPolicy(), self.config, Peek(0, 0))
        self.assertEqual(seen["keys"], {"observations", "actions", "goal"})
        self.assertEqual(seen["shapes"], ((EPISODE_STEPS+1, 4), (EPISODE_STEPS, 2)))
        self.assertFalse(seen["public"] & {"tasks", "pair", "wall", "walls"})

    # S1 -------------------------------------------------------------------------
    def test_adapter_cannot_exceed_the_budget(self):
        with self.assertRaises(BudgetExhausted):
            evaluate_policy(StubPolicy(), self.config, CountingAdapter(3, self.k))

    def test_adapter_must_use_the_whole_budget(self):
        with self.assertRaisesRegex(AssertionError, "budget must be used exactly"):
            evaluate_policy(StubPolicy(), self.config, CountingAdapter(1, self.k))

    def test_misreported_gradient_steps_are_caught(self):
        with self.assertRaisesRegex(AssertionError, "ratio x k"):
            evaluate_policy(StubPolicy(), self.config, CountingAdapter(2, self.k, report=123456))
        with self.assertRaisesRegex(AssertionError, "optimiser took"):
            evaluate_policy(StubPolicy(), self.config, CountingAdapter(2, 5, report=self.k))
        with self.assertRaisesRegex(AssertionError, "optimiser"):
            evaluate_policy(StubPolicy(), self.config, CountingAdapter(2, self.k, return_optimizer=False))

    def test_adapter_ratio_must_match_the_configured_ratio(self):
        with self.assertRaisesRegex(AssertionError, "update ratio"):
            evaluate_policy(StubPolicy(), self.config, CountingAdapter(2, self.k, ratio=2.0))

    def test_only_the_no_update_stand_in_may_claim_stand_in_mode(self):
        class Liar(CountingAdapter):
            def adapt(inner, policy, session, buffer):
                result = CountingAdapter.adapt(inner, policy, session, buffer)
                return AdaptationResult(policy, 0, "standin_no_parameter_updates")
        with self.assertRaisesRegex(AssertionError, "standin_"):
            evaluate_policy(StubPolicy(), self.config, Liar(2, 0))

    def test_adapter_must_return_the_copy_it_was_given(self):
        class Swap(CountingAdapter):
            def adapt(inner, policy, session, buffer):
                result = CountingAdapter.adapt(inner, policy, session, buffer)
                result.policy = copy.deepcopy(policy)
                return result
        with self.assertRaisesRegex(AssertionError, "copy it was given"):
            evaluate_policy(StubPolicy(), self.config, Swap(2, self.k))

    def test_suite_counts_match_the_report(self):
        report = evaluate_policy(StubPolicy(), self.config, CountingAdapter(2, self.k))
        for run in report["adaptation_runs"].values():
            if run["status"].startswith("not_run"):
                continue
            self.assertEqual(run["environment_steps"], self.k)
            self.assertEqual(run["gradient_steps"], self.k)
            self.assertEqual(run["optimizer_steps"], self.k)
            self.assertEqual((run["buffer_size_at_start"], run["buffer_size_at_end"]), (0, 2))

    # S2 -------------------------------------------------------------------------
    def test_adaptation_buffer_must_start_empty(self):
        class Prefilled(CountingAdapter):
            def new_buffer(inner):
                return ["old data"]
        with self.assertRaisesRegex(AssertionError, "empty buffer"):
            evaluate_policy(StubPolicy(), self.config, Prefilled(2, self.k))

    def test_policy_carrying_data_or_an_optimiser_is_refused(self):
        policy = StubPolicy()
        policy.buffer = [np.zeros(3)]
        with self.assertRaisesRegex(TypeError, "parameters only"):
            check_parameters_only(policy)
        policy = StubPolicy()
        policy.optimizer = torch.optim.Adam(policy.network.parameters())
        with self.assertRaisesRegex(TypeError, "parameters only"):
            check_parameters_only(policy)
        policy = StubPolicy()
        policy.network.replay = [1, 2, 3]  # hidden inside the network module
        with self.assertRaisesRegex(TypeError, "parameters only"):
            check_parameters_only(policy)

    def test_evaluation_that_changes_parameters_is_caught(self):
        class Drifting(StubPolicy):
            def start_episode(self, view, rng, sampled):
                with torch.no_grad():
                    self.network.bias.add_(1.0)
                return StubPolicy.start_episode(self, view, rng, sampled)
        with self.assertRaisesRegex(AssertionError, "parameters changed"):
            evaluate_policy(Drifting(), self.config)

    def test_adaptation_leaves_the_original_policy_unchanged(self):
        policy = StubPolicy()
        before = policy.parameter_digest()
        report = evaluate_policy(policy, self.config, CountingAdapter(2, self.k))
        self.assertEqual(policy.parameter_digest(), before)
        self.assertEqual(report["parameter_digest"], before)
        self.assertEqual(report["policy_kind"], "learned_observation_and_goal_only")


if __name__ == "__main__":
    unittest.main()
