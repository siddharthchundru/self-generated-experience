"""Check the training runner (spec sections 5 to 7; v0.12 optimiser rules).

What these tests protect (tiny settings: they check the code, not results):
- every condition in a seed starts practice from the same pretrained policy;
- the budget: collecting conditions take exactly N environment steps, every
  condition takes exactly N x ratio practice gradient steps, demonstrations only
  takes the same with no environment steps, and no optimiser step happens
  outside warm-up and practice;
- Adam is reset when practice starts (its step count covers practice only);
- the optional learning-rate warm-up ramps linearly and is off by default;
- the interleaved schedule takes ratio x T updates per episode for every ratio
  in the tuning grid;
- the validation warm-up rule stops inside its bounds;
- settings that break agreed rules are refused (plain learner outside its two
  conditions, partial episodes, the k placeholder for pilot or main runs);
- the adapter and evaluation settings come from the run's settings (audit SF4);
- the script refuses to write into results/ from a dirty tree, and writes its
  files elsewhere.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from collection import CONDITIONS, DEMOS_ONLY, FRONTIER, TASK_DIRECTED  # noqa: E402
from episode import EPISODE_STEPS  # noqa: E402
from gcsl import GAMMA_PLAIN, GAMMA_WEIGHTED, GCSLLearner, GCSLPolicy, TrajectoryBuffer  # noqa: E402
from training import (K_PLACEHOLDER, TrainingSettings, _interleaved_due, adapter_for,  # noqa: E402
                      evaluation_config_for, run_training)

T = EPISODE_STEPS
TINY = {"warmup_updates": 30, "demos_per_pair": 2, "budget_steps": 2 * T, "batch_size": 32}


def tiny(condition, **overrides):
    return TrainingSettings(condition=condition, **{**TINY, **overrides})


class TestRuns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {c: run_training(tiny(c)) for c in CONDITIONS}

    def test_same_pretrained_policy_in_every_condition(self):
        digests = {r["summary"]["pretrained_digest"] for r in self.results.values()}
        self.assertEqual(len(digests), 1)
        finals = {r["summary"]["final_digest"] for r in self.results.values()}
        self.assertEqual(len(finals), len(CONDITIONS))  # practice changed each differently

    def test_budget_is_matched_across_conditions(self):
        for condition, result in self.results.items():
            budget = result["summary"]["budget"]
            self.assertTrue(budget["passed"])
            self.assertEqual(budget["practice_gradient_steps"], 2 * T)
            self.assertEqual(budget["warmup_gradient_steps"], 30)
            self.assertEqual(budget["all_optimizer_steps_in_process"], 30 + 2 * T)
            expected_env = 0 if condition == DEMOS_ONLY else 2 * T
            self.assertEqual(budget["practice_environment_steps"], expected_env, condition)
            self.assertEqual(len(result["episodes"]), 2)

    def test_adam_is_reset_when_practice_starts(self):
        for result in self.results.values():
            self.assertEqual(result["summary"]["budget"]["practice_optimizer_steps"], 2 * T)

    def test_logs_carry_exposure_and_diversity(self):
        row = self.results[TASK_DIRECTED]["episodes"][0]
        for key in ("fork_visits", "new_bins", "surprise_bits", "batches", "environment_steps", "gradient_steps"):
            self.assertIn(key, row)
        self.assertEqual(row["batches"]["updates"], T)
        self.assertGreater(row["batches"]["effective_sample_size"], 0)
        summary = self.results[FRONTIER]["summary"]
        self.assertEqual(summary["data_diversity"]["collected_episodes"], 2)
        self.assertEqual(self.results[DEMOS_ONLY]["summary"]["data_diversity"]["collected_episodes"], 0)

    def test_held_out_goals_only_under_frontier(self):
        for condition, result in self.results.items():
            if condition != FRONTIER:
                self.assertEqual(result["summary"]["commanded_held_out_goals"], 0)


class TestSchedules(unittest.TestCase):
    def test_interleaved_due_sums_to_ratio_times_t(self):
        for ratio in (0.5, 1.0, 2.0):
            due = _interleaved_due(ratio)
            self.assertEqual(sum(due.values()), int(ratio * T))
            self.assertEqual(set(due), set(range(1, T + 1)))

    def test_interleaved_run_matches_budget(self):
        for ratio in (0.5, 2.0):
            result = run_training(tiny(TASK_DIRECTED, schedule="interleaved", updates_per_env_step=ratio,
                                       budget_steps=T))
            self.assertEqual(result["summary"]["budget"]["practice_gradient_steps"], int(ratio * T))


class TestLearningRateWarmup(unittest.TestCase):
    def test_linear_ramp_and_off_by_default(self):
        buffer = TrajectoryBuffer()
        rng = np.random.default_rng(0)
        obs = np.zeros((T + 1, 4), dtype=np.float32)
        buffer.add({"observations": obs, "actions": np.zeros((T, 2)), "goal": np.zeros(2)}, "test")
        learner = GCSLLearner(GCSLPolicy(0), GAMMA_WEIGHTED, 1e-3, 8, lr_warmup_steps=4)
        rates = []
        for _ in range(6):
            rates.append(learner.current_learning_rate())
            learner.update(buffer, rng)
            self.assertAlmostEqual(learner.optimizer.param_groups[0]["lr"], rates[-1])
        np.testing.assert_allclose(rates, [2.5e-4, 5e-4, 7.5e-4, 1e-3, 1e-3, 1e-3])
        self.assertEqual(GCSLLearner(GCSLPolicy(0), 1.0, 1e-3).current_learning_rate(), 1e-3)
        self.assertEqual(TrainingSettings(condition=TASK_DIRECTED).lr_warmup_steps, 0)

    def test_runner_applies_ramp_to_practice_only(self):
        result = run_training(tiny(TASK_DIRECTED, lr_warmup_steps=100, budget_steps=T))
        self.assertAlmostEqual(result["episodes"][0]["learning_rate"], 1e-3 * 61 / 100)


class TestValidationWarmup(unittest.TestCase):
    def test_stops_within_bounds(self):
        settings = tiny(TASK_DIRECTED, warmup_rule="validation", warmup_min_updates=40, warmup_max_updates=100,
                        validation_check_every=20, validation_window=40, validation_examples=64,
                        validation_per_pair=2, budget_steps=0)
        result = run_training(settings)
        warmup = result["summary"]["warmup"]
        self.assertTrue(40 <= warmup["updates"] <= 100)
        self.assertEqual(warmup["updates"] % 20, 0)
        self.assertEqual([c["updates"] for c in warmup["checks"]], list(range(20, warmup["updates"] + 1, 20)))


class TestSettings(unittest.TestCase):
    def test_refusals(self):
        with self.assertRaises(ValueError):
            TrainingSettings(condition="random_actions", learner="plain")
        with self.assertRaises(ValueError):
            TrainingSettings(condition=TASK_DIRECTED, budget_steps=250_000)
        with self.assertRaises(ValueError):
            TrainingSettings(condition=TASK_DIRECTED, purpose="pilot")       # k still the placeholder
        TrainingSettings(condition=TASK_DIRECTED, purpose="pilot", adaptation_steps=416 * T)
        with self.assertRaises(NotImplementedError):
            TrainingSettings(condition=TASK_DIRECTED, adaptation_fresh_optimizer=False)
        with self.assertRaises(ValueError):
            TrainingSettings(condition=TASK_DIRECTED, updates_per_env_step=0.33)

    def test_defaults_match_agreed_settings(self):
        s = TrainingSettings(condition=TASK_DIRECTED)
        self.assertEqual((s.batch_size, s.warmup_updates, s.updates_per_env_step), (256, 10_000, 1.0))
        self.assertEqual((s.budget_steps, s.practice_episodes, s.demos_per_pair), (249_960, 4166, 20))
        self.assertEqual(s.adaptation_steps, K_PLACEHOLDER)
        self.assertEqual(TrainingSettings(condition=FRONTIER, learner="plain").gamma, GAMMA_PLAIN)

    def test_adapter_and_evaluation_settings_follow_training(self):
        s = TrainingSettings(condition=TASK_DIRECTED, learning_rate=3e-4, updates_per_env_step=2.0,
                             adaptation_steps=40 * T, lr_warmup_steps=200)
        adapter, config = adapter_for(s), evaluation_config_for(s)
        self.assertEqual((adapter.gamma, adapter.learning_rate, adapter.batch_size), (s.gamma, 3e-4, 256))
        self.assertEqual(adapter.updates_per_env_step, config.adaptation_updates_per_env_step)
        self.assertEqual(config.adaptation_steps, 40 * T)
        self.assertEqual(adapter.lr_warmup_steps, 0)
        self.assertEqual(adapter_for(TrainingSettings(condition=TASK_DIRECTED, lr_warmup_steps=200,
                                                      lr_warmup_in_adaptation=True)).lr_warmup_steps, 200)


class TestScript(unittest.TestCase):
    def test_writes_files_outside_results(self):
        import run_training as script
        with tempfile.TemporaryDirectory() as folder:
            settings = Path(folder) / "settings_in.json"
            settings.write_text(json.dumps({k: v for k, v in TINY.items() if k != "budget_steps"}))
            out = Path(folder) / "run"
            script.main(["--condition", "random_actions", "--budget-steps", str(T),
                         "--settings", str(settings), "--out", str(out)])
            for name in ("settings.json", "run_record.json", "episodes.jsonl", "summary.json",
                         "pretrained.pt", "final.pt"):
                self.assertTrue((out / name).exists(), name)
            self.assertEqual(len((out / "episodes.jsonl").read_text().splitlines()), 1)
            torch.load(out / "final.pt")
            with self.assertRaises(SystemExit):  # a finished run is never overwritten
                script.main(["--condition", "random_actions", "--budget-steps", str(T),
                             "--settings", str(settings), "--out", str(out)])

    def test_refuses_results_from_a_dirty_tree(self):
        import provenance
        import run_training as script
        dirty = {"git_commit": "abc", "working_tree_dirty": True}
        with mock.patch.object(provenance, "git_state", return_value=dirty):
            with self.assertRaises(SystemExit):
                script.main(["--condition", "random_actions", "--budget-steps", str(T),
                             "--out", str(ROOT / "results" / "should_not_exist")])
        self.assertFalse((ROOT / "results" / "should_not_exist").exists())


class TestAuditOf4Nov(unittest.TestCase):
    """Tests added after the independent audit of 4 Nov (SF1, SF2, SF4, MI1 to MI4, MI6)."""

    def test_settings_file_is_not_overridden_by_flag_defaults(self):  # SF1
        import run_training as script
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "s.json"
            path.write_text(json.dumps({"seed": 3, "maze": "M0", "learner": "plain", "purpose": "tuning"}))
            settings, _ = script.parse_settings(["--condition", FRONTIER, "--settings", str(path), "--out", folder])
            self.assertEqual((settings.seed, settings.maze, settings.learner, settings.purpose),
                             (3, "M0", "plain", "tuning"))
            with self.assertRaises(SystemExit):
                script.parse_settings(["--condition", FRONTIER, "--maze", "M1", "--settings", str(path),
                                       "--out", folder])
            path.write_text(json.dumps({"purpose": "pilot"}))  # k still the placeholder
            with self.assertRaises(ValueError):
                script.parse_settings(["--condition", TASK_DIRECTED, "--settings", str(path), "--out", folder])

    def test_suite_refuses_adapter_with_other_learner_settings(self):  # SF2
        from eval.protocol import task_pairs
        from eval.suite import _adapt
        plain = TrainingSettings(condition=TASK_DIRECTED, learner="plain", batch_size=32,
                                 updates_per_env_step=0.5, adaptation_steps=T)
        config = evaluation_config_for(plain)
        self.assertEqual((config.adaptation_gamma, config.adaptation_batch_size), (GAMMA_PLAIN, 32))
        pair = task_pairs("M1")[0]
        tasks = [(pair, pair.wall(pair.shorter))]
        weighted = TrainingSettings(condition=TASK_DIRECTED, batch_size=32, updates_per_env_step=0.5,
                                    adaptation_steps=T)
        with self.assertRaises(AssertionError):
            _adapt(GCSLPolicy(0), tasks, config, adapter_for(weighted), phase=900)
        _, record, _ = _adapt(GCSLPolicy(0), tasks, config, adapter_for(plain), phase=900)
        self.assertEqual(record["learner_settings"], {"gamma": GAMMA_PLAIN, "learning_rate": 1e-3,
                                                      "batch_size": 32})

    def _streams(self, condition):
        """RNG state before every learner update, and before each episode's action sampling."""
        import training
        updates, actions = [], []
        original_update, original_start = GCSLLearner.update, GCSLPolicy.start_episode

        def update(learner, buffer, rng):
            updates.append(json.dumps(rng.bit_generator.state, sort_keys=True))
            return original_update(learner, buffer, rng)

        def start(policy, view, rng, sampled):
            actions.append(json.dumps(rng.bit_generator.state, sort_keys=True))
            return original_start(policy, view, rng, sampled)

        with mock.patch.object(GCSLLearner, "update", update), mock.patch.object(GCSLPolicy, "start_episode", start):
            training.run_training(tiny(condition))
        return updates, actions

    def test_batch_and_action_streams_are_shared_across_conditions(self):  # SF4: paired seeds
        demos_updates, _ = self._streams(DEMOS_ONLY)
        task_updates, task_actions = self._streams(TASK_DIRECTED)
        frontier_updates, frontier_actions = self._streams(FRONTIER)
        warmup = TINY["warmup_updates"]
        self.assertEqual(demos_updates[warmup], task_updates[warmup])      # first practice batch draw
        self.assertEqual(task_updates[warmup], frontier_updates[warmup])
        self.assertNotEqual(task_updates[0], task_updates[warmup])          # warm-up has its own stream
        self.assertEqual(task_actions, frontier_actions)
        self.assertEqual(len(task_actions), 2)

    def test_validation_set_is_separate_from_d0(self):  # SF4: validation rule (M11, M12)
        import training
        d0 = training.build_demonstrations("M1", 2, 0)
        validation = training.build_demonstrations("M1", 2, 0, seed_tag=training.TAG_VALIDATION)
        d0_states = {tuple(o) for d in d0 for o in d["observations"]}
        self.assertFalse({tuple(d["observations"][0]) for d in validation} & d0_states)
        seen = []
        original = training.validation_loss
        with mock.patch.object(training, "validation_loss",
                               lambda policy, batch: seen.append(batch) or original(policy, batch)):
            run_training(tiny(TASK_DIRECTED, warmup_rule="validation", warmup_min_updates=20,
                              warmup_max_updates=40, validation_check_every=20, validation_window=20,
                              validation_examples=64, validation_per_pair=2, budget_steps=0))
        self.assertTrue(seen)
        self.assertFalse({tuple(o) for o in seen[0]["observation"]} & d0_states)

    def test_validation_bounds_must_be_whole_checks(self):  # MI1
        with self.assertRaises(ValueError):
            TrainingSettings(condition=TASK_DIRECTED, warmup_rule="validation", warmup_min_updates=30,
                             warmup_max_updates=100, validation_check_every=30, validation_window=30)

    def test_extra_optimizer_step_fails_the_budget_audit(self):  # MI3
        original = GCSLLearner.update
        calls = [0]

        def update(learner, buffer, rng):
            calls[0] += 1
            stats = original(learner, buffer, rng)
            if calls[0] == TINY["warmup_updates"] + 5:
                torch.optim.SGD(learner.policy.network.parameters(), lr=0.0).step()
            return stats

        with mock.patch.object(GCSLLearner, "update", update):
            with self.assertRaisesRegex(AssertionError, "budget audit failed"):
                run_training(tiny(TASK_DIRECTED))

    def test_demonstration_on_the_wrong_route_is_refused(self):  # MI4
        import training
        original = training.make_demonstrations

        def mislabelled(*args, **kwargs):
            demos = original(*args, **kwargs)
            demos[0]["route"] = "south" if demos[0]["route"] == "north" else "north"
            return demos

        with mock.patch.object(training, "make_demonstrations", mislabelled):
            with self.assertRaisesRegex(AssertionError, "did not drive"):
                training.build_demonstrations("M1", 2, 0)

    def test_pilot_and_main_need_agreed_settings(self):  # MI6
        k = 24_960  # 10% of the pilot N, rounded down to whole episodes
        TrainingSettings(condition=TASK_DIRECTED, purpose="pilot", adaptation_steps=k)
        for bad in ({"demos_per_pair": 2}, {"batch_size": 32}, {"maze": "M0"}, {"adaptation_steps": 0},
                    {"adaptation_steps": 1500}, {"budget_steps": 99_960}):
            with self.assertRaises(ValueError, msg=bad):
                TrainingSettings(condition=TASK_DIRECTED, purpose="pilot", **{"adaptation_steps": k, **bad})
        with self.assertRaises(ValueError):
            TrainingSettings(condition=TASK_DIRECTED, learning_rate=0.0)


if __name__ == "__main__":
    unittest.main()
