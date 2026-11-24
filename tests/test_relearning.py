"""Check the re-learning control and the saved final buffer (spec section 7, H4).

What these tests protect (tiny settings: they check the code, not results):
- the final buffer saved with a run is exactly what training sampled from
  (arrays, action bins and sources round-trip; batches are identical);
- the re-learner's gradient steps equal the original run's warm-up plus
  practice, it never builds an environment, and Adam is reset between phases;
- init "same" reproduces the original pretrained policy exactly, "fresh" does not;
- practice samples the whole final buffer from its first update;
- a validation-rule original is matched by its recorded warm-up length;
- a tampered buffer, an unfinished audit, or D0 out of place is refused;
- the plain learner is refused for pilot and main re-learning.
"""

from dataclasses import replace
import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from collection import DEMOS_ONLY, FRONTIER, TASK_DIRECTED  # noqa: E402
from episode import EPISODE_STEPS  # noqa: E402
from gcsl import TrajectoryBuffer  # noqa: E402
import relearning  # noqa: E402
from relearning import run_relearning  # noqa: E402
from training import TrainingSettings, run_training  # noqa: E402

T = EPISODE_STEPS
TINY = {"warmup_updates": 30, "demos_per_pair": 2, "budget_steps": 2 * T, "batch_size": 32}


def tiny(condition, **overrides):
    return TrainingSettings(condition=condition, **{**TINY, **overrides})


def no_environment(*args, **kwargs):
    raise AssertionError("the re-learning control must not build an environment")


class TestRelearning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = tiny(FRONTIER)
        cls.original = run_training(cls.settings)
        cls.buffer = TrajectoryBuffer.from_arrays(cls.original["final_buffer"])

    def relearn(self, **kwargs):
        with mock.patch("gymnasium.make", no_environment), mock.patch("envs.pointmaze.make_env", no_environment), \
                mock.patch("collection.make_env", no_environment, create=True), \
                mock.patch("demonstrations.make_env", no_environment, create=True):
            return run_relearning(self.settings, self.original["summary"], self.buffer, **kwargs)

    def test_buffer_round_trip(self):
        summary = self.original["summary"]
        self.assertEqual(self.buffer.digest(), summary["final_buffer_digest"])
        self.assertEqual(len(self.buffer), summary["budget"]["expected_buffer_trajectories"])
        copy_ = TrajectoryBuffer.from_arrays(self.buffer.to_arrays())
        a = self.buffer.sample(64, np.random.default_rng(3), 0.962)
        b = copy_.sample(64, np.random.default_rng(3), 0.962)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        copy_.add({"observations": np.zeros((T + 1, 4)), "actions": np.zeros((T, 2)), "goal": np.zeros(2)}, "x")
        self.assertEqual(len(copy_), len(self.buffer) + 1)  # a rebuilt buffer still grows

    def test_matched_gradient_steps_and_no_environment(self):
        result = self.relearn()
        budget = result["summary"]["budget"]
        original = self.original["summary"]["budget"]
        self.assertTrue(budget["passed"])
        self.assertEqual(budget["warmup_gradient_steps"], original["warmup_gradient_steps"])
        self.assertEqual(budget["practice_gradient_steps"], original["practice_gradient_steps"])
        self.assertEqual(budget["practice_optimizer_steps"], original["practice_gradient_steps"])  # Adam reset
        self.assertEqual(budget["all_optimizer_steps_in_process"],
                         original["warmup_gradient_steps"] + original["practice_gradient_steps"])
        self.assertEqual(budget["environment_steps"], 0)
        self.assertEqual(len(result["chunks"]), self.settings.practice_episodes)

    def test_same_reproduces_pretrained_and_fresh_does_not(self):
        same = self.relearn(init="same")["summary"]
        fresh = self.relearn(init="fresh")["summary"]
        self.assertEqual(same["pretrained_digest"], self.original["summary"]["pretrained_digest"])
        self.assertNotEqual(fresh["pretrained_digest"], self.original["summary"]["pretrained_digest"])
        self.assertNotEqual(same["final_digest"], self.original["summary"]["final_digest"])

    def test_practice_samples_the_whole_final_buffer_from_the_start(self):
        sizes = []
        original = TrajectoryBuffer.sample

        def sample(buffer, batch_size, rng, gamma):
            sizes.append(len(buffer))
            return original(buffer, batch_size, rng, gamma)

        with mock.patch.object(TrajectoryBuffer, "sample", sample):
            self.relearn()
        warmup = self.original["summary"]["budget"]["warmup_gradient_steps"]
        self.assertEqual(set(sizes[:warmup]), {self.original["summary"]["demonstrations"]})
        self.assertEqual(set(sizes[warmup:]), {len(self.buffer)})

    def test_validation_rule_original_is_matched_by_its_recorded_length(self):
        settings = tiny(TASK_DIRECTED, warmup_rule="validation", warmup_min_updates=20, warmup_max_updates=60,
                        validation_check_every=20, validation_window=20, validation_examples=64,
                        validation_per_pair=2)
        original = run_training(settings)
        buffer = TrajectoryBuffer.from_arrays(original["final_buffer"])
        result = run_relearning(settings, original["summary"], buffer, init="same")
        self.assertEqual(result["summary"]["budget"]["warmup_gradient_steps"],
                         original["summary"]["budget"]["warmup_gradient_steps"])
        self.assertEqual(result["summary"]["pretrained_digest"], original["summary"]["pretrained_digest"])

    def test_demonstrations_only_has_d0_alone(self):
        original = run_training(tiny(DEMOS_ONLY))
        buffer = TrajectoryBuffer.from_arrays(original["final_buffer"])
        result = run_relearning(tiny(DEMOS_ONLY), original["summary"], buffer)
        self.assertEqual(result["summary"]["source_mix"], {"D0": original["summary"]["demonstrations"]})

    def test_refusals(self):
        summary = self.original["summary"]
        arrays = self.buffer.to_arrays()
        for name in arrays:  # every array is covered by the digest (M22)
            tampered = {k: v.copy() for k, v in arrays.items()}
            if name == "action_bins":
                tampered[name][-1, 0] = (tampered[name][-1, 0] + 1) % 81
            elif name == "sources":
                tampered[name][-1] = "expert_top_up"
            else:
                tampered[name].flat[-1] += 0.25
            with self.assertRaisesRegex(ValueError, "differs from the one", msg=name):
                run_relearning(self.settings, summary, TrajectoryBuffer.from_arrays(tampered))
        failed = copy.deepcopy(summary)
        failed["budget"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "budget audit"):
            run_relearning(self.settings, failed, self.buffer)
        shuffled = {k: np.roll(v, 1, axis=0) for k, v in arrays.items()}
        moved = copy.deepcopy(summary)
        moved["final_buffer_digest"] = TrajectoryBuffer.from_arrays(shuffled).digest()
        with self.assertRaisesRegex(ValueError, "D0 must be"):
            run_relearning(self.settings, moved, TrajectoryBuffer.from_arrays(shuffled))
        with self.assertRaisesRegex(ValueError, "disagree"):
            run_relearning(replace(self.settings, learning_rate=3e-4), summary, self.buffer)
        with self.assertRaises(ValueError):
            run_relearning(self.settings, summary, self.buffer, init="other")
        pilot_plain = TrainingSettings(condition=TASK_DIRECTED, learner="plain", purpose="pilot",
                                       adaptation_steps=24_960)
        with self.assertRaisesRegex(ValueError, "main \\(weighted\\) learner"):
            run_relearning(pilot_plain, summary, self.buffer)

    def test_carried_over_adam_state_is_caught(self):  # M5
        from gcsl import GCSLLearner, GCSLPolicy
        from training import assert_fresh_adam
        policy = GCSLPolicy(0)
        warm = GCSLLearner(policy, 0.962, 1e-3, 32)
        for _ in range(5):
            warm.update(self.buffer, np.random.default_rng(0))
        fresh = GCSLLearner(policy, 0.962, 1e-3, 32)
        fresh.update(self.buffer, np.random.default_rng(1))
        assert_fresh_adam(fresh.optimizer)  # passes
        carried = GCSLLearner(policy, 0.962, 1e-3, 32)
        state = warm.optimizer.state_dict()
        for entry in state["state"].values():
            entry["step"].zero_()  # step counter zeroed, running averages kept
        carried.optimizer.load_state_dict(state)
        carried.update(self.buffer, np.random.default_rng(1))
        with self.assertRaisesRegex(AssertionError, "not empty"):
            assert_fresh_adam(carried.optimizer)

    def test_tags_are_distinct_from_training(self):
        import training
        tags = {relearning.TAG_RELEARN_INIT, relearning.TAG_RELEARN_WARMUP, relearning.TAG_RELEARN_PRACTICE}
        self.assertFalse(tags & {training.TAG_INIT, training.TAG_WARMUP, training.TAG_PRACTICE,
                                 training.TAG_VALIDATION, training.TAG_VALIDATION_BATCH})
        self.assertEqual(len(tags), 3)


if __name__ == "__main__":
    unittest.main()
