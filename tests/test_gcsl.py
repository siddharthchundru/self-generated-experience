"""Check the learner (spec section 5) and the demonstration set (spec section 6).

What these tests protect:
- the joint 9 x 9 action grid round-trips, and expert forces go to the nearest bin;
- relabelling: the goal is the position at t' > t, the label is the action at t,
  pairs (t, t') are uniform over all T(T+1)/2 pairs, trajectories are drawn
  uniformly, and each example's weight is gamma^(t' - t);
- the weighted loss is the weighted sum over the sum of weights, and gamma = 1
  reduces it to plain GCSL's mean cross-entropy;
- the policy holds parameters only, sees observation and goal only, its initial
  weights depend on the seed alone, and greedy is the most probable bin;
- gradient steps are counted, match the optimiser, and lower the loss;
- the common adaptation procedure passes the evaluation suite's budget checks;
- D0 has exactly half of each pair's demonstrations on each route.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demonstrations import demonstration_plan, make_demonstrations
from envs.mazes import M0_DEMO_REGION, RINGS
from envs.pointmaze import xy_to_cell
from episode import EPISODE_STEPS, whole_episodes
from eval.protocol import EvaluationConfig
from eval.suite import check_parameters_only, evaluate_policy, optimizer_step_count
from gcsl import (BIN_CENTRES, NUM_ACTIONS, GCSLAdapter, GCSLLearner, GCSLPolicy, TrajectoryBuffer,
                  actions_to_bins, bins_to_actions, effective_sample_size, updates_per_episode,
                  weighted_loss)

T = EPISODE_STEPS


def clock_trajectory(label, actions=None):
    """A synthetic trajectory whose state at time t encodes (label, t)."""
    observations = np.zeros((T + 1, 4), dtype=np.float32)
    observations[:, 0] = label
    observations[:, 1] = np.arange(T + 1)
    if actions is None:
        actions = bins_to_actions(np.arange(T) % NUM_ACTIONS)
    return {"observations": observations, "actions": actions, "goal": np.zeros(2)}


class TestEpisodeLength(unittest.TestCase):
    def test_one_constant_and_whole_episode_budgets(self):
        self.assertEqual(T, 60)
        self.assertEqual(whole_episodes(249_960, "pilot N"), 4166)
        self.assertEqual(whole_episodes(999_960, "cap N"), 16666)
        with self.assertRaises(ValueError):
            whole_episodes(250_000)


class TestActionGrid(unittest.TestCase):
    def test_round_trip_and_nearest_bin(self):
        np.testing.assert_array_equal(actions_to_bins(bins_to_actions(np.arange(NUM_ACTIONS))),
                                      np.arange(NUM_ACTIONS))
        self.assertEqual(len(BIN_CENTRES), 9)
        self.assertEqual(actions_to_bins([0.1, -0.9]), actions_to_bins([0.0, -1.0]))
        self.assertEqual(actions_to_bins([5.0, -5.0]), actions_to_bins([1.0, -1.0]))  # clipped
        # Joint index: x bin is the major axis.
        np.testing.assert_array_equal(bins_to_actions(1 * 9 + 8), [-0.75, 1.0])

    def test_grid_is_symmetric_under_reflection(self):
        rng = np.random.default_rng(0)
        actions = rng.uniform(-1, 1, size=(1000, 2))
        reflected = actions * np.array([1, -1])
        np.testing.assert_allclose(bins_to_actions(actions_to_bins(reflected)),
                                   bins_to_actions(actions_to_bins(actions)) * np.array([1, -1]))


class TestRelabelling(unittest.TestCase):
    def setUp(self):
        self.buffer = TrajectoryBuffer()
        for label in range(4):
            self.buffer.add(clock_trajectory(label), "test")

    def test_rejects_trajectories_of_the_wrong_length(self):
        bad = clock_trajectory(0)
        bad["observations"] = bad["observations"][:-1]
        with self.assertRaises(ValueError):
            self.buffer.add(bad, "test")

    def test_goal_is_a_later_position_and_label_the_action_then(self):
        batch = self.buffer.sample(5000, np.random.default_rng(0), gamma=0.9)
        t = batch["observation"][:, 1].astype(int)
        t_goal = batch["goal"][:, 1].astype(int)
        self.assertTrue(np.all(t_goal > t))
        self.assertTrue(np.all(t >= 0) and np.all(t_goal <= T))
        np.testing.assert_array_equal(batch["goal"][:, 0], batch["observation"][:, 0])  # same trajectory
        np.testing.assert_array_equal(batch["gap"], t_goal - t)
        np.testing.assert_array_equal(batch["action"], t % NUM_ACTIONS)
        np.testing.assert_allclose(batch["weight"], 0.9 ** (t_goal - t))

    def test_pairs_are_uniform_over_all_pairs(self):
        batch = self.buffer.sample(400_000, np.random.default_rng(1), gamma=1.0)
        pairs = T * (T + 1) / 2
        expected_gap = np.array([(T + 1 - g) / pairs for g in range(1, T + 1)])
        observed_gap = np.bincount(batch["gap"], minlength=T + 1)[1:] / len(batch["gap"])
        np.testing.assert_allclose(observed_gap, expected_gap, atol=0.002)
        # Under "t uniform, then t' uniform after t" the start index would be
        # uniform; under uniform pairs it falls linearly.
        t = batch["observation"][:, 1].astype(int)
        start_share = np.bincount(t, minlength=T) / len(t)
        self.assertAlmostEqual(start_share[0], T / pairs, delta=0.002)
        self.assertAlmostEqual(start_share[T - 1], 1 / pairs, delta=0.002)

    def test_trajectories_are_drawn_uniformly(self):
        batch = self.buffer.sample(40_000, np.random.default_rng(2), gamma=1.0)
        shares = np.bincount(batch["observation"][:, 0].astype(int)) / 40_000
        np.testing.assert_allclose(shares, 0.25, atol=0.01)

    def test_buffer_grows_without_eviction(self):
        buffer = TrajectoryBuffer()
        for label in range(200):
            buffer.add(clock_trajectory(label), "test")
        self.assertEqual(len(buffer), 200)
        self.assertEqual(buffer.positions(0)[0, 0], 0)
        self.assertEqual(buffer.positions(199)[0, 0], 199)


class TestLoss(unittest.TestCase):
    def test_weighted_sum_over_sum_of_weights(self):
        logits = torch.randn(5, NUM_ACTIONS)
        actions = torch.arange(5)
        weights = torch.tensor([1.0, 0.5, 0.25, 0.125, 1.0])
        ce = torch.nn.functional.cross_entropy(logits, actions, reduction="none")
        self.assertAlmostEqual(float(weighted_loss(logits, actions, weights)),
                               float((weights * ce).sum() / weights.sum()), places=6)
        self.assertAlmostEqual(float(weighted_loss(logits, actions, torch.ones(5))),
                               float(torch.nn.functional.cross_entropy(logits, actions)), places=6)

    def test_effective_sample_size(self):
        self.assertEqual(effective_sample_size(np.ones(256)), 256)
        self.assertAlmostEqual(effective_sample_size([1, 0, 0, 0]), 1)


class TestPolicy(unittest.TestCase):
    def test_parameters_only_and_digest(self):
        policy = GCSLPolicy(seed=3)
        check_parameters_only(policy)
        self.assertEqual(policy.parameter_digest(), GCSLPolicy(seed=3).parameter_digest())
        self.assertNotEqual(policy.parameter_digest(), GCSLPolicy(seed=4).parameter_digest())

    def test_seeding_does_not_disturb_global_torch_state(self):
        torch.manual_seed(123)
        expected = torch.rand(3)
        torch.manual_seed(123)
        GCSLPolicy(seed=9)
        torch.testing.assert_close(torch.rand(3), expected)

    def test_greedy_is_most_probable_bin_and_sampling_follows_probabilities(self):
        policy = GCSLPolicy(seed=0)
        view = {"observation": np.array([0.5, -0.5, 0.1, 0.0], dtype=np.float32),
                "desired_goal": np.array([2.0, 1.0], dtype=np.float32)}
        probs = policy.probabilities(view["observation"][None], view["desired_goal"][None])[0]
        greedy = policy.start_episode(view, np.random.default_rng(0), sampled=False)(view)
        np.testing.assert_array_equal(greedy, bins_to_actions(int(np.argmax(probs))))
        act = policy.start_episode(view, np.random.default_rng(0), sampled=True)
        draws = actions_to_bins(np.array([act(view) for _ in range(20_000)]))
        observed = np.bincount(draws, minlength=NUM_ACTIONS) / len(draws)
        np.testing.assert_allclose(observed, probs, atol=0.01)


class TestLearner(unittest.TestCase):
    def test_counts_steps_and_learns(self):
        policy = GCSLPolicy(seed=0)
        buffer = TrajectoryBuffer()
        fixed = bins_to_actions(np.full(T, 40))  # the zero-force bin, always
        for label in range(8):
            buffer.add(clock_trajectory(label, fixed), "test")
        learner = GCSLLearner(policy, gamma=0.962, learning_rate=1e-3, batch_size=64)
        rng = np.random.default_rng(0)
        first = learner.train(buffer, 5, rng)["loss"]
        last = learner.train(buffer, 200, rng)["loss"]
        self.assertLess(last, first / 4)
        self.assertEqual(learner.gradient_steps, 205)
        self.assertEqual(optimizer_step_count(learner.optimizer), 205)
        check_parameters_only(policy)  # the optimiser lives on the learner, not the policy

    def test_gamma_must_be_a_discount(self):
        with self.assertRaises(ValueError):
            GCSLLearner(GCSLPolicy(seed=0), gamma=1.5, learning_rate=1e-3)

    def test_updates_per_episode_for_the_tuning_grid(self):
        self.assertEqual([updates_per_episode(r) for r in (0.5, 1, 2)], [30, 60, 120])
        with self.assertRaises(ValueError):
            updates_per_episode(1 / 7)


class TestAdaptation(unittest.TestCase):
    def test_common_adaptation_procedure_passes_the_suite_checks(self):
        config = EvaluationConfig(sampled_rollouts=5, greedy_rollouts=1, minimum_successes=1,
                                  bootstrap_resamples=10, adaptation_steps=2 * T,
                                  adaptation_updates_per_env_step=0.5)
        policy = GCSLPolicy(seed=0)
        before = policy.parameter_digest()
        adapter = GCSLAdapter(gamma=0.962, learning_rate=1e-3, updates_per_env_step=0.5, batch_size=32)
        report = evaluate_policy(policy, config, adapter)
        self.assertEqual(policy.parameter_digest(), before)
        run = report["adaptation_runs"]["P1"]
        self.assertEqual(run["status"], "gcsl_gamma_0.962")
        self.assertEqual((run["environment_steps"], run["gradient_steps"], run["optimizer_steps"]),
                         (2 * T, T, T))
        self.assertEqual((run["buffer_size_at_start"], run["buffer_size_at_end"]), (0, 2))


class TestDemonstrations(unittest.TestCase):
    def test_plan_splits_every_pair_evenly_and_is_seeded(self):
        plan = demonstration_plan("M1", 20, seed=5)
        self.assertEqual(len(plan), 140)
        for row in range(1, 8):
            routes = [route for pair, route, _ in plan if pair.row == row]
            self.assertEqual((routes.count("north"), routes.count("south")), (10, 10))
        self.assertEqual(plan, demonstration_plan("M1", 20, seed=5))
        self.assertNotEqual([p.row for p, _, _ in plan], [p.row for p, _, _ in demonstration_plan("M1", 20, 6)])
        self.assertEqual(len({seed for _, _, seed in plan}), 140)
        with self.assertRaises(ValueError):
            demonstration_plan("M1", 7, seed=0)

    def test_m0_demonstrations_succeed_and_stay_in_the_demo_region(self):
        demos = make_demonstrations("M0", 20, seed=0)
        self.assertEqual(len(demos), 100)
        self.assertTrue(all(d["success"] for d in demos))
        grid = RINGS["M0"]["grid"]
        buffer = TrajectoryBuffer()
        for d in demos:
            self.assertEqual(d["observations"].shape, (T + 1, 4))
            cells = {xy_to_cell(grid, p) for p in d["observations"][:, :2]}
            self.assertLessEqual(cells, M0_DEMO_REGION)
            buffer.add(d, "demonstration")
        self.assertEqual(len(buffer), 100)


if __name__ == "__main__":
    unittest.main()
