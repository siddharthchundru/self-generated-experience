"""The learner: weighted goal-conditioned supervised learning (spec section 5).

What this file does, in one sentence: it defines the policy (an MLP over
observation and goal with a categorical head on a joint 9 x 9 grid of
actions), a buffer that stores whole fixed-length trajectories and relabels
them with goals they actually reached, the weighted imitation update, and the
common adaptation procedure that every condition's recovery is measured with.

Why it matters: the learner is the one thing held identical across every
condition. Conditions differ only in which goal each collection episode
pursues; everything in this file (architecture, relabelling, weighting, batch
size, optimiser, update ratio) is shared, and plain GCSL is the same code with
gamma = 1.

How the pieces map to the spec:
- Policy. Input: position and velocity (4) and goal position (2), and nothing
  else (no remaining horizon, no previous action). Two hidden layers of 256
  ReLU units. Output: 81 logits, one per combination of x-force bin and y-force
  bin; bin centres are -1, -0.75, ..., 1. A joint grid, not two independent
  heads, so the policy can put its mass on two actions that differ on both axes
  (one per route) without also putting mass on the two mixed corners; a
  Gaussian head would average the two routes instead.
  Greedy action: the most probable bin. Sampled action: a draw at temperature 1
  (fixed, never tuned: a lower temperature sharpens the majority route and
  would cause collapse by itself).
- Relabelling (GCSL appendix A.1). Trajectories are stored whole. A training
  example picks a stored trajectory uniformly at random, then a pair of time
  indices t < t' uniformly among all such pairs in 0..T. The input is the state
  at t, the label is the action taken at t, and the goal is the *position* at
  t'. Implementation choice, stated: "uniformly among all pairs" is read as each
  of the T(T+1)/2 ordered pairs equally likely (drawn as the smaller and larger
  of two distinct uniform indices, as GCSL's released code does with its
  min/max construction), not "t uniform, then t' uniform after t", which would
  over-weight pairs near the end of the episode.
- Weighting (weighted GCSL, Yang et al. 2022, discounted relabelling weight
  only). Each example's loss is multiplied by gamma^(t' - t); the batch loss is
  the weighted sum divided by the sum of the weights. gamma = 0.962 for the main
  learner (spec section 5, fixed by the geometry rule), 1 for plain GCSL.
- Buffer. Unbounded, nothing evicted, demonstrations kept for the whole run,
  every trajectory exactly T steps. Each example draws a trajectory uniformly,
  so D0's share of training falls as collected data accumulates, identically in
  every collecting condition (spec section 5, Training details).
- Actions are stored as joint bin indices. The scripted expert's continuous
  forces are rounded to the nearest bin on each axis; a learned policy's actions
  are bin centres already, so for them the rounding is exact.
- The policy object holds its network and nothing else (no optimiser, no
  buffer), so a deep copy for adaptation cannot carry a condition's old data
  (audit of 28 Oct, S2). The optimiser lives in `GCSLLearner`.

Not in this file: goal selection and the collection loops for each condition
(next step of Stage 2), and the hyperparameters that tuning on M0 sets
(learning rate, updates per environment step).
"""

import hashlib
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from episode import EPISODE_STEPS
from eval.suite import AdaptationResult

BINS_PER_AXIS = 9
BIN_CENTRES = np.linspace(-1.0, 1.0, BINS_PER_AXIS)  # spacing 0.25
NUM_ACTIONS = BINS_PER_AXIS ** 2
HIDDEN_UNITS = 256
GAMMA_WEIGHTED = 0.962  # spec section 5: Delta = 16 steps, target pretrained shorter-route share 0.65
GAMMA_PLAIN = 1.0

# Fixed input scaling, so positions (|x| <= 7 on M1) and velocities (|v| <= 5)
# enter the network on a similar scale. The same constants for every maze and
# condition; not tuned.
POSITION_SCALE = 5.0
VELOCITY_SCALE = 5.0
INPUT_SCALE = np.array([1 / POSITION_SCALE] * 2 + [1 / VELOCITY_SCALE] * 2 + [1 / POSITION_SCALE] * 2,
                       dtype=np.float32)


def pin_single_thread():
    """One CPU thread per process (Stage 2 setting 1: one process per seed and condition)."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def actions_to_bins(actions):
    """Joint bin index (x bin * 9 + y bin) of each continuous action, nearest bin per axis."""
    actions = np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0)
    per_axis = np.rint((actions + 1.0) / (2.0 / (BINS_PER_AXIS - 1))).astype(np.int64)
    return per_axis[..., 0] * BINS_PER_AXIS + per_axis[..., 1]


def bins_to_actions(indices):
    """Bin-centre force (x, y) of each joint bin index."""
    indices = np.asarray(indices)
    return np.stack([BIN_CENTRES[indices // BINS_PER_AXIS], BIN_CENTRES[indices % BINS_PER_AXIS]],
                    axis=-1).astype(np.float32)


class PolicyNetwork(nn.Module):
    """observation (4) + goal (2) -> 256 -> 256 -> 81 logits."""

    def __init__(self):
        super().__init__()
        self.register_buffer("input_scale", torch.from_numpy(INPUT_SCALE))
        self.layers = nn.Sequential(
            nn.Linear(6, HIDDEN_UNITS), nn.ReLU(),
            nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS), nn.ReLU(),
            nn.Linear(HIDDEN_UNITS, NUM_ACTIONS),
        )

    def forward(self, observation, goal):
        return self.layers(torch.cat([observation, goal], dim=-1) * self.input_scale)


class GCSLPolicy:
    """A goal-conditioned policy that sees observation and goal only.

    Implements the learned-policy interface of `eval.suite`:
    `start_episode(view, rng, sampled)` returns an action function, and
    `parameter_digest()` fingerprints the weights.
    """

    def __init__(self, seed, name="gcsl"):
        self.name = name
        # Initial weights depend only on the seed, and seeding here does not
        # disturb the global torch random state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network = PolicyNetwork()

    def probabilities(self, observation, goal):
        with torch.no_grad():
            logits = self.network(torch.as_tensor(observation, dtype=torch.float32),
                                  torch.as_tensor(goal, dtype=torch.float32))
            return torch.softmax(logits.double(), dim=-1).numpy()

    def start_episode(self, view, rng, sampled):
        goal = np.asarray(view["desired_goal"], dtype=np.float32)

        def act(current):
            probs = self.probabilities(current["observation"][None], goal[None])[0]
            if sampled:  # temperature 1
                index = rng.choice(NUM_ACTIONS, p=probs / probs.sum())
            else:
                index = int(np.argmax(probs))
            return bins_to_actions(index)

        return act

    def parameter_digest(self):
        digest = hashlib.sha256()
        for key, tensor in self.network.state_dict().items():
            digest.update(key.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


class TrajectoryBuffer:
    """Every stored trajectory, whole, never evicted; relabelled at sampling time."""

    def __init__(self, episode_steps=EPISODE_STEPS):
        self.episode_steps = episode_steps
        self._observations = np.zeros((0, episode_steps + 1, 4), dtype=np.float32)
        self._actions = np.zeros((0, episode_steps), dtype=np.int16)
        self._goals = np.zeros((0, 2), dtype=np.float32)
        self.sources = []
        self._size = 0

    def __len__(self):
        return self._size

    def _grow(self):
        capacity = max(64, 2 * len(self._observations))
        for name in ("_observations", "_actions", "_goals"):
            old = getattr(self, name)
            new = np.zeros((capacity,) + old.shape[1:], dtype=old.dtype)
            new[:self._size] = old[:self._size]
            setattr(self, name, new)

    def add(self, trajectory, source):
        """Store one episode: observations (T+1, 4), actions (T, 2), commanded goal (2,)."""
        observations = np.asarray(trajectory["observations"], dtype=np.float32)
        actions = np.asarray(trajectory["actions"])
        if observations.shape != (self.episode_steps + 1, 4) or actions.shape != (self.episode_steps, 2):
            raise ValueError(f"trajectories must be exactly T = {self.episode_steps} steps")
        if self._size == len(self._observations):
            self._grow()
        self._observations[self._size] = observations
        self._actions[self._size] = actions_to_bins(actions)
        self._goals[self._size] = np.asarray(trajectory["goal"], dtype=np.float32)
        self.sources.append(source)
        self._size += 1

    def positions(self, index):
        """Positions (T+1, 2) of stored trajectory `index`, e.g. for the data-diversity log."""
        return self._observations[index, :, :2]

    def sample(self, batch_size, rng, gamma):
        """A relabelled batch: (observation, goal, action bin, weight, gap t' - t)."""
        if self._size == 0:
            raise ValueError("cannot sample from an empty buffer")
        T = self.episode_steps
        trajectory = rng.integers(self._size, size=batch_size)
        first = rng.integers(T + 1, size=batch_size)
        second = rng.integers(T, size=batch_size)
        second = second + (second >= first)  # distinct from first, uniform over the rest
        t, t_goal = np.minimum(first, second), np.maximum(first, second)
        gap = t_goal - t
        return {
            "observation": self._observations[trajectory, t],
            "goal": self._observations[trajectory, t_goal, :2],
            "action": self._actions[trajectory, t].astype(np.int64),
            "weight": np.power(gamma, gap, dtype=np.float64),
            "gap": gap,
        }


def weighted_loss(logits, actions, weights):
    """Sum of gamma-weighted cross-entropies divided by the sum of the weights."""
    per_example = F.cross_entropy(logits, actions, reduction="none")
    return (weights * per_example).sum() / weights.sum()


def effective_sample_size(weights):
    """(sum w)^2 / sum w^2: how many equally weighted examples the batch is worth."""
    weights = np.asarray(weights, dtype=np.float64)
    return float(weights.sum() ** 2 / np.square(weights).sum())


class GCSLLearner:
    """Owns the optimiser and performs weighted GCSL updates on a policy's network."""

    def __init__(self, policy, gamma, learning_rate, batch_size=256):
        if not 0 < gamma <= 1:
            raise ValueError("gamma must lie in (0, 1]")
        self.policy = policy
        self.gamma = gamma
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(policy.network.parameters(), lr=learning_rate)
        self.gradient_steps = 0

    def update(self, buffer, rng):
        """One gradient step on one relabelled batch. Returns the loss and the batch's ESS."""
        batch = buffer.sample(self.batch_size, rng, self.gamma)
        logits = self.policy.network(torch.from_numpy(batch["observation"]),
                                     torch.from_numpy(batch["goal"]))
        weights = torch.from_numpy(batch["weight"]).float()
        loss = weighted_loss(logits, torch.from_numpy(batch["action"]), weights)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.gradient_steps += 1
        return {"loss": float(loss.detach()), "effective_sample_size": effective_sample_size(batch["weight"]),
                "mean_gap": float(batch["gap"].mean())}

    def train(self, buffer, updates, rng):
        """`updates` gradient steps; returns the mean loss and mean ESS over them."""
        if updates == 0:
            return {"updates": 0, "loss": None, "effective_sample_size": None}
        stats = [self.update(buffer, rng) for _ in range(updates)]
        return {"updates": updates,
                "loss": float(np.mean([s["loss"] for s in stats])),
                "effective_sample_size": float(np.mean([s["effective_sample_size"] for s in stats]))}


def updates_per_episode(updates_per_env_step, episode_steps=EPISODE_STEPS):
    """Whole gradient steps per collected episode for a given ratio (0.5, 1, 2 -> 30, 60, 120)."""
    updates = updates_per_env_step * episode_steps
    if updates < 0 or not math.isclose(updates, round(updates)):
        raise ValueError("updates per environment step must give whole updates per episode")
    return int(round(updates))


class GCSLAdapter:
    """The common adaptation procedure (spec section 8), used for A_k by every condition.

    Starting from a copy of the final policy and an empty buffer, it repeatedly
    asks the evaluation suite's session for one practice episode on a blocked
    pair (the suite picks which, uniformly), stores it, and then takes
    `updates_per_env_step x T` gradient steps with the same learner settings as
    training. A fresh optimiser is created: optimiser state, like data, is not
    carried over from training. Interleaving collection and updates episode by
    episode is the same schedule the collection loops use.
    """

    def __init__(self, gamma, learning_rate, updates_per_env_step, batch_size=256, seed=0):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.updates_per_env_step = updates_per_env_step
        self.batch_size = batch_size
        self.seed = seed
        self._per_episode = updates_per_episode(updates_per_env_step)

    def new_buffer(self):
        return TrajectoryBuffer()

    def adapt(self, policy, session, buffer):
        learner = GCSLLearner(policy, self.gamma, self.learning_rate, self.batch_size)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, session.phase]))
        while session.remaining_steps >= session.episode_steps:
            buffer.add(session.run_episode(), "adaptation")
            learner.train(buffer, self._per_episode, rng)
        return AdaptationResult(policy, learner.gradient_steps, f"gcsl_gamma_{self.gamma:g}",
                                learner.optimizer)
