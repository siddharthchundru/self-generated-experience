"""Frontier novelty's goal rule: pursue the least-visited place the agent has reached.

What this file does, in one sentence: `FrontierSelector.choose(rng)` draws 100
candidate goals uniformly from every position in the condition's own data
(demonstrations included), fits a Gaussian kernel density estimate to up to
10,000 achieved positions, and returns the candidate where that density is
lowest.

Why it matters: this is MEGA's goal-selection rule (Pitis et al., 2020), the
one thing that distinguishes frontier novelty from task-directed practice. The
learner, buffer and budget are identical; only the commanded goal differs
(spec section 6).

Settings (spec section 6, "Frontier novelty (C4) implementation"):
- candidates: 100, drawn uniformly from the achieved-goal buffer, i.e. the
  positions of all stored states, demonstrations included;
- density: Gaussian kernel, bandwidth 0.1 on normalised positions, refit at
  the start of every collection episode on up to 10,000 achieved positions
  drawn uniformly (MEGA refits every optimisation step; goals are chosen once
  per episode, so refitting once per episode selects identically);
- selection: the candidate with the lowest density (deterministic given the
  candidate pool);
- no achievability cutoff and no Go Exploration (deviations from MEGA,
  confirmed by Siddharth on 28 Oct).

Not decided by the spec, chosen here and exposed as a parameter (listed for
Siddharth's review in the PR and in spec v0.14):
- "normalised positions": positions are divided by `position_scale`,
  default 5.0, the same fixed scaling the policy network applies to its
  inputs (`gcsl.POSITION_SCALE`). With it, bandwidth 0.1 is half a cell.

The kernel density is evaluated up to a constant factor (the normalising
constant does not change which candidate is lowest), in float64.
"""

import numpy as np

from gcsl import POSITION_SCALE

CANDIDATES = 100
BANDWIDTH = 0.1
DENSITY_FIT_POINTS = 10_000


class FrontierSelector:
    """Chooses frontier novelty's goal for each collection episode.

    `add_positions` is called with D0's positions once and with each collected
    episode's positions after it ends, so the achieved-goal set is exactly the
    positions stored in the condition's buffer.
    """

    def __init__(self, candidates=CANDIDATES, bandwidth=BANDWIDTH,
                 fit_points=DENSITY_FIT_POINTS, position_scale=POSITION_SCALE):
        if candidates <= 0 or fit_points <= 0 or bandwidth <= 0 or position_scale <= 0:
            raise ValueError("frontier settings must be positive")
        self.candidates = candidates
        self.bandwidth = bandwidth
        self.fit_points = fit_points
        self.position_scale = position_scale
        self._positions = np.zeros((1024, 2))
        self._size = 0

    def add_positions(self, positions):
        """Add achieved positions (n, 2+) in simulator coordinates."""
        positions = np.asarray(positions, dtype=np.float64)[:, :2]
        needed = self._size + len(positions)
        if needed > len(self._positions):
            grown = np.zeros((max(needed, 2 * len(self._positions)), 2))
            grown[:self._size] = self._positions[:self._size]
            self._positions = grown
        self._positions[self._size:needed] = positions
        self._size = needed

    @property
    def achieved(self):
        return self._positions[:self._size]

    def log_density(self, points, fitted):
        """Log of the (unnormalised) Gaussian kernel density of `fitted` at `points`."""
        a = np.asarray(points, dtype=np.float64) / self.position_scale
        b = np.asarray(fitted, dtype=np.float64) / self.position_scale
        squared = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        exponents = -squared / (2 * self.bandwidth ** 2)
        top = exponents.max(axis=1, keepdims=True)
        return (top[:, 0] + np.log(np.exp(exponents - top).sum(axis=1))) - np.log(len(b))

    def choose(self, rng):
        """The lowest-density candidate, plus what was compared, for logging."""
        achieved = self.achieved
        if len(achieved) == 0:
            raise ValueError("frontier novelty needs achieved positions to choose from")
        candidates = achieved[rng.integers(len(achieved), size=self.candidates)]
        fit_size = min(self.fit_points, len(achieved))
        fitted = achieved[rng.choice(len(achieved), size=fit_size, replace=False)]
        log_density = self.log_density(candidates, fitted)
        best = int(np.argmin(log_density))
        return candidates[best].copy(), {"log_density": float(log_density[best]),
                                         "median_candidate_log_density": float(np.median(log_density)),
                                         "fit_points": fit_size}
