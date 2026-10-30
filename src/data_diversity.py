"""Score the diversity of the data a condition collects (spec section 8, Data rows).

What this file does, in one sentence: it keeps a running tally of where the
ball has been in a condition's buffer and, as each collected episode arrives,
records how much of the maze the data covers, how evenly, how much that
episode added to what was already there, and which arm of the ring the data
crosses by.

Why it matters: Rep and recovery are measured on the *policy*. This file
measures the *data*. Logging both lets the analysis ask whether a condition's
repertoire tracks the diversity of what it collected (spec section 9,
exploratory), which separates "this goal rule collects more diverse data" from
"more diverse data gives a more robust policy". It adds no training runs and
no gradient steps: it only reads positions the collection loop already has.

Three scores, each with a different job:
1. Spread: the share of free space visited, and the visitation entropy,
   normalised by the largest entropy the maze allows. Positions are binned on
   a grid of BINS_PER_CELL x BINS_PER_CELL bins per maze cell; only bins
   inside free cells count, so walls never dilute the score. Plain cell
   coverage would saturate on M1 (35 free cells) early in every condition.
2. Information added by each episode, relative to the buffer before it:
   - new_bins: bins the episode visits that the buffer never had;
   - surprise_bits: the mean, over the episode's states, of -log2 of the
     buffer's visitation probability for that state's bin, with add-one
     smoothing over free bins so an unvisited bin has a finite cost. This is
     the cross-entropy of the episode under the buffer's visitation
     distribution: high when the episode goes where the data is thin.
3. Route mix in the data: west-to-east and east-to-west crossings of the
   ring's midline, by arm (the rule in `routes.py`). Every crossing becomes
   relabelled examples that traverse that arm, so this is the data-side
   counterpart of Rep. It is aggregated over the whole data, not per task
   pair, because collected segments do not belong to a pair.

Settings (BINS_PER_CELL, the add-one smoothing) are fixed here, before any
practice result exists, and are the same for every condition.
"""

import math

import numpy as np

from envs.mazes import RINGS, free_cells
from envs.pointmaze import MAZES
from routes import midline_crossings

BINS_PER_CELL = 4   # 0.25-cell bins: half the ball's top speed per step


class DataDiversityLog:
    """Running visitation tally for one run's buffer.

    `add_demonstrations` is called once with D0; `add_episode` with each
    collected episode, in collection order. Positions are (x, y, ...) rows in
    simulator coordinates; extra columns (velocity) are ignored.
    """

    def __init__(self, maze="M1", bins_per_cell=BINS_PER_CELL):
        self.maze = maze
        self.ring = RINGS.get(maze)          # None on M2, which has no ring
        self.grid = MAZES[maze]
        self.k = bins_per_cell
        n_rows, n_cols = len(self.grid), len(self.grid[0])
        self.shape = (n_rows * self.k, n_cols * self.k)
        self.free_mask = np.zeros(self.shape, dtype=bool)
        for r, c in free_cells(self.grid):
            self.free_mask[r * self.k:(r + 1) * self.k, c * self.k:(c + 1) * self.k] = True
        self.n_free = int(self.free_mask.sum())
        self.counts = {"all": np.zeros(self.shape, dtype=np.int64),
                       "collected": np.zeros(self.shape, dtype=np.int64)}
        self.crossings = {"all": _empty_crossings(), "collected": _empty_crossings()}
        self.episodes = []

    # --- input ------------------------------------------------------------------

    def _bins(self, positions):
        xy = np.asarray(positions, dtype=np.float64)[:, :2]
        n_rows, n_cols = len(self.grid), len(self.grid[0])
        rows = np.floor((n_rows / 2 - xy[:, 1]) * self.k).astype(int)
        cols = np.floor((xy[:, 0] + n_cols / 2) * self.k).astype(int)
        rows = np.clip(rows, 0, self.shape[0] - 1)
        cols = np.clip(cols, 0, self.shape[1] - 1)
        if not self.free_mask[rows, cols].all():
            raise ValueError("a position lies inside a wall bin")
        return rows, cols

    def add_demonstrations(self, trajectories):
        if self.episodes or self.counts["collected"].any():
            raise ValueError("demonstrations must be added before any collected episode")
        for positions in trajectories:
            rows, cols = self._bins(positions)
            np.add.at(self.counts["all"], (rows, cols), 1)
            self._tally_crossings(positions, ("all",))

    def add_episode(self, positions):
        """Add one collected episode; return what it added to the buffer."""
        rows, cols = self._bins(positions)
        before = self.counts["all"]
        total = before.sum()
        p = (before[rows, cols] + 1) / (total + self.n_free)       # add-one smoothing
        visited = np.zeros(self.shape, dtype=bool)
        visited[rows, cols] = True
        record = {
            "episode": len(self.episodes),
            "new_bins": int((visited & (before == 0)).sum()),
            "surprise_bits": float(np.mean(-np.log2(p))),
        }
        for key in ("all", "collected"):
            np.add.at(self.counts[key], (rows, cols), 1)
        self._tally_crossings(positions, ("all", "collected"))
        self.episodes.append(record)
        return record

    def _tally_crossings(self, positions, keys):
        if self.ring is None:
            return
        mid = self.ring["midline_col"]
        n_cols = len(self.grid[0])
        midline_x = mid + 0.5 - n_cols / 2
        north_y = len(self.grid) / 2 - self.ring["north_row"] - 0.5
        south_y = len(self.grid) / 2 - self.ring["south_row"] - 0.5
        xy = np.asarray(positions, dtype=np.float64)[:, :2]
        for j, y in midline_crossings(xy, midline_x):
            if abs(y - north_y) <= 0.5:
                arm = "north"
            elif abs(y - south_y) <= 0.5:
                arm = "south"
            else:  # the walls make this impossible, as in routes.py
                raise ValueError(f"midline crossed at y={y:.3f}, outside both arms")
            direction = "west_to_east" if xy[j + 1, 0] > xy[j, 0] else "east_to_west"
            for key in keys:
                self.crossings[key][direction][arm] += 1

    # --- output -----------------------------------------------------------------

    def summary(self):
        """Spread and route mix, for the whole buffer and for collected data only."""
        out = {"maze": self.maze, "bins_per_cell": self.k, "free_bins": self.n_free,
               "collected_episodes": len(self.episodes)}
        for key in ("all", "collected"):
            counts = self.counts[key][self.free_mask]
            total = counts.sum()
            probs = counts[counts > 0] / total if total else np.array([])
            entropy = float(-(probs * np.log(probs)).sum()) if total else None
            out[key] = {
                "states": int(total),
                "coverage": float((counts > 0).sum() / self.n_free),
                "entropy_normalised": entropy / math.log(self.n_free) if total else None,
                "crossings": _with_balance(self.crossings[key]) if self.ring else None,
            }
        return out


def _empty_crossings():
    return {d: {"north": 0, "south": 0} for d in ("west_to_east", "east_to_west")}


def _with_balance(crossings):
    """Add the minority-arm share per direction (0-0.5; None with no crossings)."""
    out = {}
    for direction, arms in crossings.items():
        n = arms["north"] + arms["south"]
        out[direction] = {**arms, "minority_arm_share": min(arms.values()) / n if n else None}
    return out
