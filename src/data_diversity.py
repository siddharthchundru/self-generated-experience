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
3. Route mix in the data: every *trip* across the ring, meaning a stretch of
   data that leaves the west column of the ring and next reaches the east
   column (or the reverse). Each trip is labelled by the arm that carried it
   (its last midline crossing, the rule in `routes.py`) and by whether that
   arm is the shorter or the longer one *for that trip's own start and end
   cells* (grid planner; equal lengths are a tie). Every trip becomes
   relabelled examples along that arm, so this is the data-side counterpart
   of Rep. The main data score (spec section 9) is the longer-arm share of
   west-to-east trips: 0 when the data has collapsed onto the shorter route,
   0.5 when both routes are equally represented.
   Why trips and not midline crossings: pooling crossings by arm cannot see
   collapse, because the shorter arm is north in the upper rows and south in
   the lower rows, so data that always takes the shorter route splits evenly
   between the arms. And a ball parked at a goal on the midline wobbles back
   and forth across it, which would count as many crossings; it makes no
   trip. (Both found by the independent audit of 30 Oct.)

Settings (BINS_PER_CELL, the add-one smoothing) are fixed here, before any
practice result exists, and are the same for every condition.
"""

import math

import numpy as np

from envs.mazes import RINGS, free_cells
from envs.pointmaze import MAZES
from expert import plan_cells
from routes import trajectory_route

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
        self.trips = {"all": _empty_trips(), "collected": _empty_trips()}  # end_rows: trips per end row
        self._shorter_cache = {}
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
            self._tally_trips(positions, rows, cols, ("all",))

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
        self._tally_trips(positions, rows, cols, ("all", "collected"))
        self.episodes.append(record)
        return record

    def _shorter_arm(self, start, end):
        key = (start, end)
        if key not in self._shorter_cache:
            north = len(plan_cells(self.grid, start, end, "north"))
            south = len(plan_cells(self.grid, start, end, "south"))
            self._shorter_cache[key] = "tie" if north == south else ("north" if north < south else "south")
        return self._shorter_cache[key]

    def _tally_trips(self, positions, rows, cols, keys):
        """Find trips across the ring and tally them by arm, direction and
        shorter/longer.

        The ring's west and east columns are corridors, so the ball's visits to
        each come in unbroken runs of steps. A trip joins a run in one end
        column to the next run in the other. It starts where the ball entered
        the first column and ends where it left the second (or the episode
        ended), so a ball that climbs the west column from row 6, crosses by
        the north arm and descends to row 6 is a row-6 trip on its longer arm.
        """
        if self.ring is None:
            return
        west, east = self.ring["west_col"], self.ring["east_col"]
        positions = np.asarray(positions, dtype=np.float64)
        cell_rows, cell_cols = (rows // self.k).tolist(), (cols // self.k).tolist()
        runs = []                         # [side, first index, last index]
        for i, c in enumerate(cell_cols):
            side = "west" if c == west else "east" if c == east else None
            if side is None:
                continue
            if runs and runs[-1][0] == side and runs[-1][2] == i - 1:
                runs[-1][2] = i
            else:
                runs.append([side, i, i])
        for (side_a, a0, _), (side_b, _, b1) in zip(runs, runs[1:]):
            if side_a == side_b:
                continue
            arm = trajectory_route(positions[a0:b1 + 1], self.maze)
            if arm == "neither":          # cannot happen: the ring forces a crossing
                raise ValueError("trip between the end columns without a midline crossing")
            shorter = self._shorter_arm((cell_rows[a0], cell_cols[a0]),
                                        (cell_rows[b1], cell_cols[b1]))
            length = "tie" if shorter == "tie" else ("shorter" if arm == shorter else "longer")
            direction = "west_to_east" if side_b == "east" else "east_to_west"
            for key in keys:
                self.trips[key][direction][arm] += 1
                self.trips[key][direction][length] += 1
                rows_seen = self.trips[key][direction]["end_rows"]
                rows_seen[cell_rows[b1]] = rows_seen.get(cell_rows[b1], 0) + 1

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
                "trips": _with_shares(self.trips[key]) if self.ring else None,
            }
        return out


def _empty_trips():
    return {d: {"north": 0, "south": 0, "shorter": 0, "longer": 0, "tie": 0, "end_rows": {}}
            for d in ("west_to_east", "east_to_west")}


def _with_shares(trips):
    """Add the longer-arm share per direction (0-1; ties excluded; None with
    no non-tie trips). For west-to-east trips this is the main data score."""
    out = {}
    for direction, t in trips.items():
        n = t["shorter"] + t["longer"]
        out[direction] = {**t, "end_rows": dict(sorted(t["end_rows"].items())),
                          "longer_arm_share": t["longer"] / n if n else None}
    return out
