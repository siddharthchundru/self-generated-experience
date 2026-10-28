"""Classify which route a simulated trajectory took around M1's ring.

What this file does, in one sentence: given the ball's positions over an
episode, it says whether the ball got across the ring by the north arm, by the
south arm, or not at all.

Why it matters: the primary repertoire measure (Rep, the minority-route share)
counts, among successful rollouts on a two-route pair, how many took the
less-used route. That is only as good as the rule that decides which route a
rollout took. The rule here is checked against hand labels on 50+ messy
trajectories before the main runs (Stage 1 gate).

The rule: the ring's central wall block spans columns 2-6 of rows 2-6, so the
vertical line through the middle of column 4 can only be crossed inside the
north arm (row 1) or the south arm (row 7). A trajectory's route is the arm of
its LAST crossing of that line.

Why the last crossing, rather than "which arm cells did it visit" (the cell-set
rule `mazes.route_class`, which is kept for planner paths): a learned policy
may enter one arm, turn back and go round the other way. The cell-set rule
calls that "both"; the last-crossing rule names the arm that actually carried
the ball across, which is the route the rollout used to reach the goal. A
brief dip into an arm without crossing the midline does not count at all.
For the west-east pairs this applies to (start in column 1, goal in column 7),
every successful rollout crosses the line at least once, so "neither" only
occurs for rollouts that never got across.
"""

import numpy as np

from envs.mazes import M1, NORTH_ARM, SOUTH_ARM
from envs.pointmaze import cell_to_xy

MIDLINE_COL = 4
MIDLINE_X = float(cell_to_xy(M1, (1, MIDLINE_COL))[0])   # x of the centre of column 4
_NORTH_Y = float(cell_to_xy(M1, (1, MIDLINE_COL))[1])    # centre of the north arm (row 1)
_SOUTH_Y = float(cell_to_xy(M1, (7, MIDLINE_COL))[1])    # centre of the south arm (row 7)

assert (1, MIDLINE_COL) in NORTH_ARM and (7, MIDLINE_COL) in SOUTH_ARM


def midline_crossings(positions):
    """(index, y) for every step where the ball moves from one side of the
    midline to the other, with y interpolated linearly at the crossing.

    A point exactly on the line counts as still being on the side it came from,
    so touching the line and going back is not a crossing.
    """
    xy = np.asarray(positions, dtype=np.float64)[:, :2]
    side = np.sign(xy[:, 0] - MIDLINE_X)
    for i in range(len(side)):
        if side[i] == 0:
            side[i] = side[i - 1] if i > 0 else 0
    crossings = []
    for i in range(1, len(side)):
        if side[i - 1] != 0 and side[i] != side[i - 1]:
            j = i - 1  # the segment from j to i crosses the line (j may lie on it)
            x0, x1 = xy[j, 0], xy[i, 0]
            frac = 0.0 if x1 == x0 else (MIDLINE_X - x0) / (x1 - x0)
            crossings.append((j, xy[j, 1] + frac * (xy[i, 1] - xy[j, 1])))
    return crossings


def trajectory_route(positions):
    """Route class of a trajectory on M1: "north", "south" or "neither".

    `positions` is a sequence of (x, y, ...) rows, e.g. the observations of an
    episode (velocity columns are ignored).
    """
    crossings = midline_crossings(positions)
    if not crossings:
        return "neither"
    y = crossings[-1][1]
    # The block makes any other y impossible; fail loudly if it ever appears.
    if abs(y - _NORTH_Y) <= 0.5:
        return "north"
    if abs(y - _SOUTH_Y) <= 0.5:
        return "south"
    raise ValueError(f"midline crossed at y={y:.3f}, outside both arms")
