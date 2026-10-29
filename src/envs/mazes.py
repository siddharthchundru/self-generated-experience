"""Maze layouts for this project, and the planner checks that verify them.

What this file does, in one sentence: it defines the maze grids (1 = wall, 0 = free)
and checks, with plain graph search, that the primary maze M1 has exactly the
two-route structure the experiment needs.

Why it matters: the repertoire endpoint counts how many distinct routes a policy can
take between the same start and goal. That only means something if the maze has
exactly two routes for the key start-goal pairs, one clearly shorter. This file
proves that property before any learning code exists.

No simulator is needed here. Gymnasium-Robotics PointMaze accepts the same grid
through its `maze_map` argument, so these layouts are used unchanged later.
"""

from collections import deque

Cell = tuple[int, int]  # (row, col)

# --- M1: primary maze (custom) -------------------------------------------------
#
#   cols:  0 1 2 3 4 5 6 7 8 9 ...
#   A ring of corridors around a central wall block (cols 1-7, rows 1-7) forms the
#   demo region. A start on the west side and a goal on the east side can be
#   joined by the NORTH arm (row 1) or the SOUTH arm (row 7). For pairs in the
#   upper rows the north arm is shorter; for pairs in the lower rows, the south.
#   A corridor starts at (4, 8), just east of the ring cell (4, 7), and leads
#   to a small held-out region that demonstrations never visit.
M1: list[list[int]] = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

# Regions, as explicit cell sets. The demo region R_d is the ring; the held-out
# region R_h is every other free cell, from the corridor entry (4, 8) eastward.
# Training goals must come from R_d; R_h is only for the S_out evaluation.
DEMO_COLS = range(1, 8)
NORTH_ARM: frozenset[Cell] = frozenset((1, c) for c in range(2, 7))
SOUTH_ARM: frozenset[Cell] = frozenset((7, c) for c in range(2, 7))

# --- M0: tuning maze (custom) ---------------------------------------------------
#
#   Same two-route structure as M1 but a different shape, so that hyperparameters
#   tuned here are not tuned on M1 (spec section 4; tuning in section 7). The ring
#   is wide and flat (rows 1-5, cols 1-9) instead of square, so its route gaps
#   differ: 8 extra cells for pairs on the arms (rows 1 and 5), 4 for rows 2 and
#   4, and row 3 is a tie. Like M1 it has 24 ring cells and a corridor, from
#   (3, 10), to a held-out region with no loops of its own.
M0: list[list[int]] = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]
M0_DEMO_COLS = range(1, 10)
M0_NORTH_ARM: frozenset[Cell] = frozenset((1, c) for c in range(2, 9))
M0_SOUTH_ARM: frozenset[Cell] = frozenset((5, c) for c in range(2, 9))

# --- M2: standard PointMaze Medium layout (unmodified; from Gymnasium-Robotics) ---
M2: list[list[int]] = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]


# --- Ring descriptions -----------------------------------------------------------
#
# What the expert and the route classifier need to know about a two-route maze:
# its two arms, the columns of the west-east pairs, the rows of the arms, and
# the column whose middle line can only be crossed inside an arm (the central
# wall block spans it in every other row).
RINGS = {
    "M1": {"grid": M1, "north_arm": NORTH_ARM, "south_arm": SOUTH_ARM,
           "west_col": 1, "east_col": 7, "north_row": 1, "south_row": 7,
           "tie_row": 4, "midline_col": 4},
    "M0": {"grid": M0, "north_arm": M0_NORTH_ARM, "south_arm": M0_SOUTH_ARM,
           "west_col": 1, "east_col": 9, "north_row": 1, "south_row": 5,
           "tie_row": 3, "midline_col": 5},
}


def ring_of(grid) -> dict:
    """The ring description for a two-route maze grid (M0 or M1)."""
    for ring in RINGS.values():
        if ring["grid"] == grid:
            return ring
    raise ValueError("this grid has no ring description (only M0 and M1 do)")


# --- Graph helpers ---------------------------------------------------------------

def free_cells(grid: list[list[int]]) -> set[Cell]:
    return {(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == 0}


M1_DEMO_REGION: frozenset[Cell] = frozenset(c for c in free_cells(M1) if c[1] in DEMO_COLS)
M1_HELD_OUT_REGION: frozenset[Cell] = frozenset(free_cells(M1)) - M1_DEMO_REGION
M0_DEMO_REGION: frozenset[Cell] = frozenset(c for c in free_cells(M0) if c[1] in M0_DEMO_COLS)
M0_HELD_OUT_REGION: frozenset[Cell] = frozenset(free_cells(M0)) - M0_DEMO_REGION


def neighbours(cell: Cell, free: set[Cell]) -> list[Cell]:
    r, c = cell
    return [n for n in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)) if n in free]


def shortest_path(grid, start: Cell, goal: Cell, blocked: frozenset[Cell] = frozenset()):
    """Breadth-first search over free cells, avoiding `blocked`. Returns a list of
    cells from start to goal, or None if the goal is unreachable."""
    free = free_cells(grid) - set(blocked)
    if start not in free or goal not in free:
        return None
    parent = {start: None}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        if cell == goal:
            path = []
            while cell is not None:
                path.append(cell)
                cell = parent[cell]
            return path[::-1]
        for n in neighbours(cell, free):
            if n not in parent:
                parent[n] = cell
                queue.append(n)
    return None


def cycle_rank(cells: set[Cell]) -> int:
    """Number of independent loops in the graph of `cells` (edges - nodes + components).
    A rank of exactly 1 means there is one loop, so two cells on opposite sides of it
    are joined by exactly two topologically distinct simple routes."""
    edges = sum(len(neighbours(c, cells)) for c in cells) // 2
    seen, components = set(), 0
    for c in cells:
        if c in seen:
            continue
        components += 1
        stack = [c]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(neighbours(x, cells))
    return edges - len(cells) + components


def route_class(path: list[Cell], ring: dict | None = None) -> str:
    """Classify a planner cell path by which arm of the ring it passes through
    (M1's ring unless another is given). For simulated trajectories use
    `routes.trajectory_route`, which handles a ball that enters one arm and
    turns back."""
    ring = ring or RINGS["M1"]
    cells = set(path)
    north, south = bool(cells & ring["north_arm"]), bool(cells & ring["south_arm"])
    if north and south:
        return "both"
    if north:
        return "north"
    if south:
        return "south"
    return "neither"
