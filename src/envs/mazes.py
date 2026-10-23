"""Maze layouts for Project A, and the planner checks that verify them.

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
#   A corridor leaves the ring at (4, 7) and leads east to a small held-out
#   region that demonstrations never visit.
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

# Regions. The demo region is the ring; the held-out region is
# everything east of it.
DEMO_COLS = range(1, 8)
NORTH_ARM: frozenset[Cell] = frozenset((1, c) for c in range(2, 7))
SOUTH_ARM: frozenset[Cell] = frozenset((7, c) for c in range(2, 7))

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


# --- Graph helpers ---------------------------------------------------------------

def free_cells(grid: list[list[int]]) -> set[Cell]:
    return {(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == 0}


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


def route_class(path: list[Cell]) -> str:
    """Classify a cell sequence by which arm of the M1 ring it passes through.
    Trajectories from the simulator are converted to cell sequences before this is
    called; that conversion is checked by hand on 50+ trajectories before the main runs."""
    cells = set(path)
    north, south = bool(cells & NORTH_ARM), bool(cells & SOUTH_ARM)
    if north and south:
        return "both"
    if north:
        return "north"
    if south:
        return "south"
    return "neither"
