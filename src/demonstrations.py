"""The demonstration set D0: scripted-expert episodes on the task pairs (spec section 6).

What this file does, in one sentence: `make_demonstrations(maze, per_pair, seed)`
runs the scripted expert `per_pair` times on each west-east task pair, exactly
half of them on the north arm and half on the south arm, and returns the
episodes in a seed-shuffled order, ready for `TrajectoryBuffer.add`.

Why it matters: mode collapse presupposes that both modes were there to begin
with, so the expert demonstrates both routes for *every* pair, with fixed
counts rather than random draws (a random draw could leave a pair lopsided by
chance). Tie pairs are split the same way. Within a seed every condition shares
the same D0, and so the same pretrained policy (spec section 7, paired seeds).

Settings agreed by Siddharth on 30 Oct (Stage 2 setting 4): 20 per pair, so 140
demonstrations on M1 (seven pairs) and 100 on M0 (five pairs). Every episode
lasts exactly T = EPISODE_STEPS steps; the expert keeps holding the goal once it
arrives. Reset noise comes from seeds that do not overlap the evaluation seeds
(a separate SeedSequence tag), and the environment refuses goals outside the
demo region, so demonstrations never command a held-out goal.
"""

import numpy as np

from envs.mazes import RINGS
from envs.pointmaze import make_env
from eval.protocol import task_pairs
from expert import run_episode

DEMOS_PER_PAIR = 20
_SEED_TAG = 0xD0  # separates demonstration seeds from evaluation seeds


def demonstration_plan(maze, per_pair, seed):
    """(pair, route, reset seed) for every demonstration, in seed-shuffled order."""
    if per_pair <= 0 or per_pair % 2:
        raise ValueError("per_pair must be a positive even number (half per route)")
    plan = []
    for pair in task_pairs(maze):
        for index in range(per_pair):
            route = "north" if index < per_pair // 2 else "south"
            reset_seed = int(np.random.SeedSequence([seed, _SEED_TAG, pair.row, index]).generate_state(1)[0])
            plan.append((pair, route, reset_seed))
    order = np.random.default_rng(np.random.SeedSequence([seed, _SEED_TAG])).permutation(len(plan))
    return [plan[i] for i in order]


def make_demonstrations(maze="M1", per_pair=DEMOS_PER_PAIR, seed=0):
    """Run the plan. Each result has observations, actions, goal, row, route and success."""
    grid = RINGS[maze]["grid"]
    env = make_env(maze)
    try:
        demonstrations = []
        for pair, route, reset_seed in demonstration_plan(maze, per_pair, seed):
            episode = run_episode(env, grid, pair.start, pair.goal, route=route, seed=reset_seed)
            demonstrations.append({
                "observations": episode["observations"], "actions": episode["actions"],
                "goal": episode["goal"], "row": pair.row, "route": route,
                "success": episode["success"], "first_success_step": episode["first_success_step"],
            })
        return demonstrations
    finally:
        env.close()
