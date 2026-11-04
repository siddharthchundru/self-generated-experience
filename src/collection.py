"""Collection: one practice episode for each condition (spec section 6).

What this file does, in one sentence: `Collector.run_episode(i)` runs the i-th
practice episode of one condition (expert top-up, random actions, task-directed
practice or frontier novelty) and returns the trajectory to store and a record
to log; demonstrations only has no collector, because it collects nothing.

Why it matters: the conditions must differ only in which goal each episode
pursues (and, for the expert and random references, in who acts). Everything
else is shared and built the same way here:
- Start draws. Episode i of every collecting condition in a seed starts at the
  start cell of the same task pair with the same reset noise. The pair is drawn
  uniformly from the task pairs, from a stream that depends only on (seed, i),
  so it does not depend on the condition or on anything the policy did.
- Fixed length. Every episode is exactly T steps; reaching the goal does not
  end it, and the goal never moves.
- Goal leakage (spec section 7). A held-out goal may be commanded only under
  frontier novelty. Every other condition's environment refuses held-out goal
  cells, and each episode's commanded goal is checked again here.
- Who acts and what is stored. Learned policies act at temperature 1 and see
  only position, velocity and the goal; the stored trajectory holds the
  observations, the executed actions and the commanded goal, never the task
  pair or a route label.

Per condition:
- Task-directed practice: commands the drawn pair's goal (cell centre plus the
  environment's reset noise).
- Frontier novelty: commands the exact position chosen by `frontier.py`, which
  may lie in the held-out region (its environment allows those cells).
- Random actions: commands the drawn pair's goal (irrelevant to behaviour; the
  learner relabels with goals actually reached). Actions are independent each
  step, either continuous uniform forces in [-1, 1] (the Stage 1 stand-in's
  rule, current default) or uniform over the 81 grid actions (audit of 4 Nov,
  decision 5, not yet decided).
- Expert top-up: the scripted expert drives the drawn pair, the route set by
  `topup_routes`: "alternate" (current default) takes north and south in turn
  on each pair, starting from a seed-chosen route, so each pair's top-up data is
  50/50 between routes to within one episode (audit of 4 Nov, decision 1, not
  yet decided).

Undecided settings exposed here (audit of 4 Nov, "Decisions for you"): the
start distribution (decision 4; default above, "task_pair_starts"), random
actions' rule (decision 5) and expert top-up's route split (decision 1).
"""

from dataclasses import dataclass

import numpy as np

from envs.mazes import M0_HELD_OUT_REGION, M1_HELD_OUT_REGION, RINGS
from envs.pointmaze import DEFAULT_ALLOWED_GOALS, make_env, xy_to_cell
from eval.protocol import task_pairs
from eval.suite import policy_view
from expert import ScriptedExpert
from gcsl import NUM_ACTIONS, bins_to_actions
from routes import trajectory_route

# Seed tags keep each random stream separate from the others, from the
# demonstrations (0xD0) and from evaluation (whose seeds start from 20261028).
TAG_STARTS = 0xC51
TAG_ACTIONS = 0xC52
TAG_TOPUP = 0xC53
TAG_FRONTIER = 0xC54

TASK_DIRECTED = "task_directed_practice"
FRONTIER = "frontier_novelty"
RANDOM = "random_actions"
EXPERT = "expert_top_up"
DEMOS_ONLY = "demonstrations_only"
COLLECTING = (EXPERT, RANDOM, TASK_DIRECTED, FRONTIER)
CONDITIONS = (DEMOS_ONLY,) + COLLECTING
PLAIN_CONDITIONS = (TASK_DIRECTED, FRONTIER)  # the plain-learner arm (spec section 6, H6)

START_DISTRIBUTIONS = ("task_pair_starts",)
RANDOM_ACTION_RULES = ("continuous", "grid")
TOPUP_ROUTE_RULES = ("alternate",)


def held_out_cells(maze):
    """Free cells outside the demo region (empty on M2, which has none)."""
    return {"M0": M0_HELD_OUT_REGION, "M1": M1_HELD_OUT_REGION}.get(maze, frozenset())


def fork_cells(maze):
    """The cell where each shorter-route pair's two routes part: its start cell.

    From (row, west column) the north route climbs the west column and the south
    route descends it, so this is where a west-to-east rollout commits to a
    route. Tie rows are excluded. Used for the exposure diagnostic (spec section
    8): how much of each condition's data sits where the route is chosen.
    """
    return {pair.row: pair.start for pair in task_pairs(maze) if pair.shorter is not None}


@dataclass(frozen=True)
class StartDraw:
    pair: object
    reset_seed: int


def start_draw(maze, seed, index, distribution="task_pair_starts"):
    """Start of collection episode `index`: identical in every condition of a seed."""
    if distribution not in START_DISTRIBUTIONS:
        raise ValueError(f"unknown start distribution {distribution!r}")
    pairs = task_pairs(maze)
    sequence = np.random.SeedSequence([seed, TAG_STARTS, index])
    pair = pairs[int(np.random.default_rng(sequence).integers(len(pairs)))]
    reset_seed = int(np.random.SeedSequence([seed, TAG_STARTS, index, 1]).generate_state(1)[0])
    return StartDraw(pair, reset_seed)


class Collector:
    """Runs one condition's collection episodes for one run (one seed, one maze)."""

    def __init__(self, condition, maze, seed, policy=None, selector=None,
                 start_distribution="task_pair_starts", random_actions="continuous",
                 topup_routes="alternate", episode_steps=None):
        if condition not in COLLECTING:
            raise ValueError(f"{condition!r} does not collect")
        if random_actions not in RANDOM_ACTION_RULES or topup_routes not in TOPUP_ROUTE_RULES:
            raise ValueError("unknown random-action or top-up route rule")
        if (condition in (TASK_DIRECTED, FRONTIER)) != (policy is not None):
            raise ValueError("task-directed practice and frontier novelty act with the policy; "
                             "the expert and random references do not")
        if (condition == FRONTIER) != (selector is not None):
            raise ValueError("only frontier novelty uses the frontier selector")
        self.condition = condition
        self.maze = maze
        self.seed = seed
        self.policy = policy
        self.selector = selector
        self.start_distribution = start_distribution
        self.random_actions = random_actions
        self.topup_routes = topup_routes
        self.grid = RINGS[maze]["grid"]
        self.demo_region = DEFAULT_ALLOWED_GOALS[maze]
        self.held_out = held_out_cells(maze)
        allowed = self.demo_region | self.held_out if condition == FRONTIER else self.demo_region
        kwargs = {} if episode_steps is None else {"max_episode_steps": episode_steps}
        self.env = make_env(maze, allowed_goals=allowed, **kwargs)
        if condition != FRONTIER and self.env.allowed_goals & self.held_out:
            raise AssertionError("only frontier novelty's environment may allow held-out goals")
        self.horizon = self.env.spec.max_episode_steps
        self._topup_counts = {}

    def close(self):
        self.env.close()

    def _topup_route(self, pair):
        count = self._topup_counts.get(pair.row, 0)
        self._topup_counts[pair.row] = count + 1
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, TAG_TOPUP, pair.row]))
        first = str(rng.choice(["north", "south"]))
        other = "south" if first == "north" else "north"
        return first if count % 2 == 0 else other

    def run_episode(self, index, after_step=None):
        """Run collection episode `index`.

        `after_step(step)` is called after each environment step (1..T); the
        training runner uses it for the interleaved update schedule. Returns
        (trajectory, record).
        """
        draw = start_draw(self.maze, self.seed, index, self.start_distribution)
        pair = draw.pair
        obs, _ = self.env.reset(seed=draw.reset_seed,
                                options={"reset_cell": list(pair.start), "goal_cell": list(pair.goal)})
        record = {"episode": index, "condition": self.condition, "start_row": pair.row}
        if self.condition == FRONTIER:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, TAG_FRONTIER, index]))
            goal_xy, choice = self.selector.choose(rng)
            obs = self.env.set_goal_position(obs, goal_xy)
            record["frontier"] = choice
        goal = obs["desired_goal"].copy()
        goal_cell = xy_to_cell(self.grid, goal)
        in_held_out = goal_cell in self.held_out
        if in_held_out and self.condition != FRONTIER:  # goal-leakage rule, spec section 7
            raise AssertionError(f"{self.condition} commanded a held-out goal {goal_cell}")
        if goal_cell not in self.env.allowed_goals:
            raise AssertionError(f"commanded goal cell {goal_cell} is not allowed")

        action_rng = np.random.default_rng(np.random.SeedSequence([self.seed, TAG_ACTIONS, index]))
        if self.condition in (TASK_DIRECTED, FRONTIER):
            act_on_view = self.policy.start_episode(policy_view(obs), action_rng, sampled=True)
            act = lambda o: act_on_view(policy_view(o))
        elif self.condition == RANDOM:
            if self.random_actions == "continuous":
                act = lambda o: action_rng.uniform(-1, 1, size=2).astype(np.float32)
            else:
                act = lambda o: bins_to_actions(int(action_rng.integers(NUM_ACTIONS)))
        else:  # expert top-up
            route = self._topup_route(pair)
            expert = ScriptedExpert(self.grid, pair.start, pair.goal, goal, route=route)
            act = lambda o: expert.act(o["observation"])
            record["expert_route"] = route

        observations, actions = [obs["observation"].copy()], []
        first_success, info = None, {"success": False}
        for step in range(1, self.horizon + 1):
            action = np.asarray(act(obs), dtype=np.float32)
            obs, _, terminated, truncated, info = self.env.step(action)
            observations.append(obs["observation"].copy())
            actions.append(action)
            if info["success"] and first_success is None:
                first_success = step
            if not np.array_equal(obs["desired_goal"], goal):
                raise AssertionError("goal moved during a collection episode")
            if terminated or (truncated != (step == self.horizon)):
                raise AssertionError("collection episode violated the fixed length")
            if after_step is not None:
                after_step(step)
        observations = np.array(observations)
        record.update({
            "goal": [float(v) for v in goal], "goal_cell": list(goal_cell),
            "goal_in_held_out": in_held_out, "success": bool(info["success"]),
            "first_success_step": first_success,
            "route": trajectory_route(observations[:, :2], self.maze),
        })
        trajectory = {"observations": observations, "actions": np.array(actions), "goal": goal}
        return trajectory, record
