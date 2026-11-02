"""Stand-in policies for the harness gate; neither is a learned policy.

The expert has privileged base-maze planning, but is NOT told evaluation walls.
Greedy chooses the shorter route; sampled selects north/south with equal chance
once per episode. Random actions are independent uniform forces each step.

Both are marked `stand_in = True`: the evaluation suite passes them the task
pair, which no learned policy ever receives (audit of 28 Oct, S3), and exempts
them from the parameter checks that learned policies must pass.
"""

from dataclasses import dataclass

import numpy as np

from envs.mazes import RINGS
from expert import ScriptedExpert


@dataclass(frozen=True)
class ExpertPolicy:
    name: str = "expert"
    stand_in: bool = True

    def start_episode(self, observation, pair, rng, sampled):
        route = str(rng.choice(["north", "south"])) if sampled else (pair.shorter or "north")
        expert = ScriptedExpert(RINGS[pair.maze]["grid"], pair.start, pair.goal,
                                observation["desired_goal"], route=route)
        return lambda obs: expert.act(obs["observation"])


@dataclass(frozen=True)
class RandomPolicy:
    name: str = "random"
    stand_in: bool = True

    def start_episode(self, observation, pair, rng, sampled):
        # Uniform-action baseline has no logits and therefore no greedy mode.
        # We deliberately use its usual random behaviour in 'greedy' slots too.
        return lambda obs: rng.uniform(-1, 1, size=2).astype(np.float32)
