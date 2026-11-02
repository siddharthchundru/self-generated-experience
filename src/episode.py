"""The episode length T, defined once for the whole project.

What this file does, in one sentence: it holds EPISODE_STEPS, the fixed number
of 10 Hz steps in every episode, which the simulator, the evaluation suite, the
learner's buffer and the compute estimate all import from here.

Why it matters: T sets the practice budget's unit (N must be a whole number of
episodes), the length of every stored trajectory (and so every relabelling
draw), how long the longer route has to be completed, and the evaluation
horizon. Before Stage 2 the value 100 was written separately in three places
(`pointmaze.MAX_EPISODE_STEPS`, `EvaluationConfig.horizon`, the compute
estimate); if one of them drifted, conditions or evaluations could silently use
different episode lengths. One constant removes that possibility.

The value (agreed by Siddharth, 30 Oct 2026; spec section 6, episode-length
rule): T = 60 steps, 6 s. The scripted expert's slowest arrival on the longer
route with the P1 block in place is step 42, leaving 18 steps of margin, and
every free-cell pair on M0, M1 and M2 is reachable by the expert within 60
steps (audit `notes/audits/2026-10-30-data-diversity-and-stage2-settings.md`).
The pilot confirms it; after the pilot it is not changed.

Scripts that deliberately run longer episodes for a different purpose (for
example `scripts/measure_route_gap.py`, which needs every expert episode to
arrive) say so where they do it; they never feed training or evaluation.
"""

EPISODE_STEPS = 60


def whole_episodes(steps: int, name: str = "budget") -> int:
    """Return how many episodes `steps` contains, refusing partial episodes.

    The spec requires every budget (N, warm-up collection, adaptation k) to be a
    whole number of episodes, so a run never ends partway through one.
    """
    if steps < 0 or steps % EPISODE_STEPS:
        raise ValueError(f"{name} of {steps} steps is not a whole number of "
                         f"{EPISODE_STEPS}-step episodes")
    return steps // EPISODE_STEPS
