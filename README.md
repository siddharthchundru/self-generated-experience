# What Should a Robot Practise?

Goal selection, behavioural repertoire and recovery in self-improving policies.

> **Status: Stage 1 (harness) in progress.**
> - Done: the literature check, and the primary maze with automated planner checks.
> - Next: simulator integration, the scripted expert, the evaluation suite, the route classifier and the determinism test.
> - No experiments have been run and no results are reported. The plan is fixed in a pre-registration before the main runs.

## Question

A goal-conditioned agent starts from a few demonstrations that cover only part of its world, then improves through a fixed budget of its own practice. With the learner, the demonstrations and the budget held fixed, how does **the choice of what to practise** affect:

1. **competence** on the tasks it was shown,
2. its **behavioural repertoire**, meaning the number of distinct ways it can solve the same task, and
3. its **recovery** when the solution it relies on is blocked?

## Why this matters

Robot policies are increasingly pretrained on demonstrations and then improved through their own practice ([Ghasemipour et al., 2025](https://arxiv.org/abs/2509.15155); [Zhou et al., 2024](https://arxiv.org/abs/2407.20635)). When that improvement is driven by task reward, behavioural modes have been shown to collapse onto the one that earns reward ([Longhini et al., 2026](https://arxiv.org/abs/2605.11387); [Ibn Hamid et al., 2025](https://arxiv.org/abs/2509.25424)). Yet recovery methods used at deployment assume the base policy still has alternative behaviours to fall back on ([Du et al., 2024](https://arxiv.org/abs/2406.15917)).

The published fixes for mode collapse add an explicit diversity objective. This project asks whether something simpler is enough: **choosing which goals to practise**.

## What this tests that prior work has not

- **Implicit vs explicit diversity.** Does practising self-chosen frontier goals ([MEGA](https://arxiv.org/abs/2007.02832)) preserve repertoire, with the learner and its objective unchanged?
- **Repertoire → recovery, tested directly.** Does a preserved repertoire actually buy recovery after a route is blocked?
- **Data vs loop.** A re-learning control trains a fresh learner on each condition's final data, to locate where any advantage lives ([Ostrovski et al., 2021](https://arxiv.org/abs/2110.14020)).

This is a controlled empirical study, not a new method. The claim of novelty rests on a first-pass literature check, in [`notes/stage0-literature-gate.md`](notes/stage0-literature-gate.md).

## Design

| Element | Choice |
| --- | --- |
| Environment | [Gymnasium-Robotics PointMaze](https://robotics.farama.org/envs/maze/point_maze/). The primary maze, M1, is a ring around a central wall: west–east pairs have exactly two routes, and which is shorter depends on the row. A corridor leads to a held-out region that demonstrations never visit. |
| Demonstrations | Scripted-expert trajectories inside the demonstration region, split equally between the two routes |
| Learner | Goal-conditioned supervised learning ([GCSL](https://arxiv.org/abs/1912.06088)), identical in every condition |
| Policy | MLP with a **categorical action head**. A single-Gaussian head averages the two routes into the wall and cannot represent a repertoire. |
| Varies | Only which goal is pursued in each practice episode |
| Held fixed | Demonstrations, interaction budget, update schedule, gradient steps per environment step |

| Condition | What drives practice |
| --- | --- |
| C0 Demonstrations only | Nothing (matched gradient steps) |
| C1 Expert top-up | More scripted-expert data within the budget |
| C2 Random actions | Uniform random actions |
| C3 Task self-practice | Goals sampled from the task distribution |
| C4 Frontier novelty | Past achieved goals in the least-visited regions |

Mixture, learning-progress and explicit-diversity conditions are planned for later stages.

## Measures

| Primary endpoint | Definition |
| --- | --- |
| Competence (S_in) | Success on fixed start–goal pairs in the demonstration region |
| Repertoire (Rep) | Mean, over two-route pairs, of the number of routes (0–2) that each account for at least 10% of successful sampled rollouts |
| Recovery (A_k) | Success after the agent's own most-used route is blocked, following a common adaptation procedure |

Secondary measures: held-out success, zero-shot recovery, mode-weighted success, data coverage, path efficiency and robustness.

**Analysis.**
- Hypotheses, endpoints and the analysis code will be pre-registered on OSF before the main runs.
- Paired seeds; seed count set by a pilot power analysis.
- Paired C4 − C3 differences with bootstrap 95% confidence intervals; paired permutation tests, Holm-corrected across the three primary endpoints; equivalence tests for null results.
- Every run is reported.

The full design is in [`docs/spec.md`](docs/spec.md).

## Roadmap

- [x] **Stage 0:** literature check
- [ ] **Stage 1:** harness
  - [x] maze M1 with planner checks
  - [ ] simulator integration, scripted expert, evaluation suite, route classifier, determinism test
- [ ] **Stage 2:** learner, conditions, pilot, frozen pre-registration, main runs, re-learning control, report
- [ ] **Stage 3:** second maze, second learner, further conditions and ablations
- [ ] **Stage 4:** a goal-conditioned manipulation task with two solutions

Each stage ends in a gate that must pass before the next starts. Before a stage closes, its code is reviewed by a separate AI coding agent (not the one that wrote it) for budget parity, goal leakage and extra gradient updates; those review reports will be committed to `notes/audits/`.

## Repository

```
src/envs/mazes.py        maze layouts and planner checks           (done)
tests/test_mazes.py      tests for the maze structure              (done)
notes/                   literature check
docs/spec.md             full design specification
src/agents/, src/eval/   learner, goal selectors, evaluation       (Stage 1–2)
```

Run the tests (Python 3.10+, no other dependencies yet):

```
python -m unittest discover -s tests
```

## Related project

[switch-or-recalibrate](https://github.com/siddharthchundru/switch-or-recalibrate) takes this project's policies, trained under practice conditions predicted to differ in repertoire (measured, not assumed), and studies how they adapt at test time to world changes and body changes.

## Author

Siddharth Chundru · [siddharthchundru.github.io](https://siddharthchundru.github.io) · sbchundru@gmail.com
