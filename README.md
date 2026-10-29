# What Should a Robot Practise?

Goal selection, behavioural repertoire and recovery in self-improving policies.

> **Status: Stage 1 (harness) nearly complete.** This repository is a tested experimental apparatus. No learning experiments have been run, and the competence–repertoire–recovery trade-off below is a hypothesis to be tested, not a finding.
> - Done: the literature check; the mazes with planner checks; the MuJoCo PointMaze simulator at 10 Hz with fixed-length episodes and guards against goal leakage; a scripted expert that drives either route on every two-route pair; the route-block perturbation; the measured route-length gap that fixes the learner's discount (γ = 0.962). Each piece was reviewed by an independent audit ([`notes/audits/`](notes/audits/)).
> - Implemented for review: the route classifier and M0; a 60-trajectory human check (56 certain labels, all agreeing); the end-to-end ring-task evaluation on expert/random stand-ins; Mac throughput and total-compute scenarios. Details and limitations: [`docs/stage1-evaluation.md`](docs/stage1-evaluation.md).
> - Next: independent review of the remaining Stage 1 changes, then the learner. Stand-in adaptation performs no parameter updates. The plan is fixed in a pre-registration before the main runs.

## Question

A goal-conditioned agent starts from a few demonstrations that cover only part of its world, then improves through a fixed budget of its own practice. With the learner, the demonstrations and the budget held fixed, how does **the choice of what to practise** affect:

1. **competence** on the tasks it was shown,
2. its **behavioural repertoire**, meaning the distinct ways it can solve the same task, and
3. its **recovery** when the solution it relies on is blocked?

## Why this matters

Robot policies are increasingly pretrained on demonstrations and then improved through their own practice ([Ghasemipour et al., 2025](https://arxiv.org/abs/2509.15155); [Zhou et al., 2024](https://arxiv.org/abs/2407.20635)). When that improvement is driven by task reward, behavioural modes have been shown to collapse onto the one that earns reward ([Longhini et al., 2026](https://arxiv.org/abs/2605.11387); [Ibn Hamid et al., 2025](https://arxiv.org/abs/2509.25424)). Yet recovery methods used at deployment assume the base policy still has alternative behaviours to fall back on ([Du et al., 2024](https://arxiv.org/abs/2406.15917)).

The published fixes for mode collapse add an explicit diversity objective. This project asks whether something simpler is enough: **choosing which goals to practise**.

## What this tests that prior work has not

- **Implicit vs explicit diversity.** Does practising self-chosen frontier goals ([MEGA](https://arxiv.org/abs/2007.02832)) preserve repertoire, with the learner and its objective unchanged?
- **Repertoire → recovery, inside one policy.** A repertoire that can be *selected from* is known to aid recovery ([Cully et al., 2015](https://arxiv.org/abs/1407.3501); [Li et al., 2024](https://arxiv.org/abs/2406.00681)). Does a repertoire held inside one goal-conditioned policy, with no mechanism for selecting a mode, speed recovery when the route it relies on is blocked?
- **Data vs loop.** A re-learning control trains a fresh learner on each condition's final data, to locate where any advantage lives ([Ostrovski et al., 2021](https://arxiv.org/abs/2110.14020)).

This is a controlled empirical study, not a new method. The claim of novelty rests on a first-pass literature check, in [`notes/stage0-literature-gate.md`](notes/stage0-literature-gate.md).

## Design

| Element | Choice |
| --- | --- |
| Environment | [Gymnasium-Robotics PointMaze](https://robotics.farama.org/envs/maze/point_maze/). The primary maze, M1, is a ring around a central wall: west–east pairs have exactly two routes, and which is shorter depends on the row. A corridor leads to a held-out region that demonstrations never visit. |
| Demonstrations | Scripted-expert trajectories inside the demonstration region, split equally between the two routes |
| Learner | Goal-conditioned supervised learning ([GCSL](https://arxiv.org/abs/1912.06088)) with the discounted relabelling weight from [weighted GCSL](https://arxiv.org/abs/2202.04478), identical in every condition. The weight makes the learner prefer shorter routes, as discounted reward does in RL fine-tuning; its strength γ is set by a rule from the maze's measured route lengths before any learning runs. Plain GCSL (no preference) is a secondary arm. |
| Policy | MLP with a **categorical action head**. A single-Gaussian head averages the two routes into the wall and cannot represent a repertoire. |
| Varies | Only which goal is pursued in each practice episode |
| Held fixed | Demonstrations, network initialisation within a seed, interaction budget (whole episodes of fixed length), update schedule, gradient steps per environment step, buffer and sampling rules, evaluation pairs. Parity is of samples, not compute; wall-clock time is reported. |

| Condition | What drives practice |
| --- | --- |
| C0 Demonstrations only | Nothing (matched gradient steps) |
| C1 Expert top-up | More scripted-expert data within the budget |
| C2 Random actions | Uniform random actions |
| C3 Task self-practice | The demonstrated west–east pairs |
| C4 Frontier novelty | Past achieved goals in the least-visited regions |

Mixture, learning-progress and explicit-diversity conditions are planned for later stages.

## Measures

| Primary endpoint | Definition |
| --- | --- |
| Competence (S_in) | Success on fixed start–goal pairs in the demonstration region |
| Repertoire (Rep) | Minority-route share: of a pair's successful sampled rollouts (100 per pair), the fraction taking the less-used route (0–0.5), averaged over pairs with at least 20 successes. Always read alongside competence and success counts. |
| Recovery (A_k) | Success after the shorter route is blocked, following a common adaptation procedure |

Secondary measures include the minority-route success rate (which separates a competent two-route policy from a barely competent one), zero-shot and best-of-m recovery, per-route quality, held-out success, data coverage, path efficiency and robustness.

**Analysis.**
- Hypotheses, endpoints and the analysis code will be pre-registered on OSF before the main runs.
- Paired seeds; seed count set by a pilot power analysis.
- Paired C4 − C3 differences with bootstrap 95% confidence intervals (for Rep, on pairs where both conditions qualify); paired permutation tests, Holm-corrected across the three primary endpoints; equivalence tests for null results.
- Every run is reported.

The full design is in [`docs/spec.md`](docs/spec.md).

## Roadmap

- [x] **Stage 0:** literature check
- [ ] **Stage 1:** harness
  - [x] maze M1 with planner checks
  - [x] simulator integration and fixed-length episodes (audited)
  - [x] determinism test
  - [x] scripted expert, route-block perturbation, route-length gap and γ (audited)
  - [ ] route classifier with hand-check, tuning maze M0 (implemented; final independent review pending)
  - [ ] evaluation suite and Mac compute estimate (implemented; final independent review pending)
- [ ] **Stage 2:** learner, conditions, pilot, frozen pre-registration, main runs, re-learning control, report
- [ ] **Stage 3:** second maze, second learner, further conditions and ablations
- [ ] **Stage 4:** a goal-conditioned manipulation task with two solutions

Each stage ends in a gate that must pass before the next starts. Before a stage closes, its code is reviewed by a separate AI coding agent (not the one that wrote it) for budget parity, goal leakage and extra gradient updates; those review reports are committed to [`notes/audits/`](notes/audits/).

## Repository

```
src/envs/mazes.py        maze layouts and planner checks           (done)
src/envs/pointmaze.py    PointMaze simulator on these mazes        (done)
src/expert.py            scripted expert (planner + PD controller) (done)
scripts/                 Stage 1 measurements (route gap, γ)       (done)
results/stage1/          measured route gap and γ                  (done)
tests/                   tests for mazes, simulator and expert     (done)
notes/                   literature check
notes/audits/            independent audit reports
docs/spec.md             full design specification
src/eval/               ring-task evaluation and compute accounting (for review)
src/agents/             learner and goal selectors                  (Stage 2)
```

Run the tests (tested on Python 3.13; versions pinned in `requirements.txt`):

```
pip install -r requirements.txt
python -m unittest discover -s tests
```

## Related project

[switch-or-recalibrate](https://github.com/siddharthchundru/switch-or-recalibrate) takes this project's policies, trained under practice conditions predicted to differ in repertoire (measured, not assumed), and studies how they adapt at test time to world changes and body changes.

## Author

Siddharth Chundru · [siddharthchundru.github.io](https://siddharthchundru.github.io) · sbchundru@gmail.com
