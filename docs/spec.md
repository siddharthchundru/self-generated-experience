# Design specification

**Working title:** What Should a Robot Practise? Goal Selection, Behavioural Repertoire and Recovery
**Repository:** `self-generated-experience`
**Version:** 0.2 · October 2026

This document is the reference for implementation. Every number in it is a planning default or a hypothesis until the pilot fixes it.

---

## 1. Question

A goal-conditioned agent starts from a few demonstrations that cover only part of its world, then improves through a fixed budget of its own practice. With the learner, the demonstrations and the budget held fixed, how does **the choice of what to practise** affect:

1. **competence** on the tasks it was shown,
2. its **behavioural repertoire**, meaning the distinct ways it can solve the same task, and
3. its **recovery** when the solution it relies on is blocked?

## 2. Position and contribution

**Already established.**
- When a demonstration-pretrained policy is improved with task reward, its behavioural modes collapse ([Longhini et al., 2026](https://arxiv.org/abs/2605.11387); [Ibn Hamid et al., 2025](https://arxiv.org/abs/2509.25424)).
- The published fixes add an *explicit* diversity objective.
- Deployment-time recovery methods ([Du et al., 2024](https://arxiv.org/abs/2406.15917)) assume the base policy still has alternative modes to fall back on.

**What this project tests.**
1. Does *choosing which goals to practise* (frontier novelty, as in [MEGA](https://arxiv.org/abs/2007.02832)) preserve repertoire on its own, with the learner and its objective unchanged?
2. Does a preserved repertoire actually buy recovery?
3. Through a re-learning control: does any advantage live in the collected data, or in the active loop that collected it?

**Type of contribution.** A controlled empirical study, not a new method. The literature check is in [`notes/stage0-literature-gate.md`](../notes/stage0-literature-gate.md).

**Role in the programme.** `self-generated-experience` produces families of pretrained policies whose repertoire differs in a controlled way. `switch-or-recalibrate` uses them as base policies.

## 3. Hypotheses (directional; registered before the main run)

- **H1 (competence).** Task-directed self-practice (C3) reaches higher in-region competence than frontier novelty (C4) at budget N.
- **H2 (repertoire).** C4 preserves more route classes than C3.
- **H3 (recovery).** C4 recovers better than C3 after its dominant route is blocked, both zero-shot and after a common adaptation procedure.
- **H4 (locus).** At least part of C4's repertoire advantage survives retraining a fresh learner on C4's final data. This hypothesis is exploratory and is not part of the confirmatory family.

A clear null on H2 and H3, shown by equivalence tests, counts as an answer.

## 4. Environment

- **Simulator.** [Gymnasium-Robotics PointMaze](https://robotics.farama.org/envs/maze/point_maze/): a force-actuated 2-DoF ball in MuJoCo at 10 Hz. Observation: position and velocity. Goal: a 2-D position. Action: a 2-D force in [−1, 1]. An episode ends when the ball is within 0.45 of the goal (the library's success radius) or after a fixed number of steps (100 by default, set in the pilot). Start and goal are placed at the centre of their cells plus uniform noise of up to ±0.25 cell widths. Reward is sparse and is logged, not learned from.
- **M1, custom, primary.** Key start–goal pairs have exactly two topologically distinct routes, one clearly shorter.
  - The *demo region* R_d contains the two-route structure. A separate *held-out region* R_h is reached through a corridor.
  - The layout is frozen before any condition runs, and route classes are verified by a planner.
- **Pairs in M1.** Key pairs run west–east within the ring. Upper-row pairs are shorter via the north arm, lower-row pairs via the south arm, and the middle row is a tie (excluded from shorter-route pairs). The planner tests are in `tests/test_mazes.py`.
- **M2.** The standard Medium layout, used for comparability and replication only.
- **M0, tuning maze.** A separate custom layout with the same two-route structure but different geometry, used only for hyperparameter tuning so that M2 stays an independent replication.
- **Perturbations, applied at evaluation only.**
  - P1, route block: a wall cell is placed on each agent's own most-used route. This is the primary perturbation.
  - P1-fixed: the same block (north arm) for every condition. This is a secondary check, so that conditions are also compared on an identical maze.
  - P2, action gain or noise.
  - P3, held-out start cells.
- **Fallback.** If MuJoCo throughput is too low for the seed budget, switch to a vectorised NumPy point-mass with the same grid semantics.

## 5. Learner (identical in every condition)

- **Algorithm.** Goal-conditioned supervised learning ([GCSL](https://arxiv.org/abs/1912.06088)): imitate one's own trajectories, relabelled with goals they actually reached. Expert demonstrations and self-collected data use the same loss.
- **Policy.** An MLP with 2×256 units and ReLU, taking observation and goal as input, with a **categorical head of about 9 bins per action dimension**. A Gaussian head averages modes and cannot represent two routes, so this choice is essential, not a default.
- **Fixed across conditions.** Warm-up steps on D0, gradient steps per environment step, batch size, and sampling temperature.
- **Action selection.** Greedy for competence evaluation; sampled for collection and for repertoire evaluation.
- **Inputs.** Observation and goal only. The previous action is deliberately not an input, because imitation learners given their own previous action tend to copy it (causal confusion; de Haan et al., 2019). Sensing of the command–outcome relation is handled in `switch-or-recalibrate` (its spec, §7.3).

## 6. Conditions

All conditions share D0, the budget N, the episode length and the start distribution. D0 is N_d scripted-expert trajectories inside R_d, built from pairs split equally between north-shorter and south-shorter rows, so both routes are demonstrated. Mode collapse presupposes that both modes were present to begin with. They differ only in which goal is pursued in each collection episode.

| ID | Condition | Collection | Stage |
| --- | --- | --- | --- |
| C0 | Demonstrations only | None; matched gradient steps | MVE |
| C1 | Expert top-up | More scripted-expert trajectories within budget | MVE if compute allows |
| C2 | Random actions | Uniform random actions | MVE |
| C3 | Task-directed self-practice | Pursue task goals sampled from R_d | MVE |
| C4 | Frontier novelty | Pursue past achieved goals in the lowest-density region (MEGA-style kernel density estimate) | MVE |
| C5 | Mixture | C3 with probability *p*, otherwise C4 | Stage 3 |
| C6 | Learning progress | ALP-GMM-style goal selection | Stage 3 |
| C7 | Explicit diversity | C3 plus a BMD-style mutual-information bonus; compares implicit with explicit preservation | Stage 3 |

## 7. Controls

- **Paired seeds.** Within a seed, every condition shares D0, initial weights and evaluation pairs.
- **Re-learning control.** A fresh learner is trained on each run's final buffer with matched gradient steps (following Ostrovski et al., 2021).
- **Budget audit.** Episode counts, environment steps and gradient steps are logged and checked equal across conditions. This check is automated.
- **Learner-capacity check.** GCSL trained on plentiful whole-maze expert data must reach high success on M1 and M2.
- **C4 sanity check.** C4's state coverage must exceed C2's.
- **Hyperparameter fairness.** Tuning is done on M0, maximising the average of C3 and C4, with equal trials per condition, then frozen before any M1 or M2 run.
- **Goal-leakage check.** No evaluation pair may appear as a training goal outside the condition's own rule. Held-out goals must never enter D0. Both are asserted in code.

## 8. Measures

| Outcome | Measure | Status |
| --- | --- | --- |
| Competence | S_in: greedy success on fixed seeded pairs in R_d | **Primary** |
| Repertoire | Rep: mean, over two-route pairs, of the number of route classes that each account for at least 10% of *successful* rollouts out of n_roll sampled rollouts (0–2). The threshold stops single lucky rollouts counting as repertoire. | **Primary** |
| Recovery | A_k: success on blocked pairs after k environment steps of the common adaptation procedure (below) | **Primary** |
| Recovery | A_0 (zero-shot success after the block); A_k under P1-fixed | Secondary |
| Repertoire | Mode-weighted success (as in BMD), route entropy | Secondary |
| Generalisation | S_out: success in R_h | Secondary (sanity check) |
| Data | Coverage of collected data | Secondary |
| Efficiency and robustness | Path ratio; success under P2 and P3 | Secondary |

Repertoire is measured on the policy and coverage on the data. A disagreement between the two is a result in itself.

**Common adaptation procedure (used for A_k).** Starting from the final policy, every condition runs C3-style practice on the blocked pairs for k environment steps, with the same gradient-step ratio as training. Practice begins with an empty buffer, so no condition reuses its own old data. k is 10% of N by default, and is set in the pilot.

## 9. Analysis

- **Pre-registration on OSF before the main run.** It fixes hypotheses, endpoints, smallest effects of interest, seed count and analysis code. The repository holds a copy for convenience.
- **Seed count from a pilot** (6 seeds per condition, because fewer gives an unreliable variance estimate), using a power analysis on paired-difference variance. Capped at 20 seeds per condition per maze.
- **Primary analysis.**
  - Paired C4 − C3 differences, with 95% bootstrap confidence intervals and paired permutation tests.
  - Holm correction across the three primary endpoints.
  - Two one-sided tests (TOST) for equivalence when a difference is not significant.
- **Aggregates** reported with rliable: interquartile means, performance profiles and probability of improvement.
- **Every run reported,** including failed ones.

## 10. Plan and gates

| Stage | Work | Gate |
| --- | --- | --- |
| 0 | Literature gate ✔; pre-registration draft | Comparison not already published ✔ |
| 1 | Harness: repo skeleton, determinism test, M1 with planner check, scripted expert, evaluation suite, route classifier (hand-checked on at least 50 trajectories), throughput measurement | Expert reaches ~100% on M1; a full evaluation runs end to end |
| 2 | GCSL with capacity check; C0–C4; tuning on M0; pilot; pre-registration frozen; main runs on M1; re-learning control; report v1 | Every primary endpoint decided, or a reformulation documented |
| 3 | M2 replication; C5–C7; a TD learner (SAC with HER); policy-class and D0-mix ablations | Conclusions survive a second learner and a second maze, or the report says they don't |
| 4 | Port to a goal-conditioned manipulation task with two solutions | Scoped later |

**Development process.** Each stage's implementation is reviewed by an independent auditing agent for budget mismatches, goal leakage and extra gradient updates. Audit reports are committed to `notes/audits/`.
