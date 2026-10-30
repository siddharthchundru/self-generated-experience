# Design specification

**Working title:** What Should a Robot Practise? Goal Selection, Behavioural Repertoire and Recovery
**Repository:** `self-generated-experience`
**Version:** 0.10 · October 2026

This document is the reference for implementation. Every number in it is a planning default or a hypothesis, except where it says *measured*.

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
- A diverse repertoire that can be *selected from* aids recovery: a pre-built archive searched after damage ([Cully et al., 2015](https://arxiv.org/abs/1407.3501)), a set of near-optimal policies ([Kumar et al., 2020](https://arxiv.org/abs/2010.14484); [Zahavy et al., 2023](https://arxiv.org/abs/2205.13521)), and, closest to this project, switching between a policy's learned modes when a maze path is blocked ([Li et al., 2024](https://arxiv.org/abs/2406.00681)).

**What this project tests.**
1. Does *choosing which goals to practise* (frontier novelty, as in [MEGA](https://arxiv.org/abs/2007.02832)) preserve repertoire on its own, with the learner and its objective unchanged?
2. Does a repertoire preserved *inside* one goal-conditioned policy, with no latent and no mechanism for selecting a mode, speed recovery under a common adaptation procedure? The existing work recovers by selecting among indexed modes, not by relearning.
3. Through a re-learning control: does any advantage live in the collected data, or in the active loop that collected it?

**Type of contribution.** A controlled empirical study, not a new method. The literature check is in [`notes/stage0-literature-gate.md`](../notes/stage0-literature-gate.md), including an addendum on quality-diversity and repertoire-based recovery.

**Role in the programme.** This project produces families of pretrained policies whose repertoire differs in a controlled way. The companion project, [switch-or-recalibrate](https://github.com/siddharthchundru/switch-or-recalibrate), uses them as base policies.

## 3. Hypotheses (directional; registered before the main run)

**Confirmatory (main learner, M1, shorter-route pairs).**
- **H1 (competence).** Task-directed self-practice (C3) reaches higher in-region competence (S_in) than frontier novelty (C4) at budget N.
- **H2 (repertoire).** C4 keeps a larger minority-route share (Rep) than C3.
- **H3 (recovery).** After the shorter route is blocked, C4 recovers better than C3 after the common adaptation procedure (A_k).
  - *What H3 adds beyond H2.* Under sampled actions, the share of rollouts that commit to the open route at the fork is roughly Rep times competence, and those rollouts are what the adaptation procedure imitates. H3 is therefore expected to follow largely from H2. It can diverge if the preserved minority route is of low quality (it rarely reaches the goal, or reaches it slowly), or if adaptation from an empty buffer erases the head start. Per-route quality (Section 8) is reported to tell these apart, so a result where H2 holds and H3 fails is interpretable.

**Secondary (registered; reported in full; outside the confirmatory family).**
- **H3b (zero-shot recovery).** As H3, measured immediately after the block (A_0), with actions sampled at temperature 1. Under greedy actions a policy that prefers the blocked route takes it every time, whatever its repertoire, so a greedy A_0 would say nothing about repertoire.
- **H3c (recovery by selection).** As H3, measured as best-of-m: success if any of m sampled attempts on a blocked pair succeeds, with no learning. This is how the selection-based recovery literature measures recovery, so it makes the results comparable with it.
- **H4 (locus).** At least part of C4's repertoire advantage survives retraining a fresh learner on C4's final data. Exploratory.
- **H5 (preservation versus inaction).** C4 reaches higher S_in than demonstrations only (C0), while its Rep is within an equivalence margin of C0's. This separates preserving repertoire while improving from preserving it by not practising the task pairs. If C4 matches C0 on both, H2 is reported as preservation without improvement.
- **H6 (does collapse need a preference for shorter routes?).** Plain GCSL is the same learner with γ = 1. With plain GCSL, C3-plain loses less Rep *from its own pretrained starting point* than C3 does. C4-plain − C3-plain is also tested on the three primary endpoints.

A clear null on H2 and H3, shown by equivalence tests, counts as an answer, provided two checks have passed: the precondition check (the pretrained policy has both routes; Section 7) and the pilot collapse check (Section 10, Stage 2). Without them, a null on H2 could mean only that there was nothing to preserve, or that nothing collapses under this learner.

## 4. Environment

- **Simulator.** [Gymnasium-Robotics PointMaze](https://robotics.farama.org/envs/maze/point_maze/): a force-actuated 2-DoF ball in MuJoCo at 10 Hz. Observation: position and velocity. Goal: a 2-D position. Action: a 2-D force in [−1, 1]. The ball's speed is capped at 5 cells per second, so it can cover half a cell per step.
- **M1, custom, primary.** Key start–goal pairs have exactly two topologically distinct routes, one clearly shorter.
  - The *demo region* R_d contains the two-route structure. A separate *held-out region* R_h is reached through a corridor.
  - The layout is frozen before any condition runs, and route classes are verified by a planner.
- **Pairs in M1.** Key pairs run west–east within the ring. Upper-row pairs are shorter via the north arm, lower-row pairs via the south arm. The middle row is a tie; tie pairs are excluded from the primary endpoints and used as a control (Section 8). The planner tests are in `tests/test_mazes.py`.
- **M2.** The standard Medium layout, used for comparability and replication only.
- **M0, tuning maze.** A separate custom layout with the same two-route structure but different geometry, used only for hyperparameter tuning so that M2 stays an independent replication.
- **Perturbations, applied at evaluation only.**
  - **P1, shorter-route block (primary).** On each shorter-route pair, a wall cell blocks the shorter route. The maze is the same for every condition. Under the weighted learner the shorter route is the one policies rely on most, so this tests losing the relied-on solution at equal difficulty for all conditions. Implemented as `make_env(blocked_cells=...)`; the evaluation suite fixes the middle cell of the shorter arm: **(1, 4) for north, (7, 4) for south on M1** (M0: (1, 5) or (5, 5)). The expert succeeds on the longer route with a real wall at the start, middle or end of the shorter arm.
  - **P1-own (secondary).** A wall cell blocks each agent's own most-used route, defined from greedy rollouts. Results are stratified by which route was blocked, because blocking the longer route leaves an easier test.
  - P2, action gain or noise.
  - P3, held-out start cells.
- **Fallback.** If MuJoCo throughput is too low for the seed budget, switch to a vectorised NumPy point-mass with the same grid semantics. *Measured:* about 2,600 environment steps per second in a single process on the cloud workspace; to be re-measured on the Mac.

## 5. Learner (identical in every condition)

- **Algorithm.** Weighted goal-conditioned supervised learning: GCSL ([Ghosh et al.](https://arxiv.org/abs/1912.06088)) with the *discounted relabelling weight* from weighted GCSL ([Yang et al., ICLR 2022](https://arxiv.org/abs/2202.04478)). The policy imitates its own trajectories, relabelled with goals they actually reached. Expert demonstrations and self-collected data use the same loss.
  - Relabelling follows GCSL (Appendix A.1): trajectories are stored whole; at training time a stored trajectory, a start index t and a later goal index t′ are sampled uniformly. The relabelled goal is the *position* part of the state at t′.
  - Each relabelled example's loss is multiplied by γ^(t′−t), and the batch loss is the weighted sum divided by the sum of weights. Only this weight is used. Weighted GCSL's other two weights are advantage-based and need a learned value function, so they are excluded.
  - Training uses the whole buffer, as in GCSL. Details, identical in every condition:
    - *Buffer capacity:* unbounded. Nothing is evicted; the buffer always holds D0 plus every collected trajectory.
    - *Demonstrations:* stay in the buffer for the whole run.
    - *Mix of demonstrations and self-collected data:* no fixed ratio. Each training example draws a stored trajectory uniformly at random, then t and t′ (GCSL's procedure), so D0's share of training falls as data accumulates. Every collecting condition gathers the same number of episodes, so the share falls identically across them at every point in training. Exceptions by design: demonstrations only (C0) has no new data, and expert top-up (C1) adds expert data. A fixed share for D0 was considered and not adopted, because it adds a free parameter.
    - *Relabelled examples per trajectory:* not fixed. All trajectories have length T, so each contributes equally in expectation.
- **Setting γ (a geometry rule, not tuned).**
  - After Stage 1, measure Δ: the median, over shorter-route pairs, of the difference in steps between the scripted expert's longer and shorter routes.
  - Set γ so that the predicted pretrained shorter-route share, 1 / (1 + γ^Δ), equals 0.65; that is, γ = 0.538^(1/Δ).
  - *Measured (Stage 1, 28 Oct):* the expert was run 50 times per route on each shorter-route pair, recording the step at which it first reached the goal; a pair's gap is the median over seeds of longer minus shorter (same seed, so the same start and goal noise). **Δ = 16 steps (1.6 s), so γ = 0.962**, a weight that halves every 18 steps (1.8 s). Record: `results/stage1/route_gap.json` in the repository.
  - γ and its half-life in seconds are reported. The rule uses only maze geometry and the expert, so it is fixed before any practice result is seen. It is not chosen by whether collapse appears.
  - *Why not MEGA's 0.98.* MEGA and weighted GCSL used 0.98, set for their own tasks. The rule instead ties the strength of the preference to this maze and to a target share stated in advance. Here it happens to give a *stronger* preference than 0.98: with Δ = 16, 0.98 would predict a pretrained shorter-route share of 0.58, against the rule's 0.65. (The longest expert route takes 41 steps.) Because γ follows mechanically from the target share and the measurement method, both written down before any learning ran, a stronger-than-MEGA value is not a choice made to produce collapse.
  - **The expert's controller settings are frozen together with γ.** Δ depends on how the expert drives: under other reasonable settings Δ ranged from 15.5 to 26 steps (independent audit, `notes/audits/2026-10-28-expert-and-gamma.md`). Measuring with the expert is still the right definition, because the weighting acts on the step timings in the demonstrations, which come from the same expert; a purely geometric gap in cells would need a speed conversion anyway. The settings (KP = 10, KD = 1.5, switch radius 0.4) and the commit are recorded with Δ. Any later change to the expert reopens Δ and is documented as such.
  - Sensitivity at target shares of 0.58 and 0.72 (γ = 0.980 and 0.943) is run in the pilot only, and reported descriptively.
  - Plain GCSL is the special case γ = 1.
- **Policy.** An MLP with 2×256 units and ReLU, taking observation and goal as input, with a **categorical head over a joint grid of 9 × 9 action bins**: one probability for each combination of x-force and y-force bin.
  - GCSL's released code builds a joint grid over all action dimensions (`DiscretizedActionEnv`, default 3 bins per dimension). Appendix A.1 says only "a discretized grid of actions". The 9 × 9 resolution is our choice.
  - This choice is essential, not a default:
    - a Gaussian head averages modes and cannot represent two routes;
    - independent per-dimension heads cannot represent two good actions that differ on both axes (they would also put mass on the two mixed corners).
- **Fixed across conditions.** γ, warm-up steps on D0, gradient steps per environment step, batch size and episode length.
- **Sampling temperature is fixed at 1 and is not tuned.** A temperature below 1 sharpens whichever route is already the majority, which would cause collapse by itself, even under plain GCSL.
- **Action selection.**
  - Greedy for competence evaluation (S_in) and for recovery after adaptation (A_k).
  - Sampled at temperature 1 for collection, for repertoire evaluation, for zero-shot recovery (A_0) and for best-of-m recovery.
  - *Deviation from GCSL, stated in the paper:* GCSL collects with the greedy action after a random warm-up. We sample during collection, because GCSL's only exploration is the policy's own stochasticity, and greedy collection would starve the repertoire measure of variation.
- **Inputs.** Observation and goal only. The remaining horizon is not an input. GCSL §5.1 describes the horizon as an input but reports that GCSL works without it, and Appendix A.1 specifies the time-invariant policy used in their experiments; we follow A.1. The previous action is deliberately not an input, because imitation learners given their own previous action tend to copy it (causal confusion; de Haan et al., 2019). 
- **Why the learner prefers shorter routes.** Plain GCSL's objective rewards reaching the goal by the end of the episode, not reaching it quickly (GCSL §2). Under plain GCSL, task-directed practice would collapse repertoire only through indirect pressures (timeouts, random drift), which may be too weak to test anything. The published collapse results arise under reward-driven fine-tuning, where reaching goals sooner is preferred through discounting. The discounted relabelling weight gives GCSL that preference. Weighted GCSL shows that this weight tightens the bound on the discounted goal-reaching objective. Its experiments are offline, so using it to induce collapse during practice is our design choice, not a published result. The design treats the weight as making the condition under study possible, under three rules:
  - It resembles how collapse arises in practice: discounting is standard in RL fine-tuning.
  - The rule is the same in every condition. Its effect is not the same everywhere: the weight acts where data concentrates, which for C3 is at the forks of the two-route pairs. That concentration is the mechanism under study, not a confound. But it means the weighting does not make C4's task harder in general. Each condition's exposure at the forks is logged (Section 8), so changes in repertoire can be related to how much each condition practised there.
  - Its strength is fixed by the geometry rule before any result is seen.
  Plain GCSL is kept as a secondary arm (C3-plain, C4-plain; H6), so results are reported both with and without the preference.
- **Expected effect at pretraining.** Weighting also applies to D0, so the pretrained policy starts skewed toward the shorter route but not collapsed. With the rule above, the predicted shorter-route share is 0.65 for the median pair. This is a rough guide only:
  - It ignores generalisation across goals.
  - Episodes have a fixed length (Section 6), so a rollout that arrives early keeps generating relabelled examples at the goal. The formula therefore understates the skew.
  - *The pressure differs by pair (measured).* One γ applies everywhere, but the gaps follow the ring's geometry: about 8 steps for the pairs next to the tie row (rows 3 and 5), 16 for rows 2 and 6, and 25 for the pairs on the arms (rows 1 and 7). The predicted shorter-route shares are 0.58, 0.65 and 0.73. The outer pairs are therefore the most likely to fail the precondition check (predicted minority share about 0.27 before the understatement above) and to collapse fastest in practice; the inner pairs the least. Averaged Rep blurs this, so a pair-level analysis is added (Section 9).
  - *Possible side effect (unverified).* With a half-life of 18 steps, examples whose goal lies more than about 4 seconds ahead get under a fifth of full weight, so the learner learns mostly from short hops. The effective-sample-size diagnostic (Section 8) shows whether long-range examples are starved.
  The predicted share is pre-registered; the precondition check (Section 7) decides whether both routes survive pretraining.

## 6. Conditions

All conditions share D0, the budget N, the episode length and the start distribution. They differ only in which goal is pursued in each collection episode (and, for the plain-GCSL arm, in γ). C0–C2 are reference data sources rather than goal-selection rules; the confirmatory contrast (C4 − C3) is the one that varies only goal selection.

**Task pairs.** The seven west–east ring pairs of M1 (start in column 1, goal in column 7 of the same row; six with a shorter route, one tie) are the *task pairs*. They are the pairs the demonstrations show, the goals task-directed practice pursues, and the pairs every evaluation uses, each time with fresh start and goal noise drawn from fixed evaluation seeds.

**Demonstrations (D0).** N_d scripted-expert trajectories on the task pairs, inside R_d, with pairs split equally between north-shorter and south-shorter rows. **For every two-route pair, exactly half of that pair's demonstrations follow each route** (fixed counts, not random draws, with order shuffled by seed); tie pairs are split the same way. The expert therefore demonstrates both routes for every pair, not only across pairs. Mode collapse presupposes that both modes were present to begin with. Within a route, the expert follows planner waypoints with a PD controller (`src/expert.py`); it drove the commanded route in every tested episode, and its demonstrations on the ring pairs never enter R_h.

**Episodes.**
- Every episode has a fixed length T and does not end when the goal is reached. The goal stays fixed for the episode.
- Success means being within the goal radius at the final step, matching GCSL's objective.
- Equal environment steps therefore mean equal episode counts across conditions.
- The practice budget N is a whole number of episodes, so a run never ends partway through one.

**Episode length T.** Set by a feasibility rule, then confirmed in the pilot: the tightest length at which the scripted expert completes the longer route on every two-route pair with a stated margin (set in the pilot). It must stay loose enough that the longer route is feasible for the demonstrations and for the recovery test, where it is the only open route after the block. Feasibility is also checked with the P1 block in place. Under the weighted learner, timeouts are no longer the main source of pressure toward the shorter route; they remain the main source under plain GCSL. *Measured:* the longest expert route on a two-route pair takes 41 steps, and the longest expert trip between any two free cells of M1 (held-out region included) about 53, so T will be roughly 50–60 steps, close to MEGA's 50. Whether T must also cover frontier novelty's goals in R_h is decided with the margin in the pilot.

**Frontier novelty (C4) implementation, and deviations from MEGA.**
- **Candidates.** 100 candidate goals are drawn uniformly from the achieved-goal buffer: the positions of all states in the condition's own data, D0 included.
- **Density.** A Gaussian kernel density estimate with bandwidth 0.1 on normalised positions, as in MEGA. It is refit at the start of every collection episode on up to 10,000 achieved goals sampled uniformly. MEGA refits at every optimisation step; goal selection happens once per episode, so a per-episode refit selects identically.
- **Selection.** The lowest-density candidate becomes the goal (deterministic, as in MEGA; only the candidate pool is random).
- *Deviation: no value-based achievability cutoff.* MEGA removes candidates whose estimated Q-value falls below an adaptive cutoff. GCSL has no value function, and MEGA's own ablation found that the cutoff slightly hurt performance on PointMaze (Appendix D.4). Candidates come only from goals already achieved, which already favours achievable ones. C4's rate of reaching its chosen goals is logged.
- *Deviation: no increase in action randomness after reaching the goal* ("Go Exploration" in MEGA). It changes actions, not goal choice, and would add route variation directly. MEGA gave it to every method it compared, so omitting it for all conditions keeps goal choice as the only difference.
  - *What C4 therefore tests.* C4 is MEGA's goal-selection rule, MEGA's contribution, not a reproduction of MEGA. Because MEGA gave Go Exploration to every method, its comparison isolated goal selection, but every method in that comparison also had post-goal exploration. Whether the rule's advantage depends on it is not established. In MEGA, reaching a frontier goal is followed by random steps beyond the frontier; here the agent stays near the goal for the rest of the episode. A weak C4 result (low coverage, or C4 failing the C4 sanity check against C2) could therefore mean that the rule needs a companion mechanism for pushing past the frontier, not that frontier selection fails. The paper states this.
  - *Expected mechanism without Go Exploration (a hypothesis, not a finding).* Relabelling cannot add coverage, because it only reuses states already visited. It adds competence: each attempt at a frontier goal becomes training data for wherever the agent actually ended up, which near the frontier is frontier states. Frontier selection aims practice where competence is lowest, relabelling raises competence there, and an agent that reaches the frontier more reliably and sooner has more of the episode left for incidental visits to new states, while the learner generalises somewhat past its training data. Coverage should therefore grow more slowly than in MEGA. The diagnostic in Section 8 (Exposure row) tests this ordering.
- C4's goals range over the whole maze, including R_h. Its environment is built with R_h added to the allowed goals; every other condition's environment refuses held-out goals.
- *Relation to quality-diversity.* Selecting the least-visited achieved goal is novelty search over end positions, the family that groups goal exploration processes with novelty search and quality-diversity (Colas, Sigaud & Oudeyer, 2018). C4 is the condition in the first experiment closest to quality-diversity.

| ID | Condition | Collection | Stage |
| --- | --- | --- | --- |
| C0 | Demonstrations only | None; matched gradient steps | MVE |
| C1 | Expert top-up | More scripted-expert trajectories within budget | MVE if compute allows |
| C2 | Random actions | Uniform random actions | MVE |
| C3 | Task-directed self-practice | Pursue a task pair drawn uniformly from the seven, with a fresh noisy start and goal | MVE |
| C4 | Frontier novelty | Pursue past achieved goals in the lowest-density region (above) | MVE |
| C3-plain, C4-plain | As C3 and C4 with γ = 1 (plain GCSL); same hyperparameters as the weighted learner; no re-learning control | As C3 and C4 | MVE, M1 only, secondary (H6) |
| C5 | Mixture | C3 with probability *p*, otherwise C4 | Stage 3 |
| C6 | Learning progress | ALP-GMM-style goal selection | Stage 3 |
| C7 | Explicit diversity (form chosen after the first experiment) | **C7a, diversity in the objective:** an adaptation of BMD. BMD adds a mutual-information bonus to a reward and fine-tunes with PPO; GCSL has no reward, so this needs a latent-conditioned learner (for example, relabelled examples weighted by exp(λ·[log q(z∣s) − log p(z)])), and rules for setting the latent during repertoire and recovery evaluation. It may fit more naturally on the Stage 3 value-based learner, which has a reward. **C7b, diversity in goal selection:** an archive of how each task goal has been reached (a coarse via-cell descriptor), with two-goal episodes: a waypoint from an under-filled niche, then the task goal, after Go-Explore (Ecoffet et al., 2021). The learner is unchanged, but "one goal per episode" becomes "one goal sequence per episode", a stated deviation. In M1 the via-cell descriptor nearly coincides with route class, so C7b is a targeted upper reference, not evidence that quality-diversity preserves repertoire. | Stage 3 |

**Held-out positions in relabelled goals.** Any condition's rollouts may wander into R_h, and relabelling then turns those positions into training goals. This is allowed for every condition: filtering relabelled goals would give conditions different learners. The leakage rules are therefore: R_h never appears in D0 and is never a *commanded* goal except under frontier novelty. Consequently S_out is a sanity check for every collecting condition, not a held-out test; only demonstrations only (C0) is guaranteed never to train on R_h positions. The share of each condition's relabelled goals that fall in R_h is logged.

**Quality-diversity in the first experiment: not added.** A population of policies cannot be scored on the single-policy repertoire measure, run through the common adaptation procedure, or used as a base policy in the companion project; with one goal per episode an archive-based goal rule reduces to C4 or reweights task goals; and at an equal environment-step budget population-based quality-diversity is starved. **Stage 3 exploratory candidate:** quality-diversity as an experience source. A MAP-Elites collector over open-loop controllers (piecewise-constant forces; descriptor: end cell; quality: path length) gathers the data, and the unchanged learner trains on all of it, as with C2. No waypoint controllers built on the scripted expert, which would leak expert knowledge.

## 7. Controls

- **Paired seeds.** Within a seed, every condition shares D0, initial weights and evaluation pairs, and therefore the same pretrained policy. C3-plain and C4-plain share them too.
- **Re-learning control.** A fresh learner is trained on each run's final buffer with matched gradient steps (following Ostrovski et al., 2021). Main learner only.
- **Budget audit.** Environment steps (and so episode counts, since episodes have a fixed length) and gradient steps are logged and checked equal across conditions. This check is automated and is also one of the auditor agent's checks.
  - *Counted separately from the practice budget N:* warm-up on D0 (shared: every condition in a seed starts from the same pretrained policy) and the k adaptation steps used for recovery (identical for every condition).
  - *Not in the budget:* computation by the goal selector, such as frontier novelty's density fitting. The study claims parity of samples (environment steps and parameter updates), not of compute. Wall-clock time per condition is reported so the difference in computational cost is visible.
- **Learner-capacity check.** The learner trained on plentiful whole-maze expert data must reach high success on M1 and M2. (The expert reaches every free cell from every other free cell on both mazes, so this data can be generated.)
- **Precondition check (pretraining).** After warm-up on D0, the pretrained policy must have mean Rep of at least 0.25 over shorter-route pairs, and Rep of at least 0.10 on every shorter-route pair. Thresholds are fixed in the pre-registration. The check is run for both the weighted and the plain learner before the pilot.
- **C4 sanity check.** C4's state coverage must exceed C2's: coverage of collected data, as defined in the Data row of Section 8.
- **Collapse check (pilot).** In the pilot, C3's Rep at budget N must be below the pretrained Rep, with the paired 95% interval across pilot seeds excluding zero. See the Stage 2 gate.
- **Hyperparameter fairness.** Tuning is done on M0 with the weighted learner, maximising the average of C3 and C4, with equal trials per condition, then frozen before any M1 or M2 run. γ, sampling temperature and the expert's controller settings are not tuned. The plain-GCSL arm uses the same hyperparameters.
- **Goal-leakage check.** No goal may be commanded outside the condition's own rule. Held-out goals must never enter D0 or be commanded except under frontier novelty (held-out positions reached by wandering may be relabelled; Section 6). Both are asserted in code, and the share of relabelled goals in R_h is logged per condition.

## 8. Measures

**Primary endpoints are computed on shorter-route pairs only.**

| Outcome | Measure | Status |
| --- | --- | --- |
| Competence | S_in: greedy success on fixed seeded pairs in R_d | **Primary** |
| Repertoire | Rep: minority-route share (below) | **Primary** |
| Repertoire | Minority-route success rate: successes on the less-used route divided by all n_roll rollouts. Unlike Rep it is low for a barely competent policy that happens to split its few successes evenly; it mixes in competence, so it does not replace Rep, but it approximates the quantity H3 says recovery depends on (repertoire times competence). | Secondary |
| Recovery | A_k: greedy success on shorter-route pairs with the shorter route blocked (P1), after k environment steps of the common adaptation procedure (below) | **Primary** |
| Recovery | A_0: zero-shot success after the P1 block, sampled at temperature 1 (H3b); A_k under P1-own, stratified by which route was blocked | Secondary |
| Recovery by selection | Best-of-m: on each blocked pair, success if any of m sampled attempts succeeds, with no learning (H3c). m fixed in the pre-registration. | Secondary |
| Repertoire quality | Per-route quality: for each route class, success rate and time to goal among sampled rollouts that take it, before the block. A two-route analogue of quality-diversity's per-niche quality; computed from the rollouts already used for Rep. | Secondary |
| Repertoire | Route-class count (routes each above 10% of successful rollouts, 0–2; also reported at 5% and 20%, because the count jumps at its threshold); route entropy; mode-weighted success (as in BMD). On M1, with exactly two route classes, route entropy is a fixed function of Rep and adds no information; it is reported for M2, where layouts can have more than two route classes, and not interpreted separately on M1. | Secondary |
| Repertoire control | Rep on tie pairs. The weighting predicts no systematic drift there, only random drift. | Secondary |
| Joint | Competence against repertoire for every condition, per seed and aggregated, so trade-offs are shown rather than argued | Secondary (reporting) |
| Exposure | Per condition: visits to the fork states of shorter-route pairs; effective sample size of the relabelling weights; share of weight mass by maze region. For C4: the rate of reaching its chosen goals and the step at which it first reaches them, as curves over training, plotted against coverage. The mechanism in Section 6 predicts that goal-reaching on frontier goals rises ahead of coverage growth; coverage growing without that rise would mean coverage comes from elsewhere. | Diagnostic |
| Generalisation | S_out: success in R_h. A sanity check only: frontier novelty practises there, and every collecting condition may train on held-out positions it wanders into (Section 6). | Secondary |
| Data | Diversity of what each condition collects (`src/data_diversity.py`), logged for collected data and for the whole buffer: **spread** (share of free space visited, on 0.25-cell bins, and visitation entropy normalised by its maximum); **information added** per collected episode (bins new to the buffer, and surprise: the episode's mean −log₂ visitation probability under the buffer before it, with add-one smoothing); **route mix** (west-to-east and east-to-west midline crossings by arm, and the minority-arm share per direction, the data-side counterpart of Rep, aggregated over the data because collected segments do not belong to a task pair). Settings fixed before any practice result. | Secondary |
| Efficiency and robustness | Path ratio; success under P2 and P3 | Secondary |

**Rep, the minority-route share.**
- For each shorter-route pair, n_roll = 100 rollouts are sampled at temperature 1. Each is classified by the route classifier: the arm in which the ball last crossed the ring's middle column. A successful rollout on a task pair must cross, so a successful rollout classed "neither" raises an error instead of being dropped.
- Among the successful ones, Rep is the fraction that take the less-used route class, so it runs from 0 to 0.5. A per-pair bootstrap confidence interval over rollouts is reported. (With a true share of 0.2 and 80 successes the standard error is about 0.045.)
- **Low-success pairs.** A pair with fewer than m_min = 20 successes has no Rep: it is undefined, not 0. Zero successes is the extreme case. Excluded pairs are counted and reported per condition, with a sensitivity analysis scoring them as 0. The rule exists because Rep alone cannot tell a policy with 45 north, 45 south and 10 failures from one with 1 north, 1 south and 98 failures: both score 0.5. Rep is therefore always reported beside S_in, success counts and the minority-route success rate. n_roll and m_min may be revised in the pilot and are frozen in the pre-registration.
- **Common pairs for comparisons.** Exclusion depends on competence, so two conditions' averages can cover different pairs. Every paired comparison of Rep (for example frontier novelty against task-directed practice) uses, within each seed, only the pairs on which *both* conditions have at least m_min successes. The number of pairs dropped is reported.
- Rep is averaged over shorter-route pairs, and also reported per pair (Section 9).
- A continuous share is used rather than a count of route classes above a threshold. A count treats an 89/11 split like 50/50, and a small drift flips it from 2 to 1. A threshold does not stop a few lucky rollouts counting either: with two successes, one on each route passes any threshold. The minimum-success rule handles that case.

Repertoire is measured on the policy and coverage on the data. A disagreement between the two is a result in itself. Under plain GCSL some seeds may drift onto the longer route. Rep is symmetric in which route is the minority, and P1-own results are stratified by the route blocked. Route class is a hand-designed two-niche descriptor, suited to M1; M2's multi-route layout is classified by the same planner cell sets, and the paper says so.

**Scope of recovery.** Three kinds of recovery are distinguished. Sampled zero-shot attempts (A_0 and best-of-m) and adaptation across episodes (A_k) are measured. Reaction to an obstruction that *appears during* an episode is not: the block is present from the first step, and the policy has no memory, so any reaction within an episode comes only from its current state (for example, being pressed against the wall). A wall appearing mid-episode is a test-time adaptation question and belongs to the companion project, switch-or-recalibrate.

**Common adaptation procedure (used for A_k).**
- Starting from the final policy, every condition runs C3-style practice on the blocked pairs for k environment steps, with the same learner and gradient-step ratio as training.
- Practice begins with an empty buffer, so no condition reuses its own old data.
- k is chosen in the pilot by a fixed rule: from the grid {5%, 10%, 20%} of N, the smallest value at which C0's mean A_k lies between 0.2 and 0.8, which avoids floor and ceiling effects.

## 9. Analysis

- **Pre-registration on OSF before the main run.** It fixes hypotheses, endpoints, smallest effects of interest, equivalence margins, m_min, m for best-of-m, the precondition thresholds, the predicted pretrained shorter-route share (overall and per pair), seed count and analysis code. The repository holds a copy for convenience.
- **Seed count from a pilot** (6 seeds per condition, because fewer gives an unreliable variance estimate), using a power analysis on paired-difference variance. Capped at 20 seeds per condition per maze. The plain-GCSL arm uses the same seed count.
- **Total compute** is estimated from Stage 1 throughput before the seed cap is frozen. It counts every main run, its re-learning run, and the adaptation runs for P1 and P1-own.
- **Primary analysis (H1–H3).**
  - Paired C4 − C3 differences, with 95% bootstrap confidence intervals and paired permutation tests. For Rep, on the common pairs only (Section 8).
  - Holm correction across the three primary endpoints.
  - For endpoints not significant after correction, two one-sided tests (TOST) for equivalence against the pre-registered margins, with Holm correction across the same three endpoints.
- **Secondary analyses** (registered, reported in full, not part of the Holm family):
  - H3b: paired C4 − C3 on A_0 (sampled).
  - H3c: paired C4 − C3 on best-of-m.
  - Per-route quality: paired C4 − C3 on minority-route success and time to goal, among pairs where both conditions keep the minority route.
  - H5: C4 − C0 on S_in (superiority) and on Rep (equivalence).
  - H6: the change in Rep from each learner's own pretrained starting point, compared between C3 and C3-plain; C4-plain − C3-plain on the three primary endpoints.
  - H4: the re-learning control. Interpretation is limited in both directions. If the advantage reproduces, the final dataset is sufficient for it; that does not show the active loop was unnecessary to collect that data. If it fails to reproduce, the advantage depends on the order and timing in which data arrived during training (the effect Ostrovski et al. describe); that does not show the data is irrelevant.
  - Minority-route success rate: paired C4 − C3, on all shorter-route pairs (it is defined even with zero successes).
- **Exploratory: pair-level analysis with the route gap as a covariate.** The six shorter-route pairs differ in how strongly the weighting favours the shorter route (measured gaps of about 8, 16 and 25 steps; Section 5). For each pair and seed, the loss of Rep from the pretrained policy and the paired C4 − C3 difference in Rep are related to the pair's gap, as a dose-response question: does task-directed practice collapse faster, and does frontier novelty's advantage grow, where the pressure is stronger? Reported with a mixed model (seed as a random effect) or, if that is unstable, per gap level. Labelled exploratory because the pairs share one network, so a preference learned on one pair can carry over to its neighbours, and six pairs at three gap levels are too few to estimate a trend with confidence.
- **Exploratory: does repertoire track the diversity of the data?** Within the main learner, per seed and condition, final Rep and A_k are related to the data scores (Section 8). **One data score is named in advance as the main one: the minority-arm share of west-to-east crossings in the collected data**, because it measures the property the hypotheses are about (both routes, in the direction of the task pairs). Spread and information added are reported descriptively beside it, without tests, so that relating several data scores to two outcomes does not turn up a correlation by chance. The question is which property of the data, if any, goes with a preserved repertoire, separating "this goal rule collects more diverse data" from "more diverse data gives a more robust policy". Correlational and confounded by condition, so it is reported within condition (across seeds) as well as across conditions; the re-learning control (H4) remains the test of whether the advantage lives in the data at all.
- **Aggregates** reported with rliable: interquartile means, performance profiles and probability of improvement.
- **Every run reported,** including failed ones, with wall-clock time per condition.

## 10. Plan and gates

| Stage | Work | Gate |
| --- | --- | --- |
| 0 | Literature gate ✔ (addendum on quality-diversity and repertoire-based recovery, 27 Oct); pre-registration draft | Comparison not already published ✔ |
| 1 | Harness: repo skeleton, determinism test ✔, M1 with planner check ✔, fixed-length episodes ✔, scripted expert (both routes per pair) ✔, P1 block in the simulator ✔, evaluation suite, route classifier (hand-checked on at least 50 trajectories), throughput measurement ✔ (cloud; Mac to come), M0 layout. Measure Δ per pair and set γ by the rule ✔ (Δ = 16, γ = 0.962); estimate total compute. | Expert reaches ~100% on M1 on both routes ✔; a full evaluation runs end to end; γ fixed ✔ |
| 2 | Learner (weighted and plain) with capacity check; C0–C4 and the plain-GCSL arm; tuning on M0; episode length set by the feasibility rule; precondition check; pilot (including γ sensitivity, descriptive only); k set by its rule; pre-registration frozen; main runs on M1; re-learning control; report v1 | **Before the pilot:** the precondition check passes for both learners. **Before the main run:** the collapse check passes. If it fails, the only permitted revision is to increase N, up to the compute cap. If C3 still does not collapse, the main run proceeds as registered and H2 and H3 are reported as uninformative about preservation ("no collapse under this learner at this budget"). γ, temperature, the learner, the expert's controller settings and the episode-length rule are not changed after seeing pilot results. **After:** every primary endpoint decided, or a reformulation documented |
| 3 | M2 replication; C5–C7 (C7's form chosen from C7a and C7b after the first experiment); a value-based learner (SAC with HER); policy-class and D0-mix ablations; optionally a quality-diversity experience source (exploratory) | Conclusions survive a second learner family and a second maze, or the report says they don't |
| 4 | Port to a goal-conditioned manipulation task with two solutions | Scoped later |

**Scope note.** The plain-GCSL arm adds two conditions to the main runs on M1, about 40% more main runs. Each main run also carries a re-learning run (main learner only) and two adaptation runs, so total compute is several times the training cost alone. It is estimated in Stage 1 before the seed cap is frozen. γ sensitivity is pilot-only. The recovery-by-selection and per-route quality measures, the C4 goal-reaching curves, the data-diversity scores and the pair-level analysis are evaluation, logging or analysis only and add no training runs.

**Build process.** One agent implements and a separate agent audits each stage for budget mismatches, goal leakage and extra gradient updates. Audit reports are committed to `notes/audits/`.
