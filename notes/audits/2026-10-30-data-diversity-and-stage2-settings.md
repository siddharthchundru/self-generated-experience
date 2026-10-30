# Audit: PR #6 (`stage2-data-diversity`) and proposed Stage 2 settings, 30 Oct 2026

The test suite passes (86 tests, 66 s). Nothing in the repository was modified. The probe scripts are `probe1.py` to `probe4.py` in this scratchpad folder.

## Part 1: the PR

**What checked out as correct**
- **Binning.** `_bins` (data_diversity.py:79-80) is `floor(k*(n_rows/2 - y))` and `floor(k*(x + n_cols/2))`. Integer division by k gives exactly `xy_to_cell` (pointmaze.py:177), including its rule for points that fall on a cell edge.
- **Midline and arm geometry** (lines 119-121) match `cell_to_xy` and `routes._geometry`.
- **Crossing direction.** `xy[j+1,0] > xy[j,0]` (line 130) is right even when `xy[j]` lies exactly on the midline.
- **Surprise of the first episode.** With an empty buffer, p = 1/n_free, so the test's log2(free bins) is exact. In real runs D0 always comes first, so this case never actually occurs.
- **M2.** With `ring=None` the crossings are None and the rest of the summary works.
- **C0** (no collected data). Entropy is None and the minority-arm share is None, not 0.
- **Wall-bin check.** It never fired in any test:
  - 300 random-action episodes of 100 steps each on M1, M0 and M2 (coverage reached 1.0 on all three);
  - every ordered expert pair: 1,190 on M1, 1,260 on M0, 650 on M2;
  - 24 episodes of 200 steps each pushing at full force into walls and corners.
  - The ball's centre came at most 0.053 from a wall edge, against a radius of 0.1 (soft contact). It never entered a wall bin. The check raises rather than logs, so a breach would crash the run instead of mislabelling data. That is acceptable.

**B1. BLOCKING: the main data score cannot detect collapse** (spec §9, new bullet; data_diversity.py:158-163).
- The score is the minority-arm share of west-to-east crossings. It pools crossings from rows where north is shorter and rows where south is shorter.
- Test: expert episodes on the seven task pairs, 10 rounds each. The share was **0.5 in all three cases**:
  - always the shorter route (complete collapse);
  - each route half the time;
  - always the longer route.
- C3, the condition H2 predicts will collapse, therefore scores as having perfectly diverse data. The spec's reason for choosing this score ("both routes, in the direction of the task pairs") does not hold.
- Fix: label each crossing with the side of the ring it came from: north-shorter rows (1-3), tie row, or south-shorter rows (5-7). Use the row the ball was in at its last visit to the west column, or a full traversal from the west column to the east column. Then report the share of crossings that take the longer arm. Freeze this before any practice result.

**S1. SHOULD-FIX: crossing counts are inflated by the ball wobbling at the midline** (lines 123-132).
- Test: 20 episodes in which the ball parks at a goal in (1,4), the cell the midline runs through.
- West-to-east crossings: 44 with no added noise, 73 with action noise of standard deviation 0.3, and 94 with standard deviation 0.6. Every one was in the north arm.
- The pretrained frontier-novelty condition (C4) can choose goals anywhere in the arms. Its counts would then measure wobbling, not traversals. This also contradicts the docstring's statement that "every crossing becomes relabelled examples that traverse that arm".
- Fix: count traversals (from the west ring column to the east ring column) or add hysteresis. This combines naturally with the B1 fix.

**S2. SHOULD-FIX: the spec's justification for route entropy on M2 has no implementation behind it.**
- The new claim, that on M1 route entropy is a fixed function of Rep, is correct. `metrics.py:59` computes the binary entropy over the north and south success counts, and Rep is min/total, so entropy = h(Rep) wherever Rep is defined. (Entropy is also computed when there are fewer than 20 successes and Rep is None.)
- The justification, "reported for M2, where layouts can have more than two route classes", rests on code that does not exist. `metrics.ROUTES` has only north and south. Stage 1 docs line 31 says M2 gets no two-route rule. M2 is not a ring maze, and spec line 163 assumes "planner cell sets" for M2 that are not defined.
- Fix: either define M2's route classes or drop the M2 rationale.

**MINOR**
- **Coverage definition** (§7 C4 sanity check, now pointing to the Data row) is consistent with the Data row. It does not say at what point in training it is measured (budget N?) or which coverage the Exposure-row curve uses.
- **Entropy normalisation** uses log(n_free) even when there are fewer states than bins, so the maximum is not reachable early on. This is harmless for comparisons between conditions at equal step counts.
- **C2 coverage does not saturate.** Random actions from the task starts at T=60 reach coverage 0.76 after 100 episodes and 0.90 after 4,167 episodes (about 250,000 steps). Normalised entropy is about 0.90 throughout. Bins of a quarter cell are therefore a reasonable resolution.

## Part 2: proposed settings

**B2. BLOCKING: N must be a whole number of episodes.**
- 250,000/60 and 1,000,000/60 are not integers. Spec §6 requires whole episodes, and `compute.counts` raises for both values with T=60.
- Fix: use 249,960 and 999,960, or choose N as a multiple of 60. Apply the same to the k grid ({5, 10, 20}% of N) and to the 100,000-step tuning budget.

**S3. SHOULD-FIX: T=60 is not consistently set.**
- `MAX_EPISODE_STEPS=100` (pointmaze.py:60), `EvaluationConfig.horizon=100` (protocol.py:45) and `ComputeAssumptions.horizon=100` all differ from the proposal. If any of them is not overridden, training and evaluation horizons silently diverge.
- Fix: keep one constant and assert it everywhere. Rerun `estimate_compute.py` with T=60, the actual updates-per-step ratio and the rented machine's throughput.

**Measured feasibility of T=60** (expert, final-step success):
- Longer route under the P1 block: 0/300 failures on M1, worst arrival at step 42 (18 steps of margin); 0/200 failures on M0, worst 37.
- All pairs of free cells: 0 failures on M1 (worst 52), M0 (worst 55) and M2 (worst 33).
- From the task starts in column 1 to held-out cells: median 36 steps, worst 54 on M1 (2 of 77 trips over 50 steps); worst 58 on M0 (8 of 60 over 50 steps).

**S4. SHOULD-FIX: deep held-out goals are marginal for frontier novelty, and the start distribution is undefined.**
- The expert has 6 steps of margin on M1 and 2 on M0, and a learned policy is slower than the expert. Frontier goals deep in the held-out region will often time out, which lowers C4's goal-reaching rate and its coverage.
- The spec says the start distribution is shared but never defines it.
- Fix: state now, before the pilot, that T is not required to cover frontier goals in the held-out region. Report the timeout rate on frontier goals separately. Define the start distribution.
- The margin of 18 steps on the P1 block should be written down as the output of the feasibility rule. This also resolves the spec's wording "margin set in the pilot" in favour of its rule that the episode-length rule is not changed after pilot results.

**S5. SHOULD-FIX: tuning is underspecified.**
- The grid is {3e-4, 1e-3} × {0.5, 1, 2}, which is 6 cells. "6 trials per condition" means equal trials only if every cell is run for both C3 and C4 with the same seeds. (The compute estimate assumes 6 cells × 3 seeds.)
- The tuning objective "maximising the average of C3 and C4" names no metric. If it includes Rep, tuning interacts with H2. Fix it as S_in, or state the metric explicitly.
- An updates-per-step ratio of 2 doubles gradient updates, which are about 80% of compute (prior audit N5). Use the worst case of the grid when budgeting against the cap.
- M0 has 5 task pairs, so 20 demonstrations per pair gives 100 demonstrations on M0, not 140.
- Warm-up stays at 10,000 updates and is not tuned. The spec gives no permitted revision if the precondition check fails. State that now.

**S6. SHOULD-FIX: expert top-up (C1) collection is unspecified.**
- The spec says only "more scripted-expert trajectories within budget". It does not say which pairs or which route split are used.
- A 50/50 route split makes C1 a reference that keeps both routes. A shortest-route-only split makes it a collapse reference.
- Fix before the pilot. Its parity with the other conditions is fine: the compute code already counts C1 in practice and in re-learning.

**S7. SHOULD-FIX: the open items from the previous audit now matter.**
- **S1 there (adaptation budget taken on trust).** `suite._adapt` (suite.py:87-96) still checks only the adapter's reported environment steps. With ratios of 0.5 or 2, it must also assert gradient steps = ratio × k.
- **S2 and S3 there (shared buffer state, privileged task inputs).** These must close before the learner is attached.
- **S4 there (results from a dirty working tree).** Remote runs must refuse a dirty tree and record the commit hash.

**MINOR**
- **Rented CPU machine.** Pin one thread per process (OMP_NUM_THREADS, MKL_NUM_THREADS, `torch.set_num_threads(1)`). 32 vCPUs are about 16 physical cores, so the throughput measured on the Mac does not transfer. Report wall-clock time per condition from that same machine. Frontier novelty's density estimation costs it more wall-clock time; per the spec this is reported, not budgeted.
- **Capacity check thresholds** (≥0.9 greedy success on the task pairs, ≥0.8 on random free-cell pairs) are reasonable. Evaluate them at T=60. On M1 the expert's random-pair trips reach 53 steps, so the 0.8 threshold implicitly requires near-expert speed. State the fallback before the pilot if the check fails.
- **Cap and pilot** (N ≤ 1M, pilot at 250k) are consistent with the rule that the only permitted revision is to increase N, if the cap is frozen before the pilot. Warm-up is counted separately, as the spec says.
- **140 demonstrations** (10 per route per pair, tie pair included) match spec §6.

**Verdict:** two items are blocking. B1: the pre-named main data score is blind to collapse. B2: N = 250,000 and 1,000,000 are not whole numbers of 60-step episodes. Both are small fixes.

---

## Response (30 Oct, same branch)

- **B1 fixed.** The route mix now counts *trips* across the ring (from where the data enters one end column to where it leaves the other), each labelled shorter or longer for its own start and end cells. The main data score is the longer-arm share of west-to-east trips. On expert data over the seven task pairs it scores 0 when always on the shorter route, 0.5 when half and half, 1 when always on the longer route (`test_score_detects_collapse_on_expert_data`). Trip end rows are recorded so the score can be read against where the data ends.
- **S1 (wobble) fixed by the same change:** a ball wobbling across the midline at a goal makes no trip (`test_wobble_at_a_midline_goal_is_not_a_trip`).
- **S2 fixed in the spec:** route entropy has no separate role until route classes are defined for M2.
- **B2:** settled in the Stage 2 settings, not the repository: N = 249,960 for the pilot and 999,960 for the cap (whole episodes of T = 60); the tuning budget and the k grid are rounded down to whole episodes.
- **S3–S7:** to be handled in early Stage 2 (see the Stage 2 handoff). The probe scripts referred to above were not committed.
