# Audit: Stage 1 expert, fixed-length episodes and γ (2026-10-28)

**How.** I read `git show` for 5887d12, 41caa92 and 38e9e1e, and the installed library source: gymnasium-robotics 1.4.2 `envs/maze/maze_v4.py` (`compute_terminated`, `update_goal`, `compute_truncated`, `compute_reward`), `envs/maze/point_maze.py` (`step`, `reset`) and the gymnasium 1.4.0 `TimeLimit` wrapper. I ran the full test suite on an untouched copy (OK, 61 s), then 12 single-line mutations, each in its own copy. I also ran independent scripts. They (a) recomputed Δ, γ, the half-life and the sensitivity values. (b) Simulated 700 west-east same-row expert episodes, 490 cross-row west-east episodes and 2,400 extra-seed episodes on rows 1 and 7, classifying each by the route it actually drove, checking whether the ball ever left the goal radius after first reaching it, and checking whether it ever entered R_h (the held-out region). (c) Rebuilt M1 with a real extra wall cell (the P1 block) at six positions and simulated the longer route. (d) Re-measured Δ under other controller settings and other definitions of Δ. (e) Measured the worst-case steps to the goal over all pairs on M1 and M2. No file in the repository was modified.

**Verdict.** The code is correct on every point I could test. With `continuing_task=True, reset_target=False` the library never terminates and never moves the goal. `TimeLimit` sets truncation at exactly step T. `info["success"]` on the last step is final-step success. The expert drives the commanded route in all 3,590 episodes I simulated: none were classified "both" or "neither", and none entered R_h. Δ = 16 reproduces exactly, and γ = 0.96205 (half-life 17.9 steps), 0.98003 (share 0.58) and 0.94268 (share 0.72) are correct. Nothing in the measurement depends on T = 100 or on the test's T = 70.

The weaknesses are in evidence and documentation, not in the code:
- P1 feasibility is tested only on the planner. The simulator can in fact place the wall, and I verified feasibility with it.
- Δ depends noticeably on the hand-set expert gains (15.5 to 26 steps), and the result file does not record those gains.
- The public `docs/spec.md` and `README.md` contradict the code.
- Two plausible regressions survive the tests.

| # | Severity | Finding | Suggested disposition |
|---|---|---|---|
| 1 | medium | **P1 feasibility is planner-only.** `tests/test_expert.py:53-66` checks that the grid planner finds a longer route that avoids the cell. It never simulates the ball with a wall there. The simulator *can* place the wall: passing a modified grid as `maze_map` works, and only the `MAZES` registry in `src/envs/pointmaze.py:48` limits it. I built M1 with a wall at (1,2), (1,4), (1,6), (7,2), (7,4) or (7,6) and ran 20 seeds per affected row. The longer route succeeded 360/360, was classified correctly every time, and first reached the goal within 41 steps at most. Clearance is tight at the arm ends: on rows 1 and 7 the longer-route trajectory passes within 0.09 of cell (1,6)/(7,6) and within 0.26 of (1,2)/(7,2), with a ball radius of 0.1. **Verified:** planning and avoidance on the grid (repository tests), and simulated success with a real wall (this audit only). **Not verified:** the actual P1 cell (still "fixed with the evaluation suite"), any margin under a final T, and repertoire or recovery under P1. | Add a simulated P1 test: make a blocked-grid env (e.g. let `make_env` accept a grid) and assert the expert succeeds on the longer route with the block. Run it again once the P1 cell is fixed. |
| 2 | medium | **Public docs contradict the code.** `docs/spec.md:47` still says "An episode ends when the ball is within 0.45 of the goal … or after a fixed number of steps". The public spec is v0.2 (`docs/spec.md:5`) and has no γ, weighted GCSL or Δ. `README.md:7` lists "Next: the scripted expert…", and `README.md:77` leaves that item unticked. | Sync `docs/spec.md` to v0.7, or at least to the episode and γ sections, and update the README status, in their own commit. |
| 3 | medium | **Δ (and so γ) depends noticeably on hand-set expert gains, and the result does not record them.** Re-measuring with other settings (20 seeds each): `SWITCH_RADIUS` 0.25 gives Δ = 22.5, 0.8 gives 15.75 and 1.2 gives 15.5. `KD` 1.0 gives 16, and `KD` 2.5 gives 26.25 (γ ≈ 0.976). This is allowed, since the spec says γ uses "the expert", and the gains are documented as not tuned on any result (`src/expert.py:34-38`). But `results/stage1/route_gap.json` records neither the gains nor the git commit that produced it, so the fixed γ cannot be tied to a specific expert. | Write KP, KD, SWITCH_RADIUS, N_SEEDS and the commit hash into the JSON. Treat the gains as frozen with γ: any later change to the expert reopens Δ. |
| 4 | low | **Final-step success is not checked against "ever reached".** In mutation `success_ever`, `run_episode` reported `first_success is not None` instead of the last step's `info["success"]` (`src/expert.py:111`), and all tests passed. The expert holds at the goal, so the two coincide for it, but the learner's evaluation will need the distinction. | Add a test in which the ball reaches the goal and is then pushed out, and assert that final success is False. |
| 5 | low | **The exact-goal last waypoint is untested.** Mutation `goal_cell_centre` (final waypoint at the cell centre instead of the noisy goal, `src/expert.py:68-69`) passed. It works because the noise is at most 0.35 from the centre, which is under 0.45. The docstring claim at `src/expert.py:16-18` is therefore true but not protected by any test. | Assert in a test that the final distance to the goal is small (e.g. < 0.1). |
| 6 | low | **Off-by-one in a reported number.** `first_success_step` is 0-indexed (`src/expert.py:103`), so the JSON's `longest_expert_longer_route_steps: 40` means 41 steps. The commit message says "The longest expert route takes 40 steps, which bounds the episode-length rule from below". Δ is unaffected, because it is a difference. | Report steps as `t + 1`, or rename the field `..._index`. |
| 7 | low | **No test asserts that demonstrations stay in R_d.** Held-out goals are kept out of training only by `RequireCells`, on the goal cell. If a demo trajectory passed through R_h, relabelled goals (state positions) would leak into it. I observed none in 1,190 west-east episodes, but nothing enforces it. The all-pairs test widens `allowed_goals` only locally (`tests/test_expert.py:74`). The defaults (`src/envs/pointmaze.py:49, 92-93`) and `scripts/measure_route_gap.py:51` use R_d, so the widening does not leak into data collection. | Add an assertion that every cell visited in an expert demo is in `M1_DEMO_REGION`, and reuse it in D0 collection. |
| 8 | low | **Spec rationale for γ is now outdated by the measurement.** The spec rejects 0.98 because it "was set for 50-step episodes", whereas this project's episodes would be "hundreds of steps". The measured routes take at most 41 steps, so T will be about 50, and the rule's γ = 0.962 is a *stronger* preference than 0.98. The rule itself is applied correctly. | Note this in the spec's γ section when T is set. Do not change γ. |
| 9 | low | **Commit 38e9e1e bundles two ideas.** It holds the Δ/γ measurement and the throughput measurement for the compute estimate (`scripts/measure_route_gap.py:80-92`). The commit message also leaves out that rows 2 and 6 have gap 16, which is the value that sets the median. | Acceptable as a single "Stage 1 measurement" commit. Split it in future. |
| 10 | info | **Checks that passed.** No early termination (`maze_v4.py:384-392`). The goal is never moved, because `update_goal` requires `reset_target` (`maze_v4.py:397-401`). Truncation comes from `TimeLimit` at `elapsed >= T`, and the library's own `compute_truncated` is always False. Sparse reward is 1 on every step inside the radius, as documented (`pointmaze.py:26-27`). Episode length is exactly T, asserted at `tests/test_expert.py:49`. Seed pairing is valid: the same seed gives an identical start and goal on both routes. Δ is robust to its definition: median of per-pair medians 16, pooled median 16, difference of medians 17. The tie row reaches the goal by step 27 on both routes. Cross-row west-east pairs (1-7 × 1-7, both routes) succeed 490/490 on the commanded route. Worst all-pairs first success is step 53 on M1 and 28 on M2, so T = 70 in the tests has margin, as `tests/test_expert.py:31` says. Every new file opens with a what-and-why docstring, except the JSON, which cannot hold comments. | None. |

### Mutations

| Mutation | Caught? |
|---|---|
| Swap route blocks (north ↔ south) | Yes (281 failures) |
| Remove damping (KD = 0) | Yes (219) |
| KD = 0.5 | Yes (21) |
| Expert ignores route (no arm block) | Yes (142) |
| SWITCH_RADIUS = 0.1 | Yes (47) |
| SWITCH_RADIUS = 0.8 | No. Behaviour stays correct; Δ shifts to 15.75 |
| SWITCH_RADIUS = 1.2 | No. Behaviour stays correct; Δ shifts to 15.5 |
| Expert stops pushing once inside the radius (no hold) | Yes (154) |
| Final waypoint at cell centre instead of exact goal | **No** (finding 5) |
| `success` = ever reached instead of last step | **No** (finding 4) |
| `continuing_task=False` | Yes (281) |
| `reset_target=True` | Yes (283) |

## Dispositions (implementing agent, 2026-10-28)

| # | Disposition |
|---|---|
| 1 | Done in 94275a4: `make_env` takes `blocked_cells`; the expert is tested with a real wall at the start, middle and end of the shorter arm (90 episodes). Re-run once the P1 cell is fixed. |
| 2 | Open: syncing the public `docs/spec.md` and README to the private spec is left for Siddharth's decision, as its own commit. |
| 3 | Done: `route_gap.json` records KP, KD, SWITCH_RADIUS and the commit. The gains are frozen with γ; changing the expert reopens Δ. |
| 4, 5, 7 | Done in d8bae8c; both surviving mutations are now caught. |
| 6 | Done in d8bae8c: steps are counted as actions taken (longest route 41). |
| 8 | Carried to the private spec's γ section: episodes will be about 50 steps, so the rule's γ = 0.962 is a stronger preference than 0.98. γ is not changed. |
| 9 | Accepted. |
