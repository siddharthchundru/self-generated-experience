# Audit, 26 Nov 2026: early checkpoints, tuning, capacity check and precondition gate

What this file is: an independent audit of the five commits on `stage2-checks` on top of `689de26`:
- early checkpoints in a run, and tuning cells as separate batches in the pretrained check (`3119321`);
- the launcher's tuning grid and checkpoint evaluations (`67f05e7`);
- the tuning selector and the M0 tuning plan (`803e5be`);
- the learner-capacity check on M1 and M2 (`666dddf`);
- the precondition gate and its plan, and the tests for all of these (`d958a34`).

It lists defects with evidence and fixes, so the author can decide what to change before the capacity, tuning and precondition launches. Each claim is labelled **established** (shown by code, a test or script output) or **reasonable concern** (plausible, not shown). The auditor was a separate agent that had not written the code and did not trust its docstrings.

## 1. Scope and method

- Read `git diff 689de26..HEAD` in full (12 files, +1,152 lines), and the code it relies on: `src/training.py` (settings, warm-up, the practice loop), `src/eval/suite.py` and `src/eval/protocol.py` (how S_in, Rep and the primary pairs are computed), `src/gcsl.py` (`train`), `src/runs.py`, `scripts/evaluate_run.py`, and sections 7 and 8 of `docs/spec.md`.
- Decisions 2, 6, 7 and 15 and the planning values are Project documents, not in the repository. Their readings were taken from the brief.
- The two trees were exported with `git archive` into `/tmp/audit26/base` (`689de26`) and `/tmp/audit26/head` (HEAD, `d958a34`). No git worktree was added, the auditor modified no tracked file and wrote nothing but this report, and nothing was written into `results/`.
- Everything below is about commit `d958a34`. While this report was being finished (about 04:50 on 26 Nov), uncommitted edits to nine tracked files appeared in the working tree, some citing this report's finding IDs. They were made by another process, not by the auditor, and were not audited.
- Adversarial scripts, in `/tmp/audit26/scripts/` (outside the repository):
  - `digests.py`, `compare_digests.sh`: the training runner at `689de26` against HEAD, with and without checkpoints;
  - a tiny end-to-end tuning launch (`plans/tiny_tuning.json`: M0, 2 x 2 grid, both tuned conditions, 1 seed, checkpoints after 1 and 2 episodes, full evaluation protocol), then `select_tuning.py` and `check_pretrained` on its output, and forged copies of it;
  - a tiny end-to-end precondition launch (`plans/tiny_pre.json`: M1, both learners, 2 seeds, budget 0), then `precondition_check.py` on it;
  - `selector_cases.py`, `pre_maze.py`: false passes of the selector and the gate, using evaluation files shaped like `evaluate_run.py`'s (the tests' own `fake_evaluation`);
  - `run_capacity.py --all --quick` twice into one folder;
  - launcher probes (`build_jobs` on the two committed plans and on edge-case plans);
  - `mutate.py`: mutation testing.
- Only tiny runs: warm-up 30 to 200 updates, 2 demonstrations per pair, batch 32, budgets of 0 to 300 steps; capacity with `--quick` (300 updates). No competence or repertoire number was looked at for its value.

Words used below:
- *N*: the practice budget in environment steps; *T*: episode length, 60 steps.
- *r*: gradient updates per environment step (0.5, 1 or 2).
- *Learning rate*: 0.0003 or 0.001 in the tuning grid. A *cell* is one (learning rate, r) combination.
- *S_in*: greedy success on the shorter-route task pairs (20 greedy rollouts per pair), the competence endpoint.
- *Rep*: the repertoire measure, the share of a pair's successful sampled rollouts that take the less-used route (0 to 0.5); undefined below 20 successes in 100.
- *Rep_bar(p)*: Rep on pair p averaged over seeds (the precondition check's default definition).
- *Tie row*: the task pair whose two routes have equal length (row 4 on M1, row 3 on M0). Rows above it are north-shorter, rows below it south-shorter.
- *Fingerprint*: the SHA-256 digest of a network's weights (`parameter_digest`).
- *Pretrained policy*: the policy after warm-up on the demonstrations (D0), which every condition in a seed must share.
- *Early-drop check*: the spec v0.12 fallback; if greedy competence falls clearly below the pretrained value in the first practice episodes, a fixed linear learning-rate warm-up is added and tuning is redone on M0.

## 2. Test results

**Full suite at HEAD** (`python -m pytest -q`): 220 passed, 424 subtests passed in 351 s (run alongside other work on a 2-CPU machine). `tests/test_stage2_checks.py` alone: 18 passed in 56 s.

**Training runs unchanged (`compare_digests.sh`), established.**
- Nine settings, each run three ways: with `689de26`, with HEAD, and with HEAD keeping checkpoints:
  - task-directed practice on M1 (checkpoints after 1, 3 and 5 episodes);
  - frontier novelty at r = 0.5 with the interleaved schedule (1, 2);
  - frontier novelty with the plain learner on M0 (2);
  - task-directed practice as a tuning run on M0, learning rate 0.0003, r = 2 (1, 5);
  - frontier novelty as a tuning run on M0, learning rate 0.001, r = 0.5 (1, 4);
  - random actions on M0, seed 3 (1, 3);
  - expert top-up with the validation warm-up rule (1);
  - demonstrations only at r = 2 with the learning-rate warm-up of 50 steps (1, 2, 3);
  - a precondition run with N = 0 (no checkpoints possible).
- Compared: pretrained and final fingerprints, the bytes of the pretrained and final weights, the episode log, the summary (without its timing and the new `checkpoint_digests` key) and the final buffer.
- All identical in every setting, across trees and with or without checkpoints. The main learner's seed-0 pretrained fingerprint is `f65147ba4669…`, the value recorded in the 4 and 24 Nov audits.
- Independent check that a checkpoint is what it says: in each of the eight settings with checkpoints, checkpoint n of the long run equals the final fingerprint of a separate run with N = n x T. So `checkpoint_ep<N>` is the policy after exactly N practice episodes and their updates, under both schedules.

**Committed plans (`build_jobs`), established.**
- `plans/tuning_m0.json` gives 162 jobs: 36 training runs, 108 evaluations, 18 pretrained evaluations. The 36 runs are exactly 6 cells x 2 conditions (task-directed practice, frontier novelty) x 3 seeds (0, 1, 2): 18 per condition, all weighted learner, M0, purpose `tuning`, N = 99,960, batch 256, 20 demonstrations per pair, warm-up 10,000. Every run keeps checkpoints after 5 and 25 episodes; every folder label matches its settings.
- Every evaluation uses the full protocol (100 sampled and 20 greedy rollouts per pair, not quick), with adaptation skipped. Checkpoint evaluations carry `--skip-adaptation` themselves.
- Each cell's pretrained evaluation reads one of that cell's own runs, is checked against that cell's group, and depends on both of its training runs.
- `plans/precondition_m1.json` gives 12 warm-up-only runs (N = 0) on M1, task-directed practice for both learners, seeds 0 to 5, purpose `precondition`, and 12 pretrained evaluations with 100 sampled rollouts and no adaptation. Its learning rate is a placeholder (SF4).

**End-to-end launches, established.**
- The tiny tuning launch finished all 36 jobs. Every evaluation's recorded fingerprint equals its run's, and each cell's pretrained evaluation fingerprint equals the pretrained fingerprint of that cell's runs. The two cells with learning rate 0.001 share one pretrained policy, as do the two with 0.0003. `select_tuning.py` read it and chose a cell.
- But the launcher reported "pretrained check: no runs" and wrote `"pretrained_check": null` in its manifest (SF1).
- The tiny precondition launch finished; `precondition_check.py` read the real files: the tie row is marked non-primary, undefined Rep is `None` and counted as 0, and the gate failed as it should for a policy that had hardly trained.

**Selector, gate and capacity false passes, established.** Accepted without complaint:
- by the selector: a grid with a whole cell missing; a cell whose label disagrees with its runs' settings; a run inside a cell with another cell's settings; a pretrained evaluation copied from another cell; cells evaluated on different seeds; one seed per cell; an `evaluation_final.json` that evaluated the pretrained weights; an extra cell outside the agreed grid, which was chosen (SF2, SF3);
- by the precondition gate: three of the six seeds; the plain learner's folder holding weighted-learner evaluations; the two learners on disjoint seeds; the two learners warmed up with different learning rates; the plain learner's files from M0; a file listing the tie row as a primary pair (SF4, MI6);
- by `run_capacity.py --all`: stale quick results from an earlier call, reported as a non-quick check (SF6).

**Mutations.** Each mutation was applied alone to a fresh copy of HEAD. Mutations of `src/training.py`, `src/runs.py` and `scripts/launch.py` were run against `tests/test_stage2_checks.py`, `tests/test_run_scripts.py` and `tests/test_training.py`; the others against `tests/test_stage2_checks.py` alone, except M1 to M3, which were run against all three. KILLED means a test failed, as it should.

| ID | Mutation | File | Result |
| --- | --- | --- | --- |
| M1 | Random-pair threshold lowered to 0.7 | `src/capacity.py` | KILLED |
| M2 | Pass rule ignores the task-pair threshold | `src/capacity.py` | **SURVIVED** |
| M3 | M1's task pairs never evaluated | `src/capacity.py` | **SURVIVED** |
| M4 | Strict comparison (> 0.8 instead of ≥ 0.8) on random pairs | `src/capacity.py` | **SURVIVED** |
| M5 | Doubling applied twice | `src/capacity.py` | KILLED |
| M6 | Evaluation pairs drawn per learner | `src/capacity.py` | KILLED |
| M7 | Evaluation pairs drawn from the run's seed | `src/capacity.py` | SURVIVED (equivalent at the default seed 0) |
| M8 | Expert failures silently dropped from the data | `src/capacity.py` | **SURVIVED** |
| M9 | Selector adds Rep to the objective | `scripts/select_tuning.py` | **SURVIVED** |
| M10 | Selector uses the last checkpoint (25 episodes) for the early drop | `scripts/select_tuning.py` | KILLED |
| M11 | Tie rule reversed (larger ratio wins) | `scripts/select_tuning.py` | KILLED |
| M12 | Tie margin ignored | `scripts/select_tuning.py` | KILLED |
| M13 | Early drop compares every seed with seed 0's pretrained policy | `scripts/select_tuning.py` | **SURVIVED** |
| M14 | Selector accepts quick evaluations | `scripts/select_tuning.py` | KILLED |
| M15 | Undefined Rep skipped instead of counted as 0 | `scripts/precondition_check.py` | KILLED |
| M16 | Every pair taken, tie row included | `scripts/precondition_check.py` | **SURVIVED** |
| M17 | One pair a side is enough to proceed | `scripts/precondition_check.py` | **SURVIVED** |
| M18 | Sides split one row off the tie row (row 5 on both sides) | `scripts/precondition_check.py` | KILLED |
| M19 | Exclusion looks at the weighted learner only | `scripts/precondition_check.py` | **SURVIVED** |
| M20 | Keeping a checkpoint draws once from the practice stream | `src/training.py` | KILLED |
| M21 | Checkpoint kept one episode late | `src/training.py` | KILLED |
| M22 | Grid cells mislabelled: each label gets the next cell's settings | `scripts/launch.py` | **SURVIVED** |
| M23 | Every cell's pretrained evaluation reads the first cell's run | `scripts/launch.py` | **SURVIVED** |
| M24 | Tuning-cell grouping applied to every purpose | `src/runs.py` | KILLED |
| M25 | Tuning-cell grouping removed | `src/runs.py` | KILLED |
| M26 | Resume ignores a changed checkpoint list | `scripts/launch.py` | **SURVIVED** |
| M27 | Checkpoint evaluations run adaptation | `scripts/launch.py` | KILLED |

Of the 27 mutations, 14 are killed and 13 survive; M7 is equivalent in practice. Of the mutations the brief named:
- killed: capacity threshold changed (M1); doubling applied twice (M5); evaluation pairs drawn per learner (M6); wrong checkpoint (M10); tie rule reversed (M11); precondition averaging wrong (M15); a checkpoint that consumes a random draw (M20);
- **survived**: selector including Rep (M9); tie row included (M16); exclusion side count wrong (M17); grid cell mislabelled (M22).

The survivors are test gaps (SF7), not defects in the current code. M26 survives because the resume test in `test_grid_launch_and_checks` had first edited the runs' purpose, so the resume fails on settings whatever the checkpoint rule does.

## 3. Verdict

**No blocking defect.** Each defect below needs a resumed launch, a hand-copied file or a stale folder to bite. None changes a result in a clean, single launch. On the six focus points:

1. **Training runs: unchanged (established).** Keeping checkpoints only writes files. In nine settings, fingerprints, episode logs, summaries and buffers match `689de26` bit for bit, with and without checkpoints, and a checkpoint equals the final weights of a run stopped at that episode.
2. **Capacity check: right thresholds, data and doubling, but two ways to report wrongly.**
   - Right (established): 0.9 on M1's seven task pairs and 0.8 on random pairs, on M1 and M2. Data is one successful expert episode for every ordered pair of distinct free cells. Evaluation pairs are fixed and identical across learners. The doubling is applied once, to both learners on both mazes.
   - `--all` reports whatever results are already in its folder, including stale quick ones, under its own flags (SF6).
   - The task-pair threshold is applied to the mean of the seven pairs, so one pair at 0.3 can pass (MI3, a reading to confirm).
3. **Tuning: the plan is exactly the agreed grid. The selector computes the stated objective, applies the tie rule and never reads Rep. But it trusts its folder.**
   - The plan has 6 cells x 2 conditions x 3 seeds, N = 99,960, M0, weighted learner, 18 trials per condition (established).
   - The selector refuses incomplete cells, unequal seed counts across cells, quick evaluations and runs that are not tuning runs on M0.
   - It does not check that the cells it finds are the agreed grid, that runs match their cell, that seeds agree, or that the pretrained and checkpoint evaluations belong to the run they are compared with (SF2, SF3).
   - The early-drop comparison is the right one in a launcher-made folder: the first checkpoint against the same seed's pretrained policy in the same cell, which the auditor verified by fingerprint.
4. **Precondition: plan and arithmetic right. The failure rule adds a condition that the written rule does not have, and the inputs are not checked for consistency.**
   - The plan gives warm-up-only runs for both learners, purpose `precondition`, 100 rollouts (established).
   - Both definitions compute what they say. The sides split correctly around the tie row.
   - A failing mean stops the pilot even when exclusion would leave two pairs a side with a passing mean. This is the likely failure case (SF5, a decision).
   - Missing seeds, mismatched learners, mazes or learning rates pass (SF4).
5. **Pretrained-check change: no hidden mismatch for non-tuning purposes (established). For tuning, the code comparison no longer crosses cells, and the launcher never runs the whole-folder check that would compare cells sharing a warm-up (SF1).**
6. **Launcher: every job is wired to the right folder, dependency and settings (established).** Resume checks cover checkpoints. A grid is allowed for any purpose, and labels can collide (MI1, MI2).

The findings to fix before the tuning launch are SF1 to SF3. SF4 and SF5 should be fixed before the precondition launch, and SF6 before the capacity launch. Each is a small change.

## 4. Findings

### Blocking

None.

### Should fix

**SF1. In a tuning launch, nothing compares runs across cells. The launcher skips its whole-folder pretrained check, and the per-cell grouping drops the code comparison. Established.**
- `scripts/launch.py:406` runs `check_pretrained(out)` only if `OUT/runs` exists. With a grid, runs live in `OUT/cells/<cell>/runs/`, so the check is skipped. The tiny tuning launch printed "pretrained check: no runs" and recorded `"pretrained_check": null`.
- The checks that do run before each pretrained evaluation cover one cell, one seed and one learner (`scripts/launch.py`, `group=folders`).
- Two cells with the same learning rate must share a pretrained policy, because warm-up does not depend on r. Only the whole-folder check (`by_key`, `src/runs.py`) compares them.
  - In a copy of the tiny launch, both runs of cell `lr0.001_r1` were given the pretrained policy of a 0.0003 cell, consistently (weights and summary).
  - The launcher's per-group check passed.
  - The whole-folder check, run by hand, failed, as it should.
- `src/runs.py:159` and `:176` put the cell into the `by_seed` key, which also holds the source-hash comparison. In a copy where one cell's runs recorded a different `src/gcsl.py` hash, the whole-folder check passed with no problem and no warning. Before this change the same folder would have failed.
- Why it matters:
  - The selector compares cells, so they must have run the same code.
  - Resuming a tuning launch after a crash and a code fix is the plausible way to get one cell on old code.
- Fix:
  - In `main`, run the check whenever `find_runs(out)` is non-empty.
  - In `check_pretrained`, keep the cell in the warm-up comparison but compare source hashes over (seed, maze, learner, purpose) without the cell. Or simpler: for tuning, require one set of source hashes across the whole folder.
  - Test: the forged copy above must fail the launch.

**SF2. The tuning selector does not check that the cells it reads are the agreed grid, or that each cell's runs belong to it. Established.**
- `scripts/select_tuning.py:58` takes every folder under `OUT/cells/`. It takes the cell's learning rate and r from the first run it reads (line 77), and checks each run only for purpose and maze (line 74). Seeds are compared across cells by count only (line 105).
- Evidence (`selector_cases.py`), each accepted:
  - a whole cell absent: chose among 5 cells;
  - a folder labelled `lr0.001_r0.5` holding r = 2 runs: two rows reported as (0.001, 2), and the missing cell went unnoticed;
  - one run in a cell made with another cell's settings;
  - cells on seeds {0, 1, 2} and {3, 4, 5};
  - one seed per cell instead of three;
  - an extra cell `lr0.003_r1` from another launch, which won.
- Why it matters: decision 2 picks a cell by a margin of 0.01 among exactly six cells, with equal trials. A leftover or partial folder would change the choice silently.
- Fix:
  - Read the plan from `OUT/launch_manifest.json` (last launch).
  - Require the set of (learning rate, r) found to equal the plan's grid exactly.
  - Require every cell to hold the plan's seeds for both conditions.
  - Require every run's settings to equal the plan's settings with the cell's overrides (checking `budget_steps`, `maze`, `learner` and `lr_warmup_steps` as well).
  - Refuse otherwise. Optionally also refuse a grid other than {0.0003, 0.001} x {0.5, 1, 2} unless `--grid-override` is given.

**SF3. The selector does not tie the pretrained, checkpoint and final evaluations to the run they are compared with. Established.**
- `scripts/select_tuning.py:79-85` finds the pretrained evaluation by file name (`cell/pretrained/weighted/seed<N>.json`) and the checkpoint evaluation by glob. It never reads `run.which`, `run.policy_digest` or `run.folder`.
- Evidence:
  - a pretrained evaluation copied from a 0.0003 cell into a 0.001 cell was accepted, its settings saying 0.0003, and the early drop changed sign;
  - an `evaluation_final.json` recording `which: pretrained` was accepted as a final evaluation.
  - Mutation M13 (every seed compared with seed 0's pretrained policy) survives.
- In a launcher-made folder the wiring is right: every evaluation's fingerprint matched its run's in the tiny launch. The risk is hand-copied or resumed files, and the cost of the check is one comparison per file.
- Fix:
  - Load each run's `summary.json`.
  - Require the final evaluation's `policy_digest` to equal `final_digest` and its `which` to be `final`.
  - Require the checkpoint evaluation's to equal `checkpoint_digests[n]`.
  - Require the pretrained evaluation's to equal the run's `pretrained_digest`.
  - Test with a fake evaluation carrying a wrong fingerprint.

**SF4. The precondition gate does not check that its inputs are the planned seeds, the right learner, M1 and one warm-up; and the plan's learning rate is a placeholder. Established.**
- `scripts/precondition_check.py:83-113` reads every `seed*.json` under each learner's folder. It checks quick, `which`, 100 rollouts and purpose. It requires only the same number of files for both learners. The maze is taken from the first file (line 92).
- Evidence (`selector_cases.py`, `pre_maze.py`), each passed:
  - three of the six seeds;
  - the plain folder holding weighted-learner evaluations;
  - the two learners on disjoint seeds;
  - the learners warmed up with learning rates 0.001 and 0.0003;
  - the plain learner's files from M0, scored on M0's rows and reported as M1.
- `plans/precondition_m1.json:4` sets learning rate 0.001 with a note to change it after tuning. Warm-up uses the practice learning rate, so the gate's verdict applies only to pretrained policies made with that rate. Nothing links it to the tuning choice, or the precondition runs to the pilot's.
- Why it matters: the gate decides whether the pilot runs and which pairs are primary endpoints. It must vouch for the pilot's own pretrained policies.
- Fix:
  - For each file, require `settings.learner` to equal the folder, `settings.seed` the file name, maze M1, budget 0.
  - Require one warm-up key (apart from learner and seed) across all files.
  - Require the same seed set for both learners, equal to the plan's seeds in `launch_manifest.json` (6).
  - Add `--tuning-report` and refuse a learning rate that differs from the chosen one.
  - Record each seed's pretrained fingerprint in the report, and have the pilot launch compare its pretrained fingerprints with them. Warm-up is deterministic, so they must match.

**SF5. The exclusion rule as coded stops the pilot whenever the mean over all six pairs is below 0.25, even if exclusion leaves at least two pairs a side with a passing mean. The brief's decision 7(b) does not say this. Established (the code); which reading is intended is a decision.**
- `scripts/precondition_check.py:120-121`: `exclude_ok = means_ok and ...`, where `means_ok` uses the mean over all shorter-route pairs, the excluded ones included.
- Evidence (`selector_cases.py`, case P5):
  - Rep_bar is 0.05 on rows 1 and 7, 0.30 on rows 2 and 6, and 0.35 on rows 3 and 5;
  - the mean over the six pairs is 0.233, and over the four kept pairs 0.325;
  - two pairs remain on each side;
  - the code says "stop the pilot". Decision 7(b) as given in the brief ("exclude pairs below 0.10 … if at least two pairs remain on each side, else stop") says proceed.
- Why it matters: spec section 3 predicts that the outer pairs (rows 1 and 7) are the most likely to fail the precondition. Two outer pairs near zero pull the six-pair mean under 0.25 almost automatically, so this is the likely failure case, not an edge case.
- Fix: decide which rule is meant.
  - If exclusion is meant to cover it, recompute the mean over the kept pairs for each learner, and proceed if it is at least 0.25 with at least two pairs a side.
  - Otherwise, state the stricter reading in decision 7.
  - Either way, add a test with this example.

**SF6. `run_capacity.py --all` reports whatever `capacity.json` files are already in its folder, under its own flags. Established.**
- `scripts/run_capacity.py:74-81` waits for the four children, ignores their exit codes, and reads `OUT/<maze>_<learner>/capacity.json` if it exists. The summary's `quick_smoke_only` is the parent's flag, not the children's. `capacity_summary.json` is overwritten.
- Evidence:
  - `--all --quick` into a folder, then `--all` (not quick) into the same folder;
  - all four children refused ("claimed by another run");
  - the summary was rewritten from the quick results (600 updates), labelled `quick_smoke_only: false`.
  - Had those results passed, it would have said "capacity check passed".
- Fix:
  - Claim `OUT` (or `capacity_summary.json`) with O_EXCL before starting.
  - Treat a non-zero exit code as a failure.
  - Require each `capacity.json` to have this call's maze, learner, updates, learning rate, on-fail rule and quick flag.
  - Write `quick_smoke_only` as "any child quick".

**SF7. Test gaps, from the surviving mutations. Established.** The current code is right in each case; a regression would go unnoticed.
- M9 (selector adds Rep to the objective): the fake evaluations have no Rep, so nothing tests that tuning ignores it. Fix: give the fake files a Rep that would change the choice, and assert it does not.
- M17 (one pair a side accepted): the test's "one north pair left" case also fails the mean (0.242), so the side count is never tested alone. Fix: a case with a passing mean and one pair left on one side.
- M19 (exclusion looks at the weighted learner only) and M16 (tie row taken in): the tests give both learners the same values and never include the tie row. Fix: a case where only the plain learner has a pair below 0.10, and one with the tie row present.
- M2, M3, M4 (capacity: task-pair threshold ignored, M1's task pairs skipped, strict comparison): every capacity test runs on M2 with an untrained policy, so the pass rule is never exercised. Fix: unit-test the pass rule on given success rates (exactly 0.8 and 0.9 pass, 0.79 fails, a task-pair failure on M1 fails), and one M1 evaluation that reports `task_pairs`.
- M8 (expert failures dropped): no test makes the expert fail. Fix: patch `run_episode` to fail once and expect the error.
- M22 (grid cells mislabelled), M23 (every cell's pretrained evaluation reads the first cell's run): `test_grid_jobs` checks labels and counts only. Fix: assert that each training job's planned settings match its folder's label, and that each pretrained evaluation's `--run` lies inside its own cell.
- M13: see SF3.
- M26 (resume ignores checkpoints): the resume test in `tests/test_stage2_checks.py:147` runs on a folder whose `settings.json` and `summary.json` it had just edited (purpose `tuning`). So every training job fails on settings first, and the checkpoint rule is never what fails it. Fix: resume an unedited launch that kept checkpoints [1, 2] with a plan asking for [1], and expect failure. (Asking for [2] after [1] also fails without the rule, because the new checkpoint evaluation fails.)
- M7 (evaluation pairs drawn from the run's seed) is equivalent at the default seed 0 and needs no test.

### Minor

**MI1. Grid labels can collide, giving two jobs one folder. Established.** `scripts/launch.py:151` formats values with `:g`, while distinctness is checked on `str` (line 131). A grid `{"updates_per_env_step": [1, 1.0]}` built two training jobs for one folder. Fix: refuse duplicate labels in `cells()`.

**MI2. A grid is accepted for any purpose, and a tuning plan is accepted on M1. Established.**
- A pilot plan with a grid over r built 4 training jobs. For a grid launch the final pretrained check is skipped (SF1). The pilot's runs in a seed would then differ in r with nothing flagged.
- A tuning plan on M1, with demonstrations only and the plain arm, also built. The selector refuses it only after the runs. Spec section 4 keeps M1 untuned.
- Fix:
  - Allow `grid` only for purposes `tuning` and `smoke`.
  - In `TrainingSettings`, require M0 and the weighted learner for purpose `tuning`.
  - In `load_plan`, require the two tuned conditions for a tuning plan.

**MI3. The capacity check applies 0.9 to the mean over M1's seven task pairs, so a pair at 0.3 passes if the other six are at 1.0. Established (the code); whether that is the intended reading is a reasonable concern.** See `src/capacity.py:145-146`. "Greedy success of at least 0.9 on the seven task pairs" can mean the average or every pair. A learner that cannot do one pair is not ruled out as the cause of a poor result on that pair. Fix: decide. If per pair, also require each pair to reach at least 0.9 (or a stated lower floor). `task_pairs_by_row` is already recorded.

**MI4. The selector's score is a pooled mean over runs. It equals the stated formula (the mean of the two conditions' seed means) only when both conditions have the same number of seeds in a cell, and the count check (line 105) compares cells, not conditions. Established.** See `scripts/select_tuning.py:90` and `:98`. Fix: average the two condition means, or require equal counts per condition (SF2's seed check covers it).

**MI5. The early-drop flag looks only at the chosen cell's mean drop at the first checkpoint. The second checkpoint (25 episodes) is evaluated but never read. Established; whether that is enough is a reasonable concern.**
- See `scripts/select_tuning.py:83` and `:111`.
- After 5 episodes a run has taken 150 to 600 updates, depending on r. A drop that appears by episode 25, or in one seed only, is not reported.
- The 0.10 threshold is Claude's, as the docstring says.
- Fix:
  - Report the drop at every checkpoint and per seed.
  - Have Siddharth fix the threshold, and say whether "the first practice episodes" means 5 or up to 25.

**MI6. The precondition gate takes the primary pairs from the file's `is_primary_pair`. A file listing the tie row as primary was accepted, and the tie row then counted in the mean. Established; the suite never writes such a file.** See `scripts/precondition_check.py:71`. Fix: require the rows found to equal the shorter-route rows of `task_pairs(maze)`.

### Notes (not code defects)

- **N1.** Checkpoints are clones of the weights taken after an episode's updates. They touch no random stream, and their fingerprints are recorded in `summary.json`. A resumed launch with other checkpoints fails, as tested. Their cost is negligible.
- **N2.** Each seed's pretrained policy is evaluated three times per learning rate (once per r cell): 18 pretrained evaluations where 6 would do. The three agree, which is itself a check.
- **N3.** Tuning evaluations compute Rep, A_0 and best-of-m (adaptation is skipped), and store them. The selector never reads them. An S_in-only evaluation would be cheaper, but the stored values are harmless as long as nothing reads them (SF7, M9).
- **N4.** The capacity doubling continues the same learner, with its optimiser state and batch stream, for as many updates again (200,000 in all), and reports both evaluations. This matches "double the updates once".
- **N5.** Capacity evaluation pairs are drawn with replacement from all ordered pairs of free cells, held-out region included. They depend on neither the learner nor the seed. M1's task pairs are all seven rows, the tie row included, as the brief says.
- **N6.** S_in, used by the selector and the early-drop check, covers the shorter-route pairs only (four on M0), not the tie row. That fits "competence on the task pairs" if the tie row is meant to be left out; worth one line in the decision record.

## 5. What was not checked

- Runs at tuning, capacity or precondition size: nothing beyond 300 practice steps, 200 warm-up updates or 600 capacity updates. Whether 100,000 updates reach the capacity thresholds, and the wall-clock cost of the tuning grid, were not measured.
- A capacity run that passes, and so the "passed" branch of `run_capacity.py --all` end to end.
- The Project documents behind decisions 2, 6, 7 and 15 and the planning values; the brief's statement of them was used.
- Mutations were run against the affected test files, not the full suite.
- The uncommitted working-tree edits that appeared during the audit (section 1).
- Determinism across machines or torch versions.

## Response (26 Nov, implementing agent)

All findings were acted on in the commit after this report; the uncommitted edits the auditor saw at 04:50 were these fixes in progress. 228 tests pass (220 before). The training runner still gives identical pretrained and final fingerprints to `689de26` (rechecked for frontier novelty, a tuning run of task-directed practice on M0 at learning rate 0.0003 and r = 2, and demonstrations only).

| Finding | What was done |
| --- | --- |
| SF1 | The launcher runs the whole-folder pretrained check whenever the folder holds runs (`find_runs`), so grid launches are checked. Code is compared by source hashes over each seed, maze, learner and purpose, across grid cells. Grid cells are identified by folder (`OUT/cells/<label>/`, `runs.grid_cell`), not by purpose, so runs outside a grid are never split into cells. Tests: a cell on different code fails; the same runs copied out of their cells fail on their two warm-ups. |
| SF2 | The selector reads the plan from `launch_manifest.json` and refuses: a non-tuning plan; a grid other than the agreed one (unless `--grid-override`); conditions other than the two tuned ones; cells other than the plan's; a missing seed; any run whose settings are not its cell's planned settings. Tests for a missing cell, an extra cell, a run in the wrong cell, and a wrong grid. |
| SF3 | Every evaluation the selector reads must record the expected `which` and the fingerprint the run's summary records for it (final, each checkpoint, pretrained). Tests: another cell's pretrained file, and a "final" evaluation of the pretrained weights, are refused. |
| SF4 | The gate reads the plan's seeds from the manifest and requires exactly those for both learners. Each file must be its folder's learner and its name's seed, a precondition run on M1 with budget 0. All files must share one warm-up apart from learner and seed. `--tuning-report` refuses a learning rate other than the tuned one. The report records each seed's pretrained fingerprint for the pilot to reproduce. The plan's learning rate and ratio are now `null`, and the launcher refuses null settings until they are filled in. |
| SF5 | Decision for Siddharth. Default changed to the reading closest to decision 7(b) as written: under exclusion, the mean is taken over the kept pairs, and the pilot proceeds if at least two pairs remain a side and that mean is at least 0.25 for each learner. Test: the auditor's case P5 now proceeds. |
| SF6 | `--all` claims its output folder with O_EXCL before starting. A child exit code other than 0 (passed) or 1 (failed check) counts as a crash. Each `capacity.json` must match the call's settings and quick flag. Test: a used folder is refused. |
| SF7 | Tests added for M2–M4 (the pass rule at exactly 0.8 and 0.9, a task-pair failure), M3 (M1 reports all seven task pairs), M8 (an expert failure stops the data), M9 (a Rep that would change the choice does not), M13 (SF3's pretrained test), M16 (a tie-row file is refused), M17 (a passing mean with one pair left a side stops), M19 (only the plain learner fails a pair), M22 and M23 (each job's settings match its cell's label; each pretrained evaluation reads its own cell's run), and M26 (a resume with other checkpoints fails, on a clean copy). |
| MI1 | Duplicate cell labels are refused. |
| MI2 | A grid is allowed only for purposes tuning and smoke. A tuning plan must run the two tuned conditions on the weighted learner. `TrainingSettings` refuses purpose tuning except on M0 with the weighted learner. One existing test that used tuning with the plain learner as a stand-in now uses purpose capacity. |
| MI3 | Decision for Siddharth. Kept as the mean over the seven task pairs, the reading of the agreed threshold; the worst pair is now recorded (`task_pairs_worst`). |
| MI4 | The score is the mean of the two conditions' seed means. |
| MI5 | Drops are reported at every checkpoint and for every seed. The flag still uses the mean at the first checkpoint. The threshold and "first episodes" are listed for Siddharth to fix. |
| MI6 | Every file must list exactly M1's six shorter-route pairs. |
| N1–N6 | Noted. N6 (S_in leaves out the tie row) goes to the decision record. |
