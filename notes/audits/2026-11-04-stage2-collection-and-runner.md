# Audit, 4 Nov 2026: Stage 2 collection loops and training runner (budgets, goal leakage, extra updates)

What this file is: an independent audit of the three commits on `stage2-practice-runner` on top of `e984574`:
- the learner's optional learning-rate warm-up (`781e7e4`);
- the collection loops and the frontier novelty selector (`3c0c28a`);
- the training runner and its script (`c1e3056`).

It lists defects with evidence and fixes, so the author can decide what to change before any tuning, precondition-check or pilot run. Each claim is labelled **established** (shown by code, a test or script output) or **reasonable concern** (plausible, not shown). The auditor was a separate agent that had not written the code. How each finding was handled is in section 6, added afterwards by the implementing agent.

## 1. Scope and method

- Read `git diff e984574..HEAD` in full (9 files, +1,374 lines).
- Read the reference documents: internal spec v0.13, the decision audit of 4 Nov, and the 30 Oct audit (`notes/audits/2026-10-30-stage2-interface-and-learner.md`, especially its SF4).
- Read the code the diff relies on: `src/gcsl.py` (buffer, learner, adapter), `src/eval/suite.py` (`_adapt`), `src/envs/pointmaze.py`, `src/envs/mazes.py`, `src/provenance.py`.
- Wrote adversarial scripts in the session scratchpad, `audit/`: `budget_matrix.py`, `leak_and_misc.py`, `cli_override.py`, `xproc.py`, `validation_rule.py` and `mutate.py` (output in `mutation_results.txt`). No tracked file was modified.
- Mutation testing on exported copies of HEAD (`git archive HEAD | tar -x` into `audit/mutants/<id>/`):
  - 30 single-point mutations, each run against `tests/test_collection.py` and `tests/test_training.py`;
  - four survivors were rerun against the full suite.
- Only tiny smoke runs: warm-up 30 updates, 2 demonstrations per pair, budget 60 or 120 steps, batch 32. No repertoire, competence or recovery number was computed or looked at.

Words used below:
- *N*: the practice budget in environment steps.
- *T*: episode length, 60 steps.
- *r* (ratio): gradient updates per environment step, one of 0.5, 1 or 2.
- *D0*: the demonstration set.
- *Held-out region*: the free cells outside the ring (on M1, the corridor from (4, 8) eastward).

## 2. Test results

**Full suite at HEAD** (`python3 -m pytest -q`): 158 passed, 424 subtests passed in 137 s. The two new test files alone: 32 passed in 21 s.

**Budget matrix (`budget_matrix.py`), established.**
- Coverage: 42 runs. Every condition (demonstrations only, expert top-up, random actions, task-directed practice, frontier novelty, and the plain-learner versions of task-directed practice and frontier novelty), at r = 0.5, 1 and 2, under both schedules.
- The script wraps the collector's environment and counts simulator `step` and `reset` calls, independently of the runner.
- In all 42 runs: simulator steps = reported environment steps = 120 (0 for demonstrations only); resets = 2 (0 for demonstrations only); the learner's counter = Adam's step count = 120 × r; the process-wide optimiser hook = 30 + 120 × r.
- **0 mismatches.** Each learner had one pretrained-weights fingerprint across all conditions, schedules and ratios.

**Across processes (`xproc.py`), established.** Expert top-up, frontier novelty and demonstrations only, each run in its own single-threaded process, gave the same pretrained fingerprint (`f65147ba4669cb0d…`).

**Mutations.** KILLED means a test failed, as it should.

| ID | Mutation | Result |
| --- | --- | --- |
| M1 | Practice keeps the warm-up Adam (no reset) | KILLED |
| M2 | Practice gets warm-up Adam's running averages, with only the step counter zeroed | **SURVIVED** (full suite too) |
| M3 | One stray optimiser step (SGD, learning rate 0) between warm-up and practice | KILLED |
| M4 | Weights changed outside any optimiser between warm-up and practice | **SURVIVED** |
| M5 | Demonstrations only takes one update fewer per episode-sized chunk | KILLED |
| M6 | One extra practice episode | KILLED |
| M7 | Interleaved schedule drops the last step's updates | KILLED |
| M8 | Practice batch random stream depends on the condition | **SURVIVED** (full suite too) |
| M9 | Initial weights depend on the condition | KILLED |
| M10 | Frontier selector never receives collected positions (only D0's) | **SURVIVED** |
| M11 | Validation warm-up rule computes its loss on D0 instead of the held-out demonstrations | **SURVIVED** |
| M12 | Validation demonstrations drawn with D0's seed tag (the same episodes as D0) | **SURVIVED** |
| M13 | Environment-step check removed and an extra episode collected | KILLED |
| M14 | Process-wide optimiser-hook check removed | **SURVIVED** |
| M15 | Start draw depends on the condition | KILLED |
| M16 | Action-sampling stream depends on the condition | **SURVIVED** (full suite too) |
| M17 | Every collector's environment allows held-out goals, and the constructor assertion is removed | KILLED |
| M18 | Per-episode held-out-goal assertion removed | **SURVIVED** (the environment's own check still holds; see MI4) |
| M19 | `set_goal_position` does not check the goal cell | KILLED |
| M20 | Frontier selector takes the highest density | KILLED |
| M21 | Frontier candidates drawn from the 10,000 fitted points instead of all achieved positions | **SURVIVED** |
| M22 | Frontier density on raw positions (no normalisation) | KILLED |
| M23 | Expert top-up always takes the seed's first route | KILLED |
| M24 | Learning-rate ramp not applied on each step | KILLED |
| M25 | Adapter ignores `lr_warmup_in_adaptation` | KILLED |
| M26 | Evaluation config keeps the placeholder ratio of 1 | KILLED |
| M27 | Script does not call the dirty-tree guard | KILLED |
| M28 | D0 check that each demonstration drove its commanded route removed | **SURVIVED** |
| M29 | Random actions' grid rule replaced by continuous forces | KILLED |
| M30 | Commanded goal used as the relabelled goal half the time, instead of a reached position | Survived the new tests; **KILLED** by the full suite (`tests/test_gcsl.py`) |

Of the 30 mutations, 11 survive the new tests and 10 survive the full suite. None of the survivors is a defect in the current code. Each is a gap in the tests, where a future regression would go unnoticed.

## 3. Verdict

**No blocking defect.** On the three standing checks:

1. **Budgets: sound (established).** Every collecting condition takes exactly N environment steps. Every condition, demonstrations only included, takes exactly N × r practice gradient steps, for all three ratios and both schedules. An independent count of simulator calls agrees. Frontier selection, expert planning and resets take no simulator steps. Warm-up is identical across conditions in a seed, within one process and across processes; its random streams and initial weights do not depend on the condition.
2. **Goal leakage: sound (established).** No condition other than frontier novelty could command a held-out goal by any route tried (a held-out reset goal cell, an exact held-out goal point, a point on the edge between (4, 7) and the corridor, expert top-up, random actions, the validation demonstrations). Three layers: the environment's allowed goals, the cell check in `set_goal_position`, and a per-episode assertion. Nothing about the task pair or the route reaches the stored trajectory or the learner.
3. **Extra gradient updates: sound (established).** The Adam reset builds a genuinely new optimiser. The three-way count catches a stray optimiser step (M3) and miscounted chunks (M5, M7). It cannot catch weights changed without an optimiser (M4); nothing in the current code does that.

## 4. Findings

### Blocking

None.

### Should fix

**SF1. The command line silently overrides `seed`, `maze`, `learner` and `purpose` from a `--settings` file. Established.**
- `scripts/run_training.py`: those four arguments have argparse defaults (0, "M1", "weighted", "smoke") written over the file's values unconditionally.
- Evidence (`cli_override.py`): a file asking seed=3 maze=M0 purpose=tuning learner=plain got 0 M1 smoke weighted; a file asking purpose=pilot with k still the placeholder was accepted as purpose smoke, k = 960.
- Why it matters: a tuning run launched with a settings file would tune on M1, which spec section 4 keeps untuned; and the k-placeholder refusal for pilot and main runs is bypassed unintentionally.
- Fix: default those arguments to None, copy them only when given, refuse a file value that conflicts with an explicit flag, and test the cases above.

**SF2. Audit SF4 of 30 Oct is half closed: the evaluation suite still accepts an adapter whose γ (the per-step discount in the relabelling weight), learning rate or batch size differ from training. Established by reading the code.**
- `adapter_for` builds the right adapter, but `_adapt` in `src/eval/suite.py` checks only `updates_per_env_step`, and `EvaluationConfig` has no fields for γ, learning rate or batch size. The adaptation record does not hold them.
- No Stage 2 evaluation script exists yet, so no result is affected.
- Fix: add the three fields to `EvaluationConfig`, filled by `evaluation_config_for`; have `_adapt` compare and record them; test that an adapter with γ = 0.962 is refused for a plain-learner run.

**SF3. Frontier novelty's "normalised positions" is a fixed division by 5, which gives a wider effective bandwidth than a per-axis standardisation would. The scaling is established; how MEGA normalises is a reasonable concern not verified (the fetch of MEGA's code was blocked).**
- D0's positions have standard deviations of 2.36 and 2.23 cells. With the fixed scaling, bandwidth 0.1 is 0.5 cells; standardised per axis it would be about 0.23 cells, shrinking as frontier novelty spreads.
- From recollection, MEGA's code standardises the fitted samples by their mean and standard deviation before fitting.
- The fixed scaling has one advantage: its bandwidth does not drift as the data spreads. This is a weakness in how the choice was documented, not evidence that another choice is better.
- Fix: check MEGA's density code before tuning; then match it or record the fixed scale as a stated deviation, giving the bandwidth in cells. Settle it before the M0 tuning runs.

**SF4. Test gaps on the paired-seed design, frontier novelty's data and the validation rule. Established (mutation survivors).**
- Paired seeds (M8, M16): the code shares practice batch draws and action-sampling draws across conditions correctly, but making either depend on the condition breaks no test. Fix: test identical first-batch indices across conditions, and identical action-stream state for episode i.
- Frontier novelty's own data (M10, M21): if the selector stops receiving collected positions, or candidates come only from the fitted subsample, no test notices. Fix: check the selector's achieved set against D0 plus 61 positions per collected episode (or build it from the buffer); unit-test that candidates span the whole set.
- Validation rule (M11, M12): computing the validation loss on D0, or drawing validation demonstrations with D0's seed tag, survives. Fix: test that validation start states differ from every D0 start state and that the validation batch is built from the validation set.

### Minor

**MI1. The validation warm-up rule can overshoot its maximum. Established.** With max 100, checks every 30, min 30, it took 120 updates. Defaults unaffected. Fix: require the minimum and maximum to be multiples of the check interval.
The rule looks only at the weighted loss on its own held-out demonstrations (a forward pass without gradients), never at Rep or collected data (established).

**MI2. Nothing checks that practice starts from exactly the pretrained weights (M4), or that the new Adam's running averages start empty (M2). Established.** The current code is correct. Fix: assert the parameter fingerprint equals the pretrained one, and the new optimiser's state is empty, right before practice.

**MI3. No test of a failing budget audit (M14). Established.** Fix: a test that injects one extra optimiser step and expects the audit to fail.

**MI4. Two defence-in-depth checks are untested (M18, M28). Established.** The per-episode held-out-goal assertion cannot fire today because the environment refuses such goals first; the D0 route check has no test showing it fires. Fix: one test for each.

**MI5. Expert top-up's pair allocation is random, not cycled. Established; partly an open decision.** Over 200 episodes (seed 0), rows 1 to 7 got 28, 26, 23, 31, 30, 37, 25 episodes; within each row north and south differed by at most 1; all succeeded on the commanded route. Decision 1 of the 4 Nov audit recommended cycling pairs; the code uses the start draw shared with the other conditions (decision 4). The two recommendations conflict for expert top-up. Fix: record the conflict and let Siddharth choose.

**MI6. Small settings gaps. Established.**
- Pilot and main runs accept non-agreed settings (for example 2 demonstrations per pair, batch 32, k = 0). Fix: for pilot and main runs require M1, batch 256, 20 demonstrations per pair, a budget of 249,960 or 999,960, and k > 0.
- `learning_rate` is not checked to be positive.
- `warmup_key()` is unused, and no code compares pretrained fingerprints across runs, though the docstring says the audit can.
- Restarting a crashed run in the same folder silently overwrites its `episodes.jsonl`. Fix: refuse any non-empty folder.

### Notes (not code defects)

- **N1.** Under "interleaved", updates during episode i sample a buffer without episode i, and the acting policy changes within the episode; the last episode's data is never trained on. Under "episode_then_updates" it is. Both take exactly N × r updates; the per-step counts are correct for all three ratios.
- **N2.** Building D0 and the validation demonstrations runs the simulator (140 and 28 episodes on M1). Both are shared warm-up, so parity is unaffected; worth one line in the summary.
- **N3.** Warm-up is recomputed in every process. Deterministic, so a compute cost, not a correctness issue.
- **N4.** No two random-stream seed lists in the diff coincide (NumPy's `SeedSequence` ignores trailing zeros; checked every list).
- **N5.** `lr_warmup_in_adaptation` defaults to False; decision 14 recommends True if the fallback is ever triggered. Off by default either way.
- **N6.** Random actions default to continuous forces (decision 5 open); stored labels differ from executed actions by up to 0.125 per axis.

## 5. What was not checked

- Long runs: nothing beyond 120 practice steps or 30 warm-up updates; the frontier selector over thousands of episodes was not measured.
- MEGA's density code: not read (blocked), so SF3's comparison rests on recollection.
- Determinism across machines or torch versions.
- End-to-end evaluation from a saved `final.pt` (no Stage 2 evaluation script exists yet).
- M0 through the runner (budget matrix used M1 only).
- The data-diversity log itself (audited 30 Oct); only its wiring was checked here.

## 6. Response (implementing agent, same day)

See the commit that follows this report for the changes. Summary: SF1, SF2, SF4, MI1 to MI4 and the MI6 items fixed in code and tests. SF3 and MI5 are design choices left to Siddharth (listed in the PR description and spec v0.14), with SF3's MEGA check reported there. N5 and N6 follow the open decisions.
