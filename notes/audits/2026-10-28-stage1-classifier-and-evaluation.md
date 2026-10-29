# Independent audit: Stage 1 route classifier and evaluation suite (28 October 2026)

Auditor: independent agent (did not write the code). Reference: `docs/spec.md` version 0.9.

## 1. Scope and method

Branches audited:

1. `stage1-route-classifier` (tip `bc04235`), diffed against `origin/main` (`67236e6`).
2. `stage1-evaluation` (tip `579c60c`), diffed against `stage1-route-classifier`.

Nothing in `/home/claude/self-generated-experience` was modified. All work was done in detached worktrees `/home/claude/audit-wt-cls` (route-classifier tip) and `/home/claude/audit-wt-eval` (evaluation tip), using `/home/claude/venv/bin/python`.

Commands and experiments run:

- `git log`, `git diff --stat`, `git show`, `git ls-tree` at each commit; `git merge-base --is-ancestor`, `git merge-tree` (the evaluation branch merges cleanly into `origin/main`; `35331c8` changes only `README.md` and `docs/spec.md` relative to `27881da`, as its message says).
- Full test suite, `python -m unittest discover -s tests`, on both worktrees.
- `python scripts/route_handcheck.py` in the classifier worktree, then `git status`/`git diff` to check the committed sheet is reproduced byte for byte.
- Extracted the stored answer line from `human_comparison.json`, ran `python scripts/check_route_labels.py --answers-file <it> --output <scratch>` and `cmp` against the committed `human_comparison.json`.
- Classifier edge-case probes (points exactly on the midline, start on the midline, touch-and-return, multiple crossings in both arms, crossing in the tie row, M0 midline, M0 trajectory under the M1 rule).
- Mutation test: replaced the last-crossing rule with the first-crossing rule in a throwaway copy; `tests.test_routes` then fails (1 of 9 tests).
- Characterised the 60 hand-check trajectories (crossing counts, crossings in both arms, successes).
- Recomputed every per-pair metric in `results/stage1/evaluation.json` and `evaluation-m0-smoke.json` from the stored per-rollout records (Rep, bootstrap interval, S_in, A_0, best-of-m, P1-own route counts, A_k), and checked seed uniqueness and pairing across policies.
- Re-ran `python scripts/evaluate_stage1.py --quick --maze M0` on Linux and compared with the committed Mac smoke result.
- Recomputed SHA-256 fingerprints of every file recorded in `results/stage1/*.json`.
- Hand-computed the operation totals of one compute scenario (N = 1,000,000, 20 seeds) independently of `src/eval/compute.py`, and the resulting hours from `mac_throughput.json`.
- Wrote two adversarial adapters against `evaluate_policy` to test what the adaptation hook enforces.
- Compared every number quoted in commit messages, `docs/stage1-evaluation.md`, `README.md` and `notes/stage1-handcheck.md` with the recorded JSON.

## 2. Test results

| Branch | Tests | Result | Time |
| --- | --- | --- | --- |
| `stage1-route-classifier` | 58 | OK | 38 s |
| `stage1-evaluation` | 75 | OK | 66 s |

## 3. Verdicts

- **`stage1-route-classifier`: ready after fixes (documentation only).** The classifier is correct on every edge case I tried, the hand-check is reproducible byte for byte and the sheet does not leak answers. One claim about how hard the hand-check sheet is goes beyond the evidence (finding S5). No code change is needed.
- **`stage1-evaluation`: ready after fixes.** No blocking defect: the protocol matches the spec, every recorded number recomputes from the stored rollouts, the M0 smoke run reproduces on Linux, and the compute totals are arithmetically right. The adaptation hook, however, enforces its budget and empty-buffer rules only by trusting what the adapter reports, and the policy interface hands over privileged task information. These are harmless with the Stage 1 stand-ins but are exactly where a Stage 2 learner could silently break budget parity or the inputs rule. They should be closed before the learner is attached (S1 to S3). Provenance should also be tightened (S4).

## 4. Findings

### Blocking

None.

### Should fix

**S1. The adaptation budget is enforced only by the adapter's self-report.** (`stage1-evaluation`)
- *Location:* `src/eval/suite.py:87-96` (`_adapt`), `:136-150`.
- *Evidence:* `_adapt` checks only `result.environment_steps == config.adaptation_steps` and `gradient_steps >= 0`. I wrote an adapter that collected one episode per blocked pair (10 episodes, 1,000 steps, against a budget of 100) and reported `environment_steps=100`. It was accepted with status `standin_no_parameter_updates`. The report's `adaptation_runs` showed 100 steps, while its top-level `environment_steps` silently included the extra 900. A second adapter returned an empty `collection`, an arbitrary `gradient_steps=123456` and the original policy object rather than the copy, and was also accepted. Nothing checks that `len(collection) * horizon == environment_steps`, that the collection is on the blocked tasks, or that gradient steps equal the training ratio times k (Section 8: "the same learner and gradient-step ratio as training"; Section 7 budget audit).
- *Suggested fix:* make the suite own the environment loop. Give the adapter a step-counting collection callable (or a wrapped env factory that raises once k steps are used), and count steps in the suite rather than trusting the report. Check that the collection records sum to k, are on the passed tasks with their walls, and that `gradient_steps == round(ratio * k)` for the configured ratio. Reserve the `standin_*` status strings for `no_update_adaptation`.

**S2. The empty-buffer and no-old-data contract is a label, not a check; evaluation never asserts that a policy was not updated.** (`stage1-evaluation`)
- *Location:* `src/eval/suite.py:91` (`copy.deepcopy(policy)`), `:145` (`"buffer_initialisation": "empty_required_by_adapter_contract"`), `:3` ("Evaluation never updates a policy").
- *Evidence:* the adapter receives a deep copy of the policy object. If the Stage 2 learner stores its replay buffer or optimiser state as attributes of the policy (a common design), the copy carries the condition's old data into adaptation, violating Section 8 ("Practice begins with an empty buffer, so no condition reuses its own old data"), and the report would still say `empty_required_by_adapter_contract`. Likewise, nothing checks that the unblocked and zero-shot evaluation phases left the policy unchanged. The stand-ins are frozen dataclasses, so this cannot happen now.
- *Suggested fix:* define a narrow policy protocol for Stage 2: parameters only, with no buffer or optimiser attributes, enforced by an `assert` on the copy. Have the suite construct the empty buffer and pass it in. Record a parameter hash before and after each evaluation phase and assert they are equal, and record the buffer size at the start of adaptation (must be 0).

**S3. The policy and adapter interfaces pass privileged task information.** (`stage1-evaluation`)
- *Location:* `src/eval/suite.py:24` (`policy.start_episode(obs, pair, rng, sampled)`), `src/eval/suite.py:138` (adapter `tasks` are `(TaskPair, wall_cell)` tuples), `src/eval/protocol.py:150-153` (`TaskPair.shorter`).
- *Evidence:* every policy receives `pair`, which exposes the start and goal cells, the maze name and `pair.shorter` (which route is shorter). The adapter receives the exact wall cell of each blocked task. The spec (Section 5, Inputs) says the learner sees "observation and goal only", and the wall must be experienced only through the environment. The expert stand-in legitimately needs `pair`; a learned-policy wrapper written against the same interface could use `pair.shorter` or the wall coordinates without any test noticing.
- *Suggested fix:* split the interface. Learned policies get `start_episode(observation, rng, sampled)` and adapters get an opaque task handle or env factory whose wall is not readable. Keep `pair` only for the stand-ins, behind a separate privileged interface, and add a test that a learned-policy stub never receives `pair`.

**S4. Result provenance: the recorded commit is unreachable and the tree was dirty.** (`stage1-evaluation`)
- *Location:* `results/stage1/evaluation.json`, `evaluation-m0-smoke.json`, `mac_throughput.json` (`git_commit` `536cb2d…`, `working_tree_dirty: true`; `mac_throughput.json` records no dirty flag).
- *Evidence:* `git cat-file -t 536cb2d` gives "Not a valid object name". All 16 source fingerprints in both evaluation files match the branch tip. But they include `src/eval/compute.py`, `scripts/benchmark_mac.py` and `scripts/estimate_compute.py`, which do not exist at the commit that adds the results (`cab7ef1`) and arrive only in `aa486f0` and `aac24d4`. So the commit series was split after the run and does not reflect the order in which work was done. `mac_throughput.json` has three mismatching fingerprints (`src/eval/suite.py`, `scripts/evaluate_stage1.py`, `scripts/check_route_labels.py`), versions that were never committed. None of these three is imported by the benchmark, whose dependencies (`mazes.py`, `pointmaze.py`, `expert.py`) all match, so the throughput numbers are unaffected.
- *Judgement:* adequate for a Stage 1 harness check. The fingerprints cover all code that can affect the numbers, package versions are recorded, every per-pair metric recomputes from the stored rollouts, and my Linux re-run of the M0 smoke matches the Mac aggregates and per-pair results exactly (rollout floats differ only at the 1e-14 to 1e-4 level). It is not adequate as a pattern for Stage 2 results.
- *Suggested fix:* replace `evaluation.json` with the separate clean M1 re-run once it finishes, from a committed tree. Make `evaluate_stage1.py` (and later training scripts) refuse to write to `results/` when the tree is dirty unless an explicit flag is given, and add `working_tree_dirty` to `benchmark_mac.py`.

**S5. The hand-check sheet is described as harder than it is.** (`stage1-route-classifier`)
- *Location:* `scripts/route_handcheck.py:13-17` ("The sheet is therefore harder than typical rollouts; the agreement rate it gives is a conservative figure"), commit `d04dea5` ("a conservative figure"), commit `bc04235` ("hard cases (dips into one arm, reversals, failures)").
- *Evidence:* of the 60 trajectories, 39 cross the midline exactly once, 14 never cross, 5 cross twice, 1 three times and 1 four times. Only 4 cross in both arms, and only 2 approach the midline within 0.3 cell without crossing. Only 19 of the 60 are successful, and **none of the successful ones crosses more than once**. Yet successful rollouts are the only ones that enter Rep, and multi-crossing successes are the case the last-crossing rule exists for. The "19 that visit both arms" are counted by the cell-set rule; most are shallow dips that never reach the midline. The labeller was also given the rule and the drawn midline before labelling (`notes/stage1-handcheck.md`, commit `1b72e64`). The check therefore verifies that the code implements the stated rule on mostly single-crossing trajectories, not that the rule is right on hard learner rollouts. The synthetic unit tests in `tests/test_routes.py` do cover multiple crossings, and the mutation test confirms they bite. The gate (at least 50 hand-checked trajectories) is met.
- *Suggested fix:* replace "harder than typical rollouts / conservative figure" with a factual description: crossing-count distribution, 19 successes, no multi-crossing successes, rule stated to the labeller. Optionally add a short targeted supplement of successful multi-crossing trajectories, for example from the indecisive controller with a later switch.

### Minor

- **M1. `common_pair_rep` does not restrict to primary pairs.** `src/eval/metrics.py:85-91` accepts whatever rows it is given. A caller passing all seven rows would include the tie row in the C4 − C3 Rep comparison (Section 8 says primary endpoints use shorter-route pairs only). *Fix:* filter to `is_primary_pair` or assert the tie row is absent.
- **M2. Route-class counts use `>=`, the spec says "above".** `src/eval/metrics.py:69`. *Fix:* use `>` or amend the spec. This matters exactly at the threshold, which the spec flags as sensitive.
- **M3. The bootstrap interval ignores the m_min rule inside resamples.** `src/eval/metrics.py:41-47`: resamples with fewer than 20 successes still contribute shares. This is negligible when the observed count is at least 20, but should be documented.
- **M4. `evaluate_stage1.py` overwrites the committed full result by default.** Default `--output` is `results/stage1/evaluation.json` regardless of `--quick` or `--maze M0`. `docs/stage1-evaluation.md:89` names `evaluation-m0.json`, while the committed file is `evaluation-m0-smoke.json`, produced with `--quick` (the docs do not say so). *Fix:* derive the default name from maze and quick, and correct the docs line.
- **M5. Stale dependency statement.** `docs/stage1-evaluation.md:4` says the classifier dependency is `27881da` combined with `67236e6`. The evaluation branch actually builds on `bc04235`, which includes the merge, the sheet update and the human comparison.
- **M6. Classifier labels were promised but never committed.** Commit `d04dea5` says the classifier labels "will be committed with the comparison". `classifier_labels.json` is not in the repository (nor git-ignored). It is not needed, because `check_route_labels.py` recomputes from the displayed positions, and I confirmed that rounding changes no label. *Fix:* commit it or correct the record.
- **M7. The sheet embeds the controller name.** `trajectories.json` and the sheet's embedded data carry `controller` (for example "random", which correlates with "didn't get across"). It is not displayed, and it reveals no classifier output, but it is avoidable. *Fix:* drop it from the blind file.
- **M8. File-header convention.** `scripts/handcheck_template.html` (and so the generated `sheet.html`) has no opening what-and-why comment. `src/eval/__init__.py`, `tests/test_compute.py` and `tests/test_handcheck_labels.py` have one-line "what" docstrings with no "why". The `src/eval/*.py` docstrings are terse and mostly "what", unlike the "What this file does / Why it matters" form used in `routes.py`, `mazes.py` and `expert.py`.
- **M9. Small compute-accounting choices** (all verified, none changes the order of magnitude):
  - the k-grid calibration (`compute.py:75-78`) re-counts C0's 10% run, which the pilot family already counts (a slight overcount);
  - the two alternate-γ pretrained policies get no pretrained evaluation (`compute.py:71-74`; a slight undercount);
  - S_out, P2 and P3 evaluations are not counted, and this exclusion is not listed in the compute limitations;
  - re-learning C0 means retraining on D0 alone, which duplicates C0 itself (conservative).
- **M10. Spec not synchronised.** The evaluation branch edits Section 4 (the P1 cells) without a version change. Section 4's fallback still reads "to be re-measured on the Mac", although 8,749 steps per second has now been measured, and the Section 10 Stage 1 checklist is unchanged. This is acceptable if deliberate pending the audit, but it should be updated on merge.
- **M11. `evaluate_stage1.py` and `benchmark_mac.py` require a git checkout** (`git rev-parse` raises outside one).

### Note

- **N1. Hand-check blindness rests on process.** The sheet and committed files contain no classifier answers. The labeller was told the rule and shown the midline before labelling, and the labels were supplied in chat. Blindness to the classifier's *output* cannot be verified from the repository, but nothing contradicts it.
- **N2. For the expert stand-in, P1-own duplicates P1** (its greedy route is always the shorter one). The two post-adaptation evaluations use different seed phases (1500 and 2500), so their rollouts are not paired with each other. This is fine by design.
- **N3. Seed collisions.** Seeds are 32-bit (`SeedSequence(...).generate_state(1)`). Structurally there are no collisions: phase is a separate entry, not an offset, so `phase+500` and large adaptation indices cannot alias other phases. The birthday probability over about 1,700 seeds is about 3e-4. All 1,700 recorded seeds per policy are distinct.
- **N4. Edge-case behaviour.** A failed rollout that crosses and comes back is labelled by its last crossing, for example "south", not "neither". This is documented and matches the sheet instructions. It affects per-route quality attempt counts and P1-own route counts, not Rep.
- **N5. What dominates the compute estimate.** Gradient updates dominate: about 48 of the 60 hours before overhead at N = 1M and 20 seeds, measured at batch 256 on one CPU thread. The batch size is an assumption, and the hours are sequential single-thread time; the M2 Pro has 10 cores.

## 5. Checked and found correct

**Route classifier (`src/routes.py`, `mazes.RINGS`)**
- The last-crossing rule is implemented as specified. A point exactly on the midline keeps the previous side, so touch-and-return, start-on-line and end-on-line produce no crossing, and passing through a point on the line counts once, with y taken at that point.
- Multiple crossings take the last one. A crossing in the tie row or any non-arm y raises `ValueError`. M0 uses its own midline (column 5, x = −2.0); M1 uses column 4 (x = −2.5). An M0 trajectory classified with the M1 rule raises rather than mislabelling.
- The midline columns are wall in every non-arm ring row on both mazes. Neither held-out region reaches the midline x.
- A successful west-to-east rollout's last crossing is necessarily west to east, so "arm of the last crossing" is the arm that carried it to the goal.
- The mutation to the first-crossing rule is caught by `tests/test_routes.py`.
- M0 geometry matches its description: 24 ring cells, one loop, gaps of 8 and 4 cells with row 3 a tie, and a loop-free held-out region reached only through (3, 10). The M1 gaps of 12, 8 and 4 cells quoted in `40a93a5` are correct.
- The expert drives both routes on every M0 pair, and the route block for a maze without a ring is refused.

**Hand-check**
- `route_handcheck.py` (seeded) regenerates `trajectories.json` and `sheet.html` byte for byte.
- `check_route_labels.py` reproduces `human_comparison.json` byte for byte from the stored answer line.
- The rounded displayed positions give the same labels as full precision (0 differences).
- The fingerprints for `routes.py`, `mazes.py` and `trajectories.json` match both branch tips.
- 56 certain labels, 56 agree (21 north, 24 south, 11 neither). The 4 unsure labels (IDs 13, 35, 48 and 59) also agree and are correctly excluded.
- The sheet's midline marker sits at the correct position (SVG x = 4.5).
- The import is strict: missing, duplicate, extra and malformed tokens are all refused.

**Spec conformance of the evaluation (Sections 4, 6, 7, 8, 9)**
- S_in is greedy. Rep comes from 100 sampled rollouts, is null below m_min = 20, and has a rollout bootstrap interval; the multinomial resampling is equivalent to resampling rollouts.
- The minority-route success rate is taken over all n_roll rollouts. Per-route quality (success given route, time to first success) comes from the pre-block sampled rollouts. Route counts are given at 5%, 10% and 20%, along with entropy.
- A_0 is sampled. Best-of-m uses disjoint groups of m = 5 from the same blocked attempts (documented).
- The P1 wall is the middle cell of the shorter arm: (1, 4) and (7, 4) on M1, (1, 5) and (5, 5) on M0. It is fixed by geometry and identical for every policy.
- P1-own is chosen from greedy rollouts and stratified by the blocked route. A successful "neither" raises in both `rollout` and `summarise_rollouts`.
- The tie pair is excluded from every primary aggregate and still reported per pair.
- Episodes have a fixed length, the goal is fixed, and success is judged at the final step; all three are asserted on every step. Evaluation goals are restricted to R_d through `RequireCells`.

**Adaptation hook, as far as Stage 1 is concerned**
- There is one total k per perturbation across pairs: 1,000 steps each for P1 and P1-own in the M1 run, not 6 × 1,000.
- Each perturbation gets a fresh deep copy, the same uniform task order for every policy, and zero gradient updates in the stand-in, which is labelled `standin_no_parameter_updates`.
- With no adapter, A_k is null and the status is `not_run`.

**Seeding**
- Seeds are paired across policies: all 1,680 non-adaptation (phase, row, seed) triples are identical between the expert and random reports, and so is the adaptation task order.
- Phases 10, 20, 30, 40, 1000, 1500, 2000 and 2500 are distinct entries of the SeedSequence, and P1 and P1-own post-adaptation seeds do not overlap.

**Recorded results**
- Every per-pair metric and aggregate in both result files recomputes exactly from the stored rollouts.
- Episode totals are as expected: 1,700 per policy on M1 (170,000 steps) and 118 on the M0 smoke run.

**Quoted numbers**
- Commit `cab7ef1`: expert S_in 100%, Rep 0.443, minority-route success rate 0.443, A_0 51.3%, best-of-five 99.2%; random S_in 0.8%, A_0 0.7%, best-of-five 3.3%.
- Commit `aa486f0`: 8,749 and 8,630 steps per second, 2,045 updates per second, 36,143 inferences per second, 78 selections per second.
- Commit `aac24d4`: 5.6 hours and 362 hours.
- README, the docs and the hand-check notes: 56 of 56, with 21/24/11.
- All of these match the JSON.

**Compute accounting**
- Conditions: 5 weighted (C0 to C4, C1 optional) plus C3-plain and C4-plain, 7 in all. Collection steps come only from C1 to C4 and the plain arms, while all 7 get matched updates. Network inference covers the 4 policy-collecting conditions, and selections the 2 frontier-novelty conditions.
- Re-learning runs only for weighted conditions. There are 2 adaptations per main policy, at k = 10% of N rounded to whole episodes.
- Also counted: the pilot (6 seeds, no re-learning), γ sensitivity, the k grid, M0 tuning (6 trials × 3 seeds, C3 and C4 with a shared pretraining) and the capacity checks (2 learners × 2 mazes, all ordered free-cell pairs: 35 × 34 + 26 × 25 = 1,840).
- My independent hand computation for N = 1,000,000 and 20 seeds reproduces the recorded totals exactly: 243,916,000 random-dynamics steps, 26,800,000 expert steps, 350,320,000 updates, 217,916,000 inferences and 658,000 selections. It gives 60.21 hours before overhead and 75.26 hours with 25% overhead, matching the file.
- The runtime estimates are not mis-stated anywhere; they are consistently described as sequential CPU planning scenarios, not intervals or a cap.

**Other checks**
- The evaluation never performs gradient updates, and no held-out goal is commanded.
- The `35331c8` merge touches only the README and the spec.
- The benchmark imports only files whose fingerprints match.

## Addendum by the implementing agent (not part of the audit)

- **Full M1 re-run on Linux.** `python scripts/evaluate_stage1.py` was re-run on the cloud workspace (Linux x86-64, Python 3.13.18, the pinned packages). For both stand-ins, every aggregate and every per-pair result is identical to the Mac run in `results/stage1/evaluation.json`. Per-rollout floating-point fields differ slightly (final goal distance by at most 0.097, path length by at most 0.33), and one rollout's first-success step and reward sum differ; every route label and every final success is the same. The committed Mac result was kept; whether to replace it with a clean re-run (finding S4) is left to the author.
- **Commit hashes.** This report was committed on `stage1-route-classifier` after the audit, and the `stage1-evaluation` commits were then rebased on top of it. Their trees are unchanged, but their hashes differ from those quoted above (`cab7ef1`, `aa486f0`, `aac24d4`, `579c60c`).
- **Status of findings.** None of S1–S5 or M1–M11 has been fixed yet. They are listed as open in both pull requests.
