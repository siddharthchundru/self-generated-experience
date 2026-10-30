# Audit, 30 Oct 2026: Stage 2 evaluation interface, provenance guard, learner and D0

What this file is: an independent audit of the four commits on `stage2-learner` on top of `9fa0541` (single episode length; dirty-tree guard; evaluation interface fixes for the 28 Oct audit's S1 to S3; weighted GCSL learner and D0). It lists defects with evidence and fixes, so the author can decide what to change before Stage 2 results are filed.

## 1. Scope and method

- Read `git diff 9fa0541..HEAD` in full, plus `docs/spec.md` sections 4 to 8, the 28 Oct audit (S1 to S4) and the 30 Oct audit (B2, S3).
- Wrote adversarial scripts against the interface, the guard and the learner. They are in the session scratchpad (`audit/attacks.py`, `guard_test.py`, `guard_results_dirty.py`, `crosspair.py`, `demos_check.py`, `floor_and_determinism.py`, `more.py`). No tracked file was modified.
- Mutation testing on an exported copy of HEAD (scratch directory), running the relevant test classes after each mutation.
- Compared the relabelling sampler with GCSL's released code (`gcsl/algo/buffer.py`, `_sample_indices`).

## 2. Test results

`python3 -m pytest -q -p no:warnings`: **120 passed, 424 subtests passed in 77 s.**

Mutations (KILLED means a test failed, as it should):

| Mutation | Result |
| --- | --- |
| Remove the optimiser step-count cross-check | KILLED |
| Relabelled goal taken from t′−1 | KILLED |
| Weight γ^(t′−t+1) instead of γ^(t′−t) | KILLED |
| Remove the empty-buffer check | KILLED |
| Policy view passes the whole observation dictionary | KILLED |
| **Remove the post-adaptation parameter check** (`adapted_guard.check`) | **SURVIVED** |
| **Never use the final state t′ = T as a relabelled goal** | **SURVIVED** |

## 3. Verdict

**No blocking defect.** The learner matches spec section 5, D0 matches section 6, the single episode length is enforced for evaluation and the compute estimate, and the S1 to S3 fixes catch every *honest mistake* I tried (miscounted gradient steps, a reused learner, a prefilled buffer, a returned substitute policy, a wrong ratio, extra episodes through the session). What remains open:

- The provenance guard (S4) will refuse the *second* Stage 2 result written from a clean tree. That will push routine use of `--allow-dirty`, which defeats the guard (SF1).
- Four scripts still write into `results/` with no guard (SF2).
- Several S1/S2 checks can be bypassed by an adapter or policy acting deliberately. Most of these are a trust boundary that Python cannot enforce. Four of them are cheap to close (SF3).

The adapter and policy are project code, and `GCSLAdapter`/`GCSLPolicy` trigger none of the bypasses, so no current result is affected.

## 4. Findings

### Blocking

None.

### Should fix

**SF1. The dirty-tree guard counts the results it just wrote as uncommitted changes, so the next run is refused.**
- *Location:* `src/provenance.py:34` (`git status --porcelain` over the whole tree), `:63` to `:76`. `results/` is not in `.gitignore`.
- *Evidence:* `guard_results_dirty.py` creates a throwaway repository, passes the guard, and writes `results/seed0.json`. A second call for `results/seed1.json`, with the source unchanged, then raises: "refusing to write …: the working tree has uncommitted changes". Stage 2 runs one process per seed and condition, so every run after the first is refused unless results are committed between runs or `--allow-dirty` is passed. If `--allow-dirty` becomes the habit, a genuinely dirty source tree goes through unnoticed.
- *Related:* `scripts/evaluate_stage1.py` calls `run_record(ROOT)` (git state and source hashes) *after* the evaluation, while the guard runs before it. If a file is edited during a long run, the record hashes code that did not run. Also, on a case-insensitive filesystem (macOS default), `RESULTS/x.json` points into `results/` but passes the guard (`guard_test.py`: "ALLOWED dirty, RESULTS/x.json").
- *Fix:* decide "dirty" from tracked and untracked changes outside `results/` (for example `git status --porcelain -- . ':(exclude)results'`), or only under `src/ scripts/ tests/ docs/`. Take `run_record` once at start, before any work, and store that copy. Optionally compare paths with `os.path.samefile` on the parent directory, or case-fold on Darwin.

**SF2. Four scripts still write into `results/` with no guard.**
- *Location:* `scripts/estimate_compute.py:53` (changed in this diff, writes `results/stage1/compute_estimate.json`); `scripts/measure_route_gap.py:122` (records dirtiness of `src` only, as a suffix on the commit string); `scripts/route_handcheck.py:111` to `:115`; `scripts/check_route_labels.py:65`.
- *Evidence:* `grep guard_results_write` matches only `benchmark_mac.py` and `evaluate_stage1.py`. The commit title says results are refused from a dirty tree; for these four scripts that is not true.
- *Fix:* call `guard_results_write` and `run_record` in each of them, or add one shared `write_result(path, payload, allow_dirty)` helper in `provenance.py` that every script must use. A test could then grep `scripts/` for `write_text` calls into `results` that bypass the helper.

**SF3. Four adaptation-check bypasses that can be closed cheaply (audit S1/S2).**
All are from `attacks.py`, with k = 2 episodes and ratio 1, and every one was *accepted* with a clean report (`environment_steps=120, gradient_steps=120, optimizer_steps=120`):
- (a) *The step counter can be written to.* `AdaptationSession.steps_used` is a public attribute. An adapter that resets it to 0 after each pair of episodes ran **5 episodes (300 steps) against k = 120** and was accepted. *Fix:* store it as `_steps_used` behind a read-only property. Also check `len(session._records) * horizon == config.adaptation_steps` in `_adapt`, which does not depend on the counter.
- (b) *A `NoUpdateAdapter` subclass can update parameters and still report `standin_no_parameter_updates`.* The subclass ran 500 Adam steps on the clone and was accepted with `gradient_steps=0`. The check is `isinstance(adapter, NoUpdateAdapter)` (`suite.py`, `_adapt`). *Fix:* use `type(adapter) is NoUpdateAdapter`. For any stand-in mode, also assert that the clone's digest is unchanged (for stand-ins, compare `copy.deepcopy` equality or skip, since they hold no parameters).
- (c) *The returned optimiser is not tied to the policy, and other optimisers are not counted.* Changing parameters under `torch.no_grad()` and returning a dummy Adam over an unrelated tensor, stepped k times, was accepted. So were 5,000 extra steps on a second optimiser that was never returned. *Fix:* assert that `{id(p) for g in result.optimizer.param_groups for p in g["params"]} == {id(p) for p in clone.network.parameters()}`. Wrap `adapter.adapt` in `torch.optim.optimizer.register_optimizer_step_post_hook` (available in the installed torch 2.14). It counts every optimiser step taken in the process during adaptation, whichever optimiser took it; assert that count equals the reported gradient steps. `more.py` confirms the hook counts steps across two optimisers.
- (d) *The parameter digest is computed by the policy itself.* A policy whose `parameter_digest()` returns a constant and that adds 0.1 to a bias at every `start_episode` passed a full evaluation. *Fix:* compute the digest in the suite, over `named_parameters()` and `named_buffers()` of every `nn.Module` found in `vars(policy)`. `check_parameters_only` already walks those modules.
- *Remaining trust boundary (not fixable in-process, and acceptable for project code):* an adapter can call `collect()` or `make_env` directly on the blocked task. `attacks.py` collected 600 uncounted blocked-environment steps this way, accepted. It can also read `session._tasks` (all six rows and walls leaked) or introspect closures. State this boundary in the `suite.py` docstring, and rely on code review plus the rule that only `GCSLAdapter` is used for A_k.

**SF4. Only the update ratio of the adapter is checked against training. Its γ, learning rate and batch size are not.**
- *Location:* `src/gcsl.py`, `GCSLAdapter.__init__`; `src/eval/suite.py`, `_adapt` (checks `updates_per_env_step` only); `EvaluationConfig.adaptation_updates_per_env_step` defaults to a placeholder 1.0.
- *Evidence:* spec section 8 says adaptation uses "the same learner and gradient-step ratio as training". `GCSLAdapter` is constructed separately from the training `GCSLLearner`. Nothing stops, for example, adapting a plain-GCSL (γ = 1) policy with γ = 0.962, or using a different learning rate than the tuned one. The report records only `gcsl_gamma_<γ>`.
- *Fix:* when the collection loops are written, build the adapter from the training run's learner settings (a single frozen settings object), and record γ, learning rate, batch size and ratio in `adaptation_runs`. Add a test that a mismatch is refused.

### Minor

**MI1. Two test gaps (mutation survivors).**
- Deleting `adapted_guard.check(...)` in `evaluate_policy` survives the whole suite: no test has a policy whose evaluation changes the *adapted* copy. *Fix:* in `TestAuditInterface`, add a stub whose `start_episode` changes parameters only when a flag is set by the adapter on the clone.
- Restricting relabelling to t′ ≤ T−1 (never the final state) survives: the gap and start-index distributions differ by at most 0.0005, under the tests' 0.002 tolerance. The final state matters because success is defined at the final step. *Fix:* in `test_goal_is_a_later_position…`, assert `(t_goal == T).any()`, and assert that the share of t′ = T is close to T/(T(T+1)/2) = 1/30.5.

**MI2. `check_parameters_only` lets data through in several ways.**
- *Evidence (`attacks.py`):* a registered torch buffer holding 1,000 trajectories is accepted, whether persistent or not. A non-persistent one is also missing from `state_dict()`, so it is outside the digest. An underscore-prefixed list on the network module (`network._replay`) is accepted, because `_`-prefixed module attributes are skipped. So are class-level attributes, data in a string attribute, and a forward-hook closure. A plain attribute on a nested submodule is correctly refused.
- *Fix:* allow-list buffer names (currently only `input_scale`) or cap total buffer size. Refuse `_`-prefixed module attributes other than torch's own (`_parameters`, `_buffers`, `_modules`, the hook dictionaries), and require the hook dictionaries to be empty. Compute the digest over `named_buffers()` as well as parameters (see SF3 d). Class attributes and strings remain a trust boundary.

**MI3. The rng passed to a learned policy reveals the evaluation phase.**
- *Evidence:* `rollout` passes `np.random.default_rng(SeedSequence([seed, 1]))`, and `rng.bit_generator.seed_seq.entropy[0]` is the episode seed. Because `episode_seed` uses public inputs, a policy can tabulate it: **88 of 88** rollouts in `attacks.py` were decoded to (row, phase). The row is legitimately inferable from the goal anyway. The phase is not: phase 40 (zero-shot) or 1000/2000 (adaptation) means "a wall is present".
- *Judgement:* the same class of trust boundary as reading `session._tasks`. Any deterministic generator built from enumerable seeds can be decoded by precomputing its state, so this cannot be closed in-process. Note it in the docstring. A code-review rule that policies use `rng` only through `choice`, `random` or `integers` is the realistic control.

**MI4. The relabelling docstring misattributes its distribution to GCSL's released code.**
- *Location:* `src/gcsl.py`, module docstring, "Relabelling" bullet.
- *Evidence:* GCSL's `_sample_indices` draws `time_idxs_1` from the range [0, L−2] and `time_idxs_2` from [0, L−1], and bumps the second index by one on a tie. Over pairs that gives 1.5 times the uniform probability to adjacent pairs (gap 1) and 0.5 times to pairs ending at the last state (computed exactly). The repository's sampler is exactly uniform over the T(T+1)/2 pairs, which the tests verify. So it is *not* what GCSL's code does.
- *Fix:* reword to "uniform over pairs; GCSL's code uses a similar min/max construction that is approximately, not exactly, uniform". The choice itself is a reasonable reading of the spec ("a start index t and a later goal index t′ are sampled uniformly"); see Note N4.

**MI5. Remaining ways to use another episode length without being refused.**
- `TrajectoryBuffer(episode_steps=…)` accepts any length, and `make_env(max_episode_steps=…)` accepts any value. A future training loop that passed 100 to both would train on 100-step episodes silently. Evaluation and the compute estimate would refuse, but training would not.
- `scripts/route_handcheck.py:42` has its own literal `T = 60`, not imported from `episode.py`, and is not labelled as a deliberate choice. `measure_route_gap.py` (T = 100) and `benchmark_mac.py --horizon` are labelled measurement paths, and `estimate_compute.py` refuses measurements at another horizon.
- *Fix:* drop the `episode_steps` parameter from `TrajectoryBuffer`, or assert it equals `EPISODE_STEPS`. Import `EPISODE_STEPS` in `route_handcheck.py`. Optionally have the future collection loop assert `env.spec.max_episode_steps == EPISODE_STEPS`.

**MI6. `make_demonstrations` does not check that each demonstration succeeded or drove the commanded route.**
- A failed or wrong-route expert episode would enter D0 silently and change the 50/50 split in effect. On M1 at seed 0, all 140 succeeded and all route labels match their command (`demos_check.py`), so this is a guard for later seeds or expert changes. *Fix:* assert `success` and `trajectory_route(...) == route` per demonstration, or record failures and refuse.

### Note (not code defects)

**N1. A second mechanism for a shorter-route preference under γ = 1: shared sub-paths on the ring (cross-pair relabelling).**
- The north route of any row r runs up the west column through the start cells (r′, 1) of every row above, along the north arm, and down the east column through the goal cells (r′, 7). So every north-route demonstration from a lower row contains, in order, a start-cell-to-goal-cell segment of each higher row, via the north arm, which is the *shorter* arm for those rows. The same holds for the south arm. Relabelling turns those segments into training examples for the other pairs.
- *Measured on D0 (M1, seed 0).* `crosspair.py` counts relabelled pairs (t, t′) with the state in pair r's start cell and the goal in its goal cell, weighted by γ^(t′−t), split by route. "Shorter-route share" is the shorter route's fraction of that example mass.

  | Row | γ = 1, own-pair examples only | γ = 1, all D0 | γ = 0.962, own-pair only | γ = 0.962, all D0 |
  | --- | --- | --- | --- | --- |
  | 1 | 0.69 | 0.75 | 0.80 | 0.87 |
  | 2 | 0.60 | 0.63 | 0.69 | 0.74 |
  | 3 | 0.55 | 0.56 | 0.60 | 0.63 |
  | 5 | 0.55 | 0.57 | 0.60 | 0.63 |
  | 6 | 0.63 | 0.65 | 0.71 | 0.75 |
  | 7 | 0.68 | 0.74 | 0.79 | 0.87 |

  For rows 1 and 7, the other rows' demonstrations contribute 416 examples at γ = 1, all on the shorter arm and none on the longer.
- *What this establishes and what it does not.* The per-pair skew of own-pair examples under γ = 1 is the spec's stated mechanism (early arrival keeps producing examples at the goal). The cross-pair contribution is additional, is strongest on the outer pairs, and also acts under γ < 1. These are counts of exact cell-matched examples, not of what the network learns, because it generalises across nearby goals. They are consistent with the smoke run's ordering (outer pairs had the lowest minority shares, and plain GCSL preferred the shorter route on each side), but they do not prove that this is its cause.
- *Suggestion:* add this mechanism to spec section 5, "Expected effect at pretraining", next to the early-arrival note. The 0.65 prediction understates the skew for two reasons, not one.

**N2. The compute estimate cannot currently be re-run, and the committed one is stale.**
`results/stage1/mac_throughput.json` was measured at horizon 100, so `estimate_compute.py` now refuses ("re-measure with the chosen horizon"), which is the intended behaviour. The committed `compute_estimate.json` still reflects T = 100 and the old budgets. Re-measuring at T = 60 is the open item from the 30 Oct audit, S3.

**N3. The spec does not yet record the agreed Stage 2 settings.**
Section 6 still says "T will be roughly 50–60", and N_d, N and the adaptation ratio appear only in code docstrings. Record T = 60, N_d = 20 per pair (10 per route), N = 249,960 for the pilot and 999,960 for the cap, and the tuning budget of 99,960 in the spec.

**N4. The relabelling distribution is a judgment call, and a defensible one.**
Uniform over pairs gives start index 0 twice the share it would get under "t uniform", and the share falls linearly to t = T−1. So states early in an episode, including the forks near the start cells, get more training weight than under the alternative reading. Both readings fit the spec's sentence. The choice is stated and tested, and it should be named in the paper.

**N5. The first adaptation updates train on a single trajectory.**
`GCSLAdapter` starts from an empty buffer and interleaves collection with updates, so at ratio 1 the first 60 gradient steps use one episode. This follows the spec (empty buffer, same schedule as collection), but it will shape A_k at small k and is worth keeping in mind when calibrating the k grid on demonstrations only (C0).

## 5. Checked and found correct

- **Episode length.** `EPISODE_STEPS = 60` is imported by `pointmaze.make_env` (default), `EvaluationConfig` (any other horizon is refused, and the test covers it), `ComputeAssumptions` and `counts` (refuses), `GCSLAdapter` and the buffer (default), and D0 (via the environment's default). Every M1 demonstration has shape (61, 4).
- **S1, honest-error coverage.** Exceeding k raises `BudgetExhausted`. Using less than k is refused. Misreported gradient steps, a missing optimiser, a wrong ratio and a substituted returned policy are refused. A learner reused across P1 and P1-own reports 240 steps where 120 are required, and is refused. A ratio times k is always a whole number, because the configuration requires ratio × T to be whole and k to be whole episodes. The session fixes the policy at construction and samples tasks itself. Copying a policy so that the copy shares parameters with the original is caught by the original-policy digest check after adaptation.
- **S2.** The adapter's buffer starts empty (checked). The policy holds only `name` and `network`. A fresh Adam optimiser is created per adaptation. Digests are checked after the unblocked and zero-shot phases, on the original after each adaptation, and on the adapted copy after its post-adaptation evaluation. Gradients do not travel with the deep copy.
- **S3.** Learned policies receive exactly `{"observation", "desired_goal"}` at `start_episode` and at every step (tested, and the test kills the mutation). Session trajectories contain only observations, actions and the commanded goal.
- **S4 path handling.** The guard refuses dirty writes for absolute paths, relative paths from the repository root and from `src/`, a symlinked directory, and a symlinked file pointing into `results/`. Outside a git checkout it requires `--allow-dirty`.
- **Learner against spec section 5.** The relabelled goal is the position at t′ > t, with t in 0..T−1 and t′ in 1..T (the final state is included). The label is the action at t. Trajectories are drawn uniformly. The weight is γ^(t′−t), and the loss is the weighted sum over the sum of weights; γ = 1 gives the plain mean. Effective sample size is (Σw)²/Σw². The joint 9 × 9 grid round-trips, is reflection-symmetric, and uses the x bin as the major index. The expert never exceeds |force| 1, and nearest-bin quantisation changes forces by 0.015 on average per axis. A closed-loop expert with quantised actions succeeded on its commanded route in 70 of 70 M1 episodes. Sampling is at temperature 1 with double-precision softmax, and greedy is the argmax. Input scaling is fixed. The seed sets initial weights without disturbing global torch state. Training, and evaluation including adaptation, reproduce exactly from seeds (`floor_and_determinism.py`). `updates_per_episode` gives 30, 60 and 120 for ratios 0.5, 1 and 2. `GCSLAdapter` matches the common procedure: empty buffer, same learner code, ratio × T updates per whole episode, and only episodes from the session.
- **D0 against spec section 6.** Exactly 10 north and 10 south demonstrations per pair, tie row included, in seed-shuffled order. 140 distinct reset seeds. Zero overlap with evaluation seeds across demonstration seeds 0 to 19 and every evaluation phase. Every M1 and M0 demonstration stays in the demo region, and the environment refuses held-out goals. All 140 M1 demonstrations succeed.
- **Compute (B2).** k is now rounded down to whole episodes, and the floating-point floor never loses an episode for any budget that is a multiple of 60, up to 2,000,000, at 5%, 10% and 20%. The updated `test_compute.py` figures are arithmetically right (99,960 = 1,666 episodes; 10% gives 166 episodes, 9,960 steps).
- **Header comments.** Every new or changed source and test file opens with a docstring saying what it does and why. The exception to their accuracy is MI4.
