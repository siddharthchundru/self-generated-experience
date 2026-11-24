# Audit, 24 Nov 2026: re-learning control, run evaluation, pretrained check and launcher

What this file is: an independent audit of the five commits on `stage2-eval-relearning-launcher` on top of `c91355e`:
- saving each run's final buffer and its digest (`315b584`);
- run folders and the pretrained-policy check across runs (`0d0de56`);
- the re-learning control (`8fcb652`);
- the Stage 2 evaluation script (`824ab4c`);
- the launcher (`8b00988`).

It lists defects with evidence and fixes, so the author can decide what to change before any pilot run. Each claim is labelled **established** (shown by code, a test or script output) or **reasonable concern** (plausible, not shown). The auditor was a separate agent that had not written the code and did not trust its docstrings.

## 1. Scope and method

- Read `git diff c91355e..HEAD` in full (12 files, +1,193 lines), and the code it relies on: `src/training.py` (settings, warm-up, runner, budget audit, `adapter_for`, `evaluation_config_for`), `src/gcsl.py` (buffer, learner, adapter), `src/eval/suite.py` (`_adapt`, the parameter guard), `src/eval/protocol.py` (`EvaluationConfig`), `src/provenance.py`, and the re-learning, paired-seed and evaluation rules of `docs/spec.md` (sections 7 and 8).
- Decision 13 of the 4 Nov decision audit is a Project document, not in the repository. Its reading was taken from the brief: warm-up on D0 for the original's recorded number of updates, Adam reset, then N × r updates on the whole final buffer, main learner only.
- Wrote adversarial scripts in a scratch folder outside the repository (`/tmp/audit/scripts/`):
  - `digests.py`: the training runner at `c91355e` against HEAD;
  - `relearn_audit.py`, `valearly.py`: independent budget, buffer and no-environment checks of the re-learning control;
  - `tamper.py`: edited and swapped run files;
  - `check_pre.py`: false passes and crashes of the pretrained check;
  - `launch_crash.py`: launcher failure handling;
  - shell experiments for concurrent evaluations, concurrent launchers, a resumed launch with a changed plan, and a pilot plan with quick evaluation;
  - `mutate.py`: mutation testing.
- The two trees were exported with `git archive` into `/tmp/audit/base` (`c91355e`) and `/tmp/audit/head` (HEAD). No git worktree was added, and no tracked file was modified except this report. Nothing was written into `results/`.
- Only tiny smoke runs: warm-up 30 updates, 2 demonstrations per pair, budget 60 or 120 steps, batch 32, and `--quick` evaluations. No repertoire, competence or recovery number was looked at for its value.

Words used below:
- *N*: the practice budget in environment steps.
- *T*: episode length, 60 steps.
- *r*: gradient updates per environment step (0.5, 1 or 2).
- *D0*: the demonstration set.
- *k*: the adaptation budget in environment steps for recovery; 960 is a placeholder.
- *A_k*: success after the common adaptation procedure with budget k.
- *Rep*: the repertoire measure.
- *γ*: the per-step discount in the relabelling weight; 0.962 for the main (weighted) learner, 1 for the plain learner.
- *Fingerprint*: the SHA-256 digest of a network's weights (`parameter_digest`).

## 2. Test results

**Full suite at HEAD** (`python -m pytest -q`): 190 passed, 424 subtests passed in 214 s. The three affected test files alone: 45 passed in 64 s.

**Training runner unchanged (`digests.py`), established.**
- Coverage: 11 settings, run with `c91355e` and with HEAD:
  - every condition with the main learner;
  - the plain learner for task-directed practice and frontier novelty;
  - frontier novelty at r = 0.5 with the interleaved schedule;
  - task-directed practice at r = 2 with the learning-rate fallback;
  - random actions on M0 with seed 3;
  - expert top-up with the validation warm-up rule.
- Pretrained fingerprint, final fingerprint, episode log and summary were compared, the summary without its three new keys.
- All identical. The main learner's seed-0 pretrained fingerprint is `f65147ba4669cb0d…`, the value recorded in the 4 Nov audit.

**Re-learning control (`relearn_audit.py`), established.**
- Coverage: 9 originals, each re-learned with `init fresh` and `init same`:
  - demonstrations only, expert top-up and random actions;
  - task-directed practice at r = 0.5;
  - frontier novelty at r = 2, interleaved;
  - frontier novelty with the plain learner;
  - the learning-rate fallback;
  - a validation-rule original;
  - N = 0.
- Every optimiser step was counted by the auditor's own hook. The environment constructors were patched to raise at `gymnasium.make`, `envs.pointmaze.make_env`, `collection.make_env` and `demonstrations.make_env`.
- In all 18 re-learning runs:
  - the counted total equals the original's warm-up plus practice (for example 30 + 240 = 270 at r = 2);
  - warm-up and practice each match the original separately;
  - no environment was built;
  - `init same` reproduced the original pretrained fingerprint, and `init fresh` did not.
- The saved buffer equals, row by row and in order, every trajectory added to the run's buffer, D0 first. The D0 rows equal a freshly rebuilt D0.
- A validation-rule original that stopped early (40 updates, maximum 200) was matched by a 40-update re-learning warm-up, with the same pretrained fingerprint under `init same` (`valearly.py`).
- Fresh re-learners of task-directed practice, frontier novelty and demonstrations only in seed 0 share one pretrained fingerprint (`00a026e0…`); seed 1 differs (`a9118da8…`).

**Tampering (`tamper.py`, `check_pre.py`), established.** Refused:
- another run's `final_buffer.npz`;
- one observation nudged by one float32 unit in the last place;
- another run's `final.pt`;
- a `settings.json` edit, by re-learning;
- writing a re-learning run into its source folder.

Accepted:
- a `settings.json` edit, by evaluation (SF1);
- a consistent swap of `final.pt` together with `summary.json` (N4);
- an extra array in the buffer file (harmless).

**Mutations.** Each mutation was applied alone to a fresh copy of HEAD and run against `tests/test_relearning.py`, `tests/test_run_scripts.py` and `tests/test_training.py`. KILLED means a test (or a run-time check a test triggers) failed, as it should.

| ID | Mutation | File | Result |
| --- | --- | --- | --- |
| M1 | Re-learning practice takes one update fewer per chunk | `src/relearning.py` | KILLED |
| M2 | Re-learning skips its buffer-digest check | `src/relearning.py` | KILLED |
| M3 | `load_buffer` skips its buffer-digest check | `src/runs.py` | **SURVIVED** (re-learning's own check still catches it at run time) |
| M4 | No Adam reset: re-learning practice reuses the warm-up learner and its Adam | `src/relearning.py`, `src/training.py` | KILLED |
| M5 | Adam reset in name only: warm-up Adam's running averages copied in after the empty-state check, step counter zeroed | `src/relearning.py`, `src/training.py` | **SURVIVED** (full suite too) |
| M6 | Evaluation uses a default `EvaluationConfig` instead of the run's | `scripts/evaluate_run.py` | KILLED |
| M7 | Evaluation's adapter uses a hard-coded learning rate of 0.001 | `scripts/evaluate_run.py` | **SURVIVED** (tests use 0.001 only; the suite would refuse other rates at run time) |
| M8 | `--which pretrained` evaluates the final weights | `scripts/evaluate_run.py` | **SURVIVED** |
| M9 | `--quick` no longer refused for pilot and main runs | `scripts/evaluate_run.py` | KILLED |
| M10 | Pretrained check ignores the commit | `src/runs.py` | KILLED |
| M11 | Pretrained check never reloads `pretrained.pt` | `src/runs.py` | KILLED |
| M12 | Pretrained check skips the `init same` reproduction check | `src/runs.py` | **SURVIVED** |
| M13 | Launcher ignores dependencies | `scripts/launch.py` | KILLED |
| M14 | Launcher skips the pretrained check before the pretrained evaluation | `scripts/launch.py` | **SURVIVED** |
| M15 | Launcher's pretrained check covers only the first run of the group | `scripts/launch.py` | **SURVIVED** |
| M16 | Pretrained evaluation depends only on the group's first training run | `scripts/launch.py` | KILLED |
| M17 | Launcher reruns jobs whose output exists | `scripts/launch.py` | KILLED |
| M18 | Launcher does not set the one-thread environment variables | `scripts/launch.py` | **SURVIVED** |
| M19 | `run_relearning.py` does not pin one thread | `scripts/run_relearning.py` | **SURVIVED** |
| M20 | Re-learning warm-up trains on the whole final buffer instead of D0 | `src/relearning.py` | KILLED |
| M21 | Re-learning warm-up length taken from the settings instead of the recorded length | `src/relearning.py` | KILLED |
| M22 | Buffer digest omits the observations | `src/gcsl.py` | **SURVIVED** (full suite too) |
| M23 | Saved buffer and its digest are D0 alone (consistent with each other) | `src/training.py` | KILLED |
| M24 | One extra re-learning gradient step, hidden from the learner's counter | `src/relearning.py` | KILLED |

Of the 24 mutations, 14 are killed and 10 survive. Every mutation the brief named is killed:
- off-by-one in practice updates (M1);
- skipping the digest check in the re-learning path (M2);
- skipping the Adam reset (M4);
- a default evaluation config (M6);
- the pretrained check ignoring the commit (M10);
- the launcher ignoring dependencies (M13);
- a hidden extra update (M24).

The survivors are test gaps (SF5), not defects in the current code.

## 3. Verdict

**No blocking defect.** On the six standing checks:

1. **Re-learning budget: sound (established).** For every condition, both learners, all three ratios, both schedules, the learning-rate fallback, the validation warm-up rule and N = 0, the re-learner takes exactly the original's warm-up plus practice updates, counted independently. It builds no environment, and its Adam starts empty.
2. **Buffer identity: sound (established).** The saved buffer is exactly the buffer training sampled; a changed, swapped or reordered buffer is refused.
3. **Evaluation:** it can run with settings other than the run's own if `settings.json` has been edited (SF1), and `--quick` is allowed for tuning, capacity and precondition runs (SF4).
4. **Pretrained check:** sound in a normal launch. It can pass when there is no commit, when trees are dirty, and for fresh re-learners (MI1, MI2). It crashes instead of reporting on an unreadable file (SF3).
5. **Launcher:** it schedules dependencies correctly, never overwrites finished training runs, and runs one thread per process. But a resumed launch does not check that finished runs match the plan (SF2). A crash inside the launcher leaves running children orphaned and no manifest (SF3). Nothing stops two launches on the same output, and two evaluations to one file both succeed (SF3).
6. **Training runner behaviour: unchanged (established).**

The findings that matter before the pilot are SF1 to SF4. Each is a small change.

## 4. Findings

### Blocking

None.

### Should fix

**SF1. `evaluate_run.py` evaluates with whatever `settings.json` says, without checking it against the settings the run recorded in `summary.json`. Established.**
- `scripts/evaluate_run.py:276` calls `runs.load_run` (`src/runs.py:44-53`), which builds the settings from `settings.json` and never compares them with `summary["settings"]`. The re-learning control does compare them (`src/relearning.py:64`).
- Evidence (`tamper.py`): in a copy of a main-learner run, `settings.json` was edited four ways. Each was accepted, and the evaluation config and adapter followed the edit:
  - learning rate 0.0003;
  - learner `plain`;
  - k = 120;
  - r = 2.
- End to end, `evaluate_run.py --quick` on the `plain` edit finished and recorded an adaptation with γ = 1.0 for a run trained with γ = 0.962.
- The suite's own check cannot catch this: it compares the adapter against a config built from the same edited file.
- Why it matters: the spec requires A_k to use the run's own γ, learning rate, batch size, ratio and k. Editing k into `settings.json` after the pilot sets it is a plausible slip.
- Fix: in `load_run`, refuse a folder whose `summary["settings"]` differs from `asdict(settings)`, so evaluation, re-learning and the pretrained check all share the rule. Add a test that edits `settings.json` and expects `evaluate_run` to refuse it.

**SF2. Resuming a launch does not check that finished runs match the plan, so one launch can silently mix budgets, ratios and k. Established.**
- `scripts/launch.py:180-182` marks a job `done_earlier` whenever its output exists. The job's settings are written to `job_settings/` only when a job starts (line 205-207), and nothing compares an existing run's `settings.json` with the plan.
- Evidence:
  - First launch: demonstrations only and task-directed practice, N = 60, r = 1.
  - Second launch, same output folder, with N = 120, r = 2, k = 120 and frontier novelty added.
  - The two old runs were kept as `done_earlier` (N = 60, r = 1, k = 960) and frontier novelty ran at N = 120, r = 2, k = 120.
  - The pretrained check passed, because N, r and k are not warm-up settings. The launcher exited 0.
  - The manifest records only the second plan.
- Why it matters: budget parity across conditions in a seed is the core of the design, and resuming after a crash or pre-emption on a rented machine is the launcher's normal use.
- Fix:
  - For every `done_earlier` training or re-learning job, require the folder's `settings.json` (and `summary["settings"]`) to equal the job's planned values. Fail the job, not silently skip it.
  - For an existing evaluation output, require its recorded config and flags (`quick_smoke_only`, `adaptation_run`) to match the plan.
  - Test with the scenario above.

**SF3. Launcher failure handling: a crash in the launcher orphans running jobs and writes no manifest; two launches on one output are not prevented; the pretrained check crashes instead of reporting. Established, except where marked.**
- A missing, corrupted or wrongly keyed `pretrained.pt`, or a truncated `summary.json`, makes `check_pretrained` raise. It only catches `ValueError` (`src/runs.py:117-121`); `torch.load` raised `UnpicklingError` and `FileNotFoundError`, and `json` raised `JSONDecodeError` (`check_pre.py`).
- Inside the launcher (`scripts/launch.py:199-200`, and line 255 at the end), that exception ends `run_jobs`. In `launch_crash.py`:
  - the launcher raised while a job was running;
  - the child kept running and later wrote its output;
  - no `launch_manifest.json` was written.
- The same applies to Ctrl-C or any other exception in the loop (reasonable concern).
- Reasonable concern: a later `--retry-incomplete` renames the folder of a still-running orphan. The orphan writes its remaining files by path (`out / "summary.json"`, `scripts/run_training.py:92-95`), so they land in the new run's folder.
- No lock against two launches on one output:
  - Two launchers started together on one plan both started the same training job. The second was refused only because the first had already written into the folder.
  - The check in `run_training.py:79-81` (make the folder, then look whether it is empty) is not atomic, so a closer race would let both write (reasonable concern).
  - Both launchers read-modify-write `launch_manifest.json`, so one launch record can be lost (reasonable concern).
- Two concurrent `evaluate_run.py` calls on one run and one output both finished with exit 0. The second replaced the first's file. `out.exists()` is checked at start (line 278), and `os.replace` (line 303) overwrites unconditionally, against the script's "never overwritten" rule.
- Fix:
  - Wrap `run_jobs` in `try/finally` that terminates running children and always writes the manifest.
  - In `check_pretrained`, catch any exception per run and report it as a problem.
  - Take an exclusive lock file in OUT (`os.open(..., O_CREAT | O_EXCL)`) for the whole launch.
  - In `evaluate_run.py`, publish with `os.link(partial, out)`, which fails if `out` exists, and then remove the partial file.
  - In the run scripts, claim the folder by creating a marker file with `O_EXCL`.

**SF4. `--quick` is refused only for pilot and main runs, and the launcher does not check evaluation flags against the purpose. Established.**
- `scripts/evaluate_run.py:255` allows `--quick` for tuning, capacity and precondition runs.
- `--quick` changes the measures, not just their precision:
  - 10 sampled rollouts with a minimum of 2 successes, instead of 100 and 20;
  - k = 60.
- The precondition check's Rep under `--quick` therefore applies a different rule. The output marks `quick_smoke_only`, but nothing stops it being used.
- Separately, a pilot plan with `"evaluate": {"quick": true}` passes `launch.py --dry-run` (`build_jobs` checks the training settings only). Its evaluation jobs would fail only after the pilot's training had finished.
- Fix:
  - Allow `--quick` only for purpose `smoke`.
  - In `build_jobs`, call `evaluate_run.build(settings, quick, skip_adaptation)` for every run, so the same refusal happens before anything starts.

**SF5. Test gaps, from the surviving mutations. Established.** The current code is correct in each case; a future regression would go unnoticed.
- M14, M15: no test makes the launcher's pre-evaluation pretrained check fail. Fix: a launcher test with a forged pretrained fingerprint in one run of the group, expecting the pretrained evaluation to be `failed` and its target absent.
- M8: `evaluate_run.py` records `policy_digest` from `summary.json` (line 293), not from the policy it evaluated, so evaluating the wrong weights goes unseen. The suite's `parameter_digest` uses a different formula, so it cannot be compared directly. Fix: record `policy.parameter_digest()` of the policy passed to the suite, assert that it equals `summary[f"{which}_digest"]`, and test `--which pretrained`.
- M22: the tamper test changes only an action bin, so a digest that skipped the observations would pass. Fix: tamper each array (observations, action bins, goals, sources) in turn and expect refusal.
- M5: the empty-state check runs right after the new learner is built (`src/relearning.py:123`), before practice. Fix: after practice, also assert that Adam's step count equals the practice updates, and test a carried-over state. The same gap was M2 of 4 Nov for the training runner.
- M12, M3: no test of an `init same` re-learner that fails to reproduce its source in the pretrained check, and none of `load_buffer` refusing a tampered file on disk. Fix: one test each.
- M7: every evaluation test uses learning rate 0.001. Fix: one test with 0.0003.
- M18, M19: thread pinning is untested. Fix: assert `torch_threads` is 1 and 1 in the run record of a launched run.
- The no-environment test (`tests/test_relearning.py:54-55`) patches `envs.pointmaze.make_env` and `collection.make_env`, but not `demonstrations.make_env`. So a re-learner that rebuilt D0 or ran the validation rule would build environments unseen. The auditor's check at `gymnasium.make` found none. Fix: patch `gymnasium.make`.

### Minor

**MI1. The pretrained check identifies code only by commit: too loose for dirty or untracked runs, too strict across harmless commits. Established.**
- Runs whose records say commit `None` (not a git checkout, or `run_record.json` missing) pass with no problem and no warning, since the set `{None}` has one element (`src/runs.py:145-149`).
- Two runs from the same commit, both dirty, with different hashes for `src/collection.py`, pass with only a warning.
- Conversely, resuming after a commit that touches only notes fails the check.
- Fix: compare the recorded `source_sha256` of `src/` and the run scripts across the group, and use the commit only as a label. Treat a missing commit as a problem for pilot and main runs.

**MI2. Fresh re-learners are not checked for sharing a pretrained policy within a seed. Established.**
- The spec's paired-seed rule applies to the re-learning arm too. The code does pair them (one fingerprint per seed, section 2), but `check_pretrained` does not group them (`src/runs.py:127`).
- Two seed-0 fresh re-learners forged to start from different policies passed.
- Fix: group re-learning runs by seed, maze, learner, purpose and `init`, and require one pretrained fingerprint.

**MI3. Runs in one seed with different purposes are never compared, and a plan without a purpose runs as `smoke`. Established.**
- A smoke run and a tuning run in seed 0 with different warm-ups (different learning rates, different pretrained fingerprints) passed `check_pretrained`.
- Inside one launch this cannot happen: the purpose is shared. But `scripts/check_pretrained.py ROOT` over a mixed folder will not flag it.
- A plan that forgets `"purpose": "pilot"` runs full-budget runs as `smoke`, skipping the pilot's setting checks, its k rule and its refusal of `--quick`.
- Fix: require `purpose` in a plan, and list cross-purpose groups in a seed as warnings.

**MI4. `summary.json`, which marks a run as finished, is written non-atomically. Reasonable concern.** `scripts/run_training.py:95` and `scripts/run_relearning.py:238` use `write_text`. A process killed mid-write leaves a truncated file that counts as finished; the launcher then skips the job, and later loads crash (see SF3). Fix: write to a temporary name and `os.replace`, as `evaluate_run.py` already does.

**MI5. Re-learning does not compare the current code with the source run's code. Reasonable concern.** `relearning.json` records the source commit (`scripts/run_relearning.py:224`), but nothing refuses or warns when it differs from the re-learner's own commit, or when either tree is dirty. A learner change between the original and its re-learner would confound H4 (the hypothesis tested by the re-learning control). Fix: refuse for pilot and main runs unless the source hashes of `src/` match; warn otherwise.

**MI6. A re-learner of a validation-rule original records the original's warm-up key (`warmup_rule: "validation"`), though its warm-up ran with the fixed rule. Established.** See `src/relearning.py:162`; the warm-up log itself says `fixed`. Harmless today, because re-learners are not grouped by key, but misleading in analysis. Fix: record the rule actually used next to the source's key.

**MI7. `evaluate_run.py` does not record the torch thread counts; the two run scripts do.** Fix: add `torch_threads` to its record.

### Notes (not code defects)

- **N1.** The training runner's behaviour is unchanged apart from saving the buffer (section 2). The extra work at the end of a run (one copy of the buffer arrays, one digest, a file of under 5 MB at the pilot budget and under 19 MB at the cap, before compression) does not touch any random stream.
- **N2.** Under the interleaved schedule the original run never trains on its last episode (4 Nov, N1); the saved final buffer includes it, so the re-learner does. This follows "final buffer"; worth one line in the write-up.
- **N3.** The re-learner samples the whole final buffer from its first practice update, so D0's share is fixed at its final value. Documented in `src/relearning.py` as the point of the control. The fresh initial weights, the fresh warm-up stream and the practice stream depend only on the seed, so re-learners are paired across conditions (verified).
- **N4.** `summary.json` is the root of trust. Replacing `final.pt` and its fingerprint in `summary.json` together is accepted (established). For a single-user project this is acceptable. If wanted, the launcher could record each finished run's `summary.json` hash in the manifest.
- **N5.** The commit check deliberately makes the pre-evaluation pretrained check fail when a launch is resumed after any commit. On a rented machine, commit fixes between launches, not during one.
- **N6.** Evaluation leaves the evaluated weights untouched: `final.pt` still matched its fingerprint after two evaluations, and the suite evaluates a deep copy behind its own fingerprint guard. Two identical evaluations gave identical numbers.

## 5. What was not checked

- Runs at pilot or main size: nothing beyond 120 practice steps, 60 warm-up updates or `--quick` evaluation. Memory and time for saving and loading a full-size buffer were estimated, not measured.
- A full evaluation (100 rollouts, a real k) through `evaluate_run.py`.
- The race in which two training processes both pass the empty-folder check (SF3) was reasoned from the code, not reproduced.
- Signals (Ctrl-C, a dropped SSH session) sent to the launcher.
- The 4 Nov decision audit's text of decision 13 (a Project document, not in the repository); the brief's statement of it was used.
- Determinism across machines or torch versions, and macOS file-system behaviour.
- Mutations were run against the three affected test files; only M5 and M22 were also run against the full suite.

## Response (24 Nov, implementing agent)

All findings were acted on in the commit after this report. 202 tests pass (190 before). The training runner still gives identical pretrained and final fingerprints for the same settings as at `c91355e` (rechecked for frontier novelty, task-directed practice and demonstrations only).

| Finding | What was done |
| --- | --- |
| SF1 | `runs.load_run` refuses a folder whose `settings.json` differs from `summary["settings"]`, so evaluation, re-learning, the pretrained check and the launcher's resume check share the rule. Test: four edits (learning rate, learner, k, ratio) all refused. |
| SF2 | Every job whose output already exists is checked against the plan before it counts as done: runs by their full settings (and re-learners by `init`), evaluations by their recorded config, `quick_smoke_only`, `adaptation_run` and `which`. A mismatch fails the job. Test: resuming with r = 2, and with a different evaluation protocol, both fail. |
| SF3 | `check_pretrained` reports any per-run exception as a problem. `run_jobs` terminates running children on any exception or Ctrl-C, and the launcher always writes its manifest. One launcher per output folder (`.launch.lock`, created with O_EXCL). `evaluate_run.py` publishes with `os.link`, which refuses an existing file. The run scripts claim their folder with an O_EXCL marker (`.claimed`) before writing. Tests for each. |
| SF4 | `--quick` is allowed only for purpose `smoke`. The launcher calls `evaluate_run.build` for every run and pretrained evaluation before anything starts. Test: pilot and precondition plans with quick evaluation fail in `build_jobs`. |
| SF5 | Tests added for M3, M7, M8, M12, M14 and M15, M18 and M19, and M22 (every buffer array tampered in turn); the no-environment test also blocks `gymnasium.make` and `demonstrations.make_env`. M8: the evaluation now records and asserts the fingerprint of the policy it hands to the suite. M5: `assert_fresh_adam` checks, right after the first practice update, that Adam's running averages equal (1 − β₁)·g and (1 − β₂)·g² (β₁, β₂: Adam's averaging constants; g: that update's gradient), which holds only from an empty state. It runs in the training runner and the re-learner, closing the 4 Nov M2 gap too. |
| MI1 | Code is compared by the recorded source hashes of `src/` and the run scripts; the commit is a label. A pilot or main run with no source hashes is a problem; otherwise a warning. |
| MI2 | Re-learners sharing seed, maze, learner, purpose and `init` must share one pretrained fingerprint. |
| MI3 | A plan must name a purpose. Runs of different purposes in one seed that started from different policies are listed as warnings. |
| MI4 | `summary.json` is written atomically (`provenance.write_text_atomic`). |
| MI5 | The re-learning script compares its code's source hashes with the source run's: refused for pilot and main if they differ, a warning otherwise; recorded as `same_code_as_source`. |
| MI6 | The re-learning summary records `warmup_rule_used: "fixed"` and the warm-up key actually used. |
| MI7 | `evaluate_run.py` records torch's thread counts. |
| N2 | Noted for the write-up: under the interleaved schedule the re-learner trains on the original's last episode, which the original never trained on. |
| N4 | Left as is: `summary.json` is the root of trust in a single-user project. |
