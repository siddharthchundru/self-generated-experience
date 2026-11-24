"""Check the Stage 2 run scripts: evaluation of a run folder, the pretrained check, the launcher.

What these tests protect (tiny settings and --quick evaluation: they check the
code, not results):
- evaluate_run.py rebuilds the policy from the run folder and refuses weights
  that do not match the run's fingerprint; it takes its adaptation settings from
  the run, never overwrites an evaluation, and refuses --quick and the k
  placeholder for pilot and main runs;
- the pretrained check passes runs that share a warm-up, and fails a run whose
  recorded or saved pretrained policy differs, runs in one seed with different
  warm-up settings, and runs from different commits;
- the launcher refuses malformed plans, checks every run's settings before
  starting, schedules dependants after their jobs, skips dependants of a
  failure, resumes a finished plan without rerunning anything, and handles a
  crashed run folder only with --retry-incomplete.
"""

from dataclasses import asdict
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from collection import TASK_DIRECTED  # noqa: E402
from episode import EPISODE_STEPS  # noqa: E402
import runs  # noqa: E402
from training import TrainingSettings  # noqa: E402

T = EPISODE_STEPS
TINY = {"warmup_updates": 30, "demos_per_pair": 2, "budget_steps": T, "batch_size": 32}


def train(folder, condition=TASK_DIRECTED, **overrides):
    import run_training
    settings = Path(folder).parent / f"{Path(folder).name}.json"
    settings.write_text(json.dumps({**TINY, **overrides}))
    run_training.main(["--condition", condition, "--settings", str(settings), "--out", str(folder)])
    return Path(folder)


class TestEvaluateAndCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.task = train(cls.tmp / "task")
        cls.frontier = train(cls.tmp / "frontier", "frontier_novelty")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def copy_run(self, source, name):
        target = self.tmp / name
        shutil.copytree(source, target)
        return target

    def test_evaluation_uses_the_run_and_never_overwrites(self):
        import evaluate_run
        out = self.tmp / "eval_task.json"
        evaluate_run.main(["--run", str(self.task), "--quick", "--out", str(out)])
        result = json.loads(out.read_text())
        self.assertEqual(result["run"]["policy_digest"],
                         json.loads((self.task / "summary.json").read_text())["final_digest"])
        config = result["evaluation"]["config"]
        settings = json.loads((self.task / "settings.json").read_text())
        self.assertEqual(config["adaptation_learning_rate"], settings["learning_rate"])
        self.assertEqual(config["adaptation_batch_size"], settings["batch_size"])
        self.assertEqual(config["adaptation_gamma"], 0.962)
        runs_ = result["report"]["adaptation_runs"]
        self.assertEqual(runs_["P1"]["gradient_steps"], T)  # quick k = one episode, ratio 1
        self.assertTrue(result["evaluation"]["quick_smoke_only"])
        with self.assertRaisesRegex(SystemExit, "never overwritten"):
            evaluate_run.main(["--run", str(self.task), "--quick", "--out", str(out)])

    def test_pretrained_without_adaptation(self):
        import evaluate_run
        out = self.tmp / "eval_pre.json"
        evaluate_run.main(["--run", str(self.task), "--which", "pretrained", "--skip-adaptation", "--quick",
                           "--out", str(out)])
        result = json.loads(out.read_text())
        self.assertFalse(result["evaluation"]["adaptation_run"])
        self.assertIsNone(result["report"]["aggregate"]["A_k"])
        self.assertEqual(result["run"]["policy_digest"],
                         json.loads((self.task / "summary.json").read_text())["pretrained_digest"])

    def test_swapped_weights_are_refused(self):
        import evaluate_run
        swapped = self.copy_run(self.task, "swapped")
        shutil.copy(self.frontier / "final.pt", swapped / "final.pt")
        with self.assertRaisesRegex(ValueError, "does not match the fingerprint"):
            evaluate_run.main(["--run", str(swapped), "--quick"])

    def test_quick_and_placeholder_refused_for_pilot(self):
        import evaluate_run
        pilot = TrainingSettings(condition=TASK_DIRECTED, purpose="pilot", adaptation_steps=24_960)
        with self.assertRaisesRegex(SystemExit, "smoke"):
            evaluate_run.build(pilot, quick=True)
        config, adapter = evaluate_run.build(pilot)
        self.assertEqual(config.adaptation_steps, 24_960)
        self.assertEqual(adapter.learning_rate, pilot.learning_rate)
        placeholder = mock.Mock(purpose="main", adaptation_steps=960)
        with self.assertRaisesRegex(SystemExit, "placeholder"):
            evaluate_run.build(placeholder)

    def test_check_passes_shared_warmup(self):
        report = runs.check_pretrained(folders=[self.task, self.frontier])
        self.assertTrue(report["passed"], report["problems"])
        self.assertEqual(len(report["warmup_groups"]), 1)
        self.assertEqual(report["warmup_groups"][0]["runs"], 2)

    def test_check_fails_a_different_pretrained_policy(self):
        forged = self.copy_run(self.frontier, "forged")
        summary = json.loads((forged / "summary.json").read_text())
        summary["pretrained_digest"] = "0" * 64
        (forged / "summary.json").write_text(json.dumps(summary))
        report = runs.check_pretrained(folders=[self.task, forged], verify_files=False)
        self.assertFalse(report["passed"])
        self.assertTrue(any("different pretrained" in p for p in report["problems"]))
        swapped = self.copy_run(self.frontier, "swapped_pre")
        shutil.copy(self.frontier / "final.pt", swapped / "pretrained.pt")
        report = runs.check_pretrained(folders=[self.task, swapped])
        self.assertFalse(report["passed"])

    def test_check_fails_mixed_warmup_and_commits(self):
        other = train(self.tmp / "other_lr", learning_rate=3e-4)
        report = runs.check_pretrained(folders=[self.task, other])
        self.assertFalse(report["passed"])
        self.assertTrue(any("differ in warm-up settings" in p for p in report["problems"]))
        # The commit is only a label; the code is compared by source hashes (MI1).
        relabelled = self.copy_run(self.frontier, "other_commit")
        record = json.loads((relabelled / "run_record.json").read_text())
        record["git_commit"] = "f" * 40
        (relabelled / "run_record.json").write_text(json.dumps(record))
        self.assertTrue(runs.check_pretrained(folders=[self.task, relabelled])["passed"])
        changed = self.copy_run(self.frontier, "other_code")
        record = json.loads((changed / "run_record.json").read_text())
        record["source_sha256"]["src/collection.py"] = "0" * 64
        (changed / "run_record.json").write_text(json.dumps(record))
        report = runs.check_pretrained(folders=[self.task, changed])
        self.assertTrue(any("different code" in p for p in report["problems"]))

    def test_unfinished_run_is_listed(self):
        crashed = self.tmp / "crashed"
        crashed.mkdir()
        (crashed / "settings.json").write_text(json.dumps(asdict(TrainingSettings(condition=TASK_DIRECTED))))
        report = runs.check_pretrained(folders=[self.task, crashed])
        self.assertEqual(report["unfinished"], [str(crashed)])
        self.assertTrue(report["passed"])

    def test_edited_settings_are_refused(self):  # SF1
        import evaluate_run
        edited = self.copy_run(self.task, "edited")
        settings = json.loads((edited / "settings.json").read_text())
        for change in ({"learning_rate": 3e-4}, {"learner": "plain"}, {"adaptation_steps": 120},
                       {"updates_per_env_step": 2.0}):
            (edited / "settings.json").write_text(json.dumps({**settings, **change}))
            with self.assertRaisesRegex(ValueError, "settings.json differs", msg=change):
                evaluate_run.main(["--run", str(edited), "--quick", "--out", str(self.tmp / "never.json")])
        self.assertFalse((self.tmp / "never.json").exists())

    def test_evaluation_follows_another_learning_rate(self):  # M7
        import evaluate_run
        run = train(self.tmp / "lr3", learning_rate=3e-4)
        out = self.tmp / "eval_lr3.json"
        evaluate_run.main(["--run", str(run), "--quick", "--out", str(out)])
        result = json.loads(out.read_text())
        self.assertEqual(result["evaluation"]["config"]["adaptation_learning_rate"], 3e-4)
        self.assertEqual(result["report"]["adaptation_runs"]["P1"]["learner_settings"]["learning_rate"], 3e-4)
        self.assertEqual(result["torch_threads"]["intra_op"], torch.get_num_threads())  # MI7: recorded

    def test_concurrent_evaluation_never_replaces_a_file(self):  # SF3
        import evaluate_run
        out = self.tmp / "race.json"
        original = evaluate_run.evaluate_policy

        def racing(*args, **kwargs):
            out.write_text("the other evaluation")
            return original(*args, **kwargs)

        with mock.patch.object(evaluate_run, "evaluate_policy", racing):
            with self.assertRaisesRegex(SystemExit, "meanwhile"):
                evaluate_run.main(["--run", str(self.task), "--quick", "--out", str(out)])
        self.assertEqual(out.read_text(), "the other evaluation")
        self.assertEqual(list(self.tmp.glob("race.json.*partial")), [])

    def test_unreadable_files_are_reported_not_raised(self):  # SF3
        broken = self.copy_run(self.frontier, "broken")
        (broken / "pretrained.pt").write_bytes(b"not a checkpoint")
        truncated = self.copy_run(self.frontier, "truncated")
        (truncated / "summary.json").write_text('{"settings": ')
        report = runs.check_pretrained(folders=[self.task, broken, truncated])
        self.assertFalse(report["passed"])
        self.assertEqual(sum("could not be checked" in p for p in report["problems"]), 2)

    def test_load_buffer_refuses_a_tampered_file(self):  # M3
        tampered = self.copy_run(self.task, "tampered_buffer")
        with np.load(tampered / "final_buffer.npz") as data:
            arrays = {k: data[k].copy() for k in data.files}
        arrays["goals"][0, 0] += 0.5
        np.savez_compressed(tampered / "final_buffer.npz", **arrays)
        with self.assertRaisesRegex(ValueError, "does not match the digest"):
            runs.load_buffer(tampered)

    def test_relearners_are_checked(self):  # M12, MI2
        import run_relearning
        same = self.tmp / "re_same"
        run_relearning.main(["--run", str(self.task), "--out", str(same), "--init", "same"])
        fresh_a, fresh_b = self.tmp / "re_fresh_a", self.tmp / "re_fresh_b"
        run_relearning.main(["--run", str(self.task), "--out", str(fresh_a)])
        run_relearning.main(["--run", str(self.frontier), "--out", str(fresh_b)])
        self.assertTrue(runs.check_pretrained(folders=[self.task, same, fresh_a, fresh_b])["passed"])
        summary = json.loads((same / "summary.json").read_text())
        self.assertEqual(summary["warmup_rule_used"], "fixed")  # MI6
        self.assertTrue(summary["source"]["same_code"])  # MI5
        record = json.loads((fresh_a / "run_record.json").read_text())
        self.assertEqual(record["torch_threads"]["intra_op"], torch.get_num_threads())
        forged = self.copy_run(same, "re_same_forged")
        summary["pretrained_digest"] = "0" * 64
        (forged / "summary.json").write_text(json.dumps(summary))
        report = runs.check_pretrained(folders=[forged], verify_files=False)
        self.assertTrue(any("did not reproduce" in p for p in report["problems"]))
        forged_fresh = self.copy_run(fresh_b, "re_fresh_forged")
        fresh_summary = json.loads((forged_fresh / "summary.json").read_text())
        fresh_summary["pretrained_digest"] = "1" * 64
        (forged_fresh / "summary.json").write_text(json.dumps(fresh_summary))
        report = runs.check_pretrained(folders=[fresh_a, forged_fresh], verify_files=False)
        self.assertTrue(any("re-learners with init 'fresh'" in p for p in report["problems"]))

    def test_relearning_records_a_code_change(self):  # MI5
        import run_relearning
        source = self.copy_run(self.task, "other_code_source")
        record = json.loads((source / "run_record.json").read_text())
        record["source_sha256"]["src/gcsl.py"] = "0" * 64
        (source / "run_record.json").write_text(json.dumps(record))
        out = self.tmp / "re_other_code"
        run_relearning.main(["--run", str(source), "--out", str(out)])  # smoke: a warning, not a refusal
        self.assertFalse(json.loads((out / "relearning.json").read_text())["same_code_as_source"])

    def test_launcher_precheck_blocks_the_pretrained_evaluation(self):  # M14, M15
        import launch
        forged = self.copy_run(self.frontier, "forged_for_launcher")
        summary = json.loads((forged / "summary.json").read_text())
        summary["pretrained_digest"] = "0" * 64
        (forged / "summary.json").write_text(json.dumps(summary))
        target = self.tmp / "pre_eval.json"
        job = launch.Job("evaluate pretrained", "evaluate_pretrained",
                         [sys.executable, "-c", f"open(r'{target}', 'w')"], target,
                         group=[self.task, forged])  # the forged run is second in the group
        launch.run_jobs(self.tmp / "launch_out", [job], workers=1, echo=lambda *_: None)
        self.assertEqual(job.status, "failed")
        self.assertFalse(target.exists())


class TestLauncher(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def plan(self, **overrides):
        plan = {"out": str(self.tmp / "out"), "settings": {**TINY, "purpose": "smoke"}, "seeds": [0],
                "conditions": ["demonstrations_only", TASK_DIRECTED], "evaluate": {"quick": True}}
        plan.update(overrides)
        path = self.tmp / "plan.json"
        path.write_text(json.dumps(plan))
        return path

    def test_plan_refusals(self):
        import launch
        for bad in ({"settings": {"purpose": "smoke", "seed": 3}}, {"settings": {}},  # MI3: purpose required
                    {"plain_conditions": ["expert_top_up"]}, {"seeds": []},
                    {"conditions": ["nothing"]}, {"seeds": [1, 1]}, {"relearning": {"init": "warm"}},
                    {"typo": 1}):
            with self.assertRaises(SystemExit, msg=bad):
                launch.load_plan(self.plan(**bad))

    def test_settings_are_checked_before_anything_runs(self):
        import launch
        plan = launch.load_plan(self.plan(settings={"purpose": "pilot"}))  # k placeholder
        with self.assertRaisesRegex(ValueError, "placeholder"):
            launch.build_jobs(plan)
        for purpose in ("pilot", "precondition"):  # SF4: quick evaluation refused before anything runs
            settings = {"purpose": purpose, "adaptation_steps": 24_960} if purpose == "pilot" else {"purpose": purpose}
            plan = launch.load_plan(self.plan(settings=settings, evaluate={"quick": True}))
            with self.assertRaisesRegex(SystemExit, "smoke runs only"):
                launch.build_jobs(plan)

    def test_job_graph(self):
        import launch
        plan = launch.load_plan(self.plan(seeds=[0, 1], plain_conditions=[TASK_DIRECTED],
                                          relearning={"enabled": True}))
        _, jobs = launch.build_jobs(plan)
        kinds = [j.kind for j in jobs]
        # per seed: 3 training runs and their evaluations, 2 re-learners (main learner only)
        # and their evaluations, 2 pretrained evaluations (one per learner)
        self.assertEqual(kinds.count("train"), 6)
        self.assertEqual(kinds.count("relearn"), 4)
        self.assertEqual(kinds.count("evaluate"), 10)
        self.assertEqual(kinds.count("evaluate_pretrained"), 4)
        names = {j.name for j in jobs}
        for job in jobs:
            self.assertTrue(set(job.deps) <= names, job.name)
        pre = next(j for j in jobs if j.name == "evaluate pretrained weighted/seed0")
        self.assertEqual(sorted(pre.deps), ["train weighted/demonstrations_only/seed0",
                                            f"train weighted/{TASK_DIRECTED}/seed0"])

    def test_scheduler_skips_dependants_of_failures(self):
        import launch
        py = sys.executable
        out = self.tmp / "fake"
        ok = launch.Job("a", "train", [py, "-c", f"open(r'{out / 'a'}', 'w')"], out / "a")
        bad = launch.Job("b", "train", [py, "-c", "raise SystemExit(1)"], out / "b")
        after_ok = launch.Job("c", "evaluate", [py, "-c", f"open(r'{out / 'c'}', 'w')"], out / "c", deps=["a"])
        after_bad = launch.Job("d", "evaluate", [py, "-c", "pass"], out / "d", deps=["b"])
        out.mkdir()
        launch.run_jobs(out, [ok, bad, after_ok, after_bad], workers=2, echo=lambda *_: None)
        self.assertEqual([j.status for j in (ok, bad, after_ok, after_bad)], ["done", "failed", "done", "skipped"])

    def test_end_to_end_resume_and_crash(self):
        import launch
        plan = self.plan(relearning={"enabled": True}, conditions=[TASK_DIRECTED])
        self.assertEqual(launch.main([str(plan), "--workers", "3"]), 0)
        out = self.tmp / "out"
        manifest = json.loads((out / "launch_manifest.json").read_text())
        statuses = {j["name"]: j["status"] for j in manifest["launches"][-1]["jobs"]}
        self.assertEqual(set(statuses.values()), {"done"}, statuses)
        self.assertTrue(manifest["launches"][-1]["pretrained_check"]["passed"])
        self.assertEqual(manifest["launches"][-1]["pretrained_check"]["relearning_runs"], 1)
        self.assertTrue((out / "pretrained" / "weighted" / "seed0.json").exists())
        # Resume: nothing runs again.
        self.assertEqual(launch.main([str(plan), "--workers", "3"]), 0)
        statuses = {j["status"] for j in json.loads((out / "launch_manifest.json").read_text())["launches"][-1]["jobs"]}
        self.assertEqual(statuses, {"done_earlier"})
        # A crashed run folder fails until --retry-incomplete.
        run = out / "runs" / "weighted" / TASK_DIRECTED / "seed0"
        shutil.rmtree(run)
        run.mkdir(parents=True)
        (run / "settings.json").write_text("{}")
        self.assertEqual(launch.main([str(plan), "--workers", "3"]), 1)
        self.assertEqual(launch.main([str(plan), "--workers", "3", "--retry-incomplete"]), 0)
        self.assertEqual(len(list(run.parent.glob("seed0.incomplete-*"))), 1)
        # One thread per process, in training and re-learning runs (M18, M19).
        for folder in (run, out / "relearning" / TASK_DIRECTED / "seed0"):
            threads = json.loads((folder / "run_record.json").read_text())["torch_threads"]
            self.assertEqual(threads, {"intra_op": 1, "inter_op": 1}, folder)
        # Resuming with another budget, ratio or k fails instead of mixing runs (SF2).
        changed = self.plan(relearning={"enabled": True}, conditions=[TASK_DIRECTED],
                            settings={**TINY, "purpose": "smoke", "updates_per_env_step": 2.0})
        self.assertEqual(launch.main([str(changed), "--workers", "3"]), 1)
        jobs = json.loads((out / "launch_manifest.json").read_text())["launches"][-1]["jobs"]
        train_job = next(j for j in jobs if j["name"] == f"train weighted/{TASK_DIRECTED}/seed0")
        self.assertEqual(train_job["status"], "failed")
        self.assertIn("other settings", train_job["log"])
        quick_off = self.plan(relearning={"enabled": True}, conditions=[TASK_DIRECTED],
                              evaluate={"quick": True, "skip_adaptation": True})
        self.assertEqual(launch.main([str(quick_off), "--workers", "3"]), 1)
        jobs = json.loads((out / "launch_manifest.json").read_text())["launches"][-1]["jobs"]
        eval_job = next(j for j in jobs if j["name"] == f"evaluate weighted/{TASK_DIRECTED}/seed0")
        self.assertIn("another protocol", eval_job["log"])

    def test_one_thread_environment(self):  # M18
        import launch
        env = launch._environment()
        self.assertEqual({env[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}, {"1"})

    def test_a_stopped_launcher_terminates_its_jobs(self):  # SF3
        import launch
        job = launch.Job("sleep", "train", [sys.executable, "-c", "import time; time.sleep(60)"],
                         self.tmp / "never")

        def echo(line):
            if line.startswith("started"):
                raise KeyboardInterrupt

        out = self.tmp / "stopped"
        with self.assertRaises(KeyboardInterrupt):
            launch.run_jobs(out, [job], workers=1, echo=echo)
        self.assertEqual(job.status, "failed")
        self.assertIn("terminated", job.log)

    def test_one_launch_per_output(self):  # SF3
        import launch
        out = self.tmp / "locked"
        with launch.LaunchLock(out):
            with self.assertRaisesRegex(SystemExit, "another launch"):
                with launch.LaunchLock(out):
                    pass
        with launch.LaunchLock(out):  # released
            pass

    def test_refuses_results_from_a_dirty_tree(self):
        import launch
        import provenance
        plan = self.plan(out=str(ROOT / "results" / "launcher-test-never-written"))
        dirty = {"git_commit": "abc", "working_tree_dirty": True}
        with mock.patch.object(provenance, "git_state", return_value=dirty), \
                mock.patch.object(launch, "guard_results_write",
                                  lambda *a: provenance.guard_results_write(*a)):
            with self.assertRaisesRegex(SystemExit, "refusing to write"):
                launch.main([str(plan), "--dry-run"])
        self.assertFalse((ROOT / "results" / "launcher-test-never-written").exists())


if __name__ == "__main__":
    unittest.main()
