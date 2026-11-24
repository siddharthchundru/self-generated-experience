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
        moved = self.copy_run(self.frontier, "other_commit")
        record = json.loads((moved / "run_record.json").read_text())
        record["git_commit"] = "f" * 40
        (moved / "run_record.json").write_text(json.dumps(record))
        report = runs.check_pretrained(folders=[self.task, moved])
        self.assertTrue(any("different commits" in p for p in report["problems"]))

    def test_unfinished_run_is_listed(self):
        crashed = self.tmp / "crashed"
        crashed.mkdir()
        (crashed / "settings.json").write_text(json.dumps(asdict(TrainingSettings(condition=TASK_DIRECTED))))
        report = runs.check_pretrained(folders=[self.task, crashed])
        self.assertEqual(report["unfinished"], [str(crashed)])
        self.assertTrue(report["passed"])


class TestLauncher(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def plan(self, **overrides):
        plan = {"out": str(self.tmp / "out"), "settings": dict(TINY), "seeds": [0],
                "conditions": ["demonstrations_only", TASK_DIRECTED], "evaluate": {"quick": True}}
        plan.update(overrides)
        path = self.tmp / "plan.json"
        path.write_text(json.dumps(plan))
        return path

    def test_plan_refusals(self):
        import launch
        for bad in ({"settings": {"seed": 3}}, {"plain_conditions": ["expert_top_up"]}, {"seeds": []},
                    {"conditions": ["nothing"]}, {"seeds": [1, 1]}, {"relearning": {"init": "warm"}},
                    {"typo": 1}):
            with self.assertRaises(SystemExit, msg=bad):
                launch.load_plan(self.plan(**bad))

    def test_settings_are_checked_before_anything_runs(self):
        import launch
        plan = launch.load_plan(self.plan(settings={"purpose": "pilot"}))  # k placeholder
        with self.assertRaisesRegex(ValueError, "placeholder"):
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
