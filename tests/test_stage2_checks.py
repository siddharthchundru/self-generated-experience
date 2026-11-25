"""Check the capacity check, the tuning driver and the precondition gate.

What these tests protect (tiny settings: they check the code, not results):
- checkpoints: keeping early weights changes nothing else in a run; each kept
  snapshot is fingerprinted, loadable and evaluable, and a resumed launch whose
  runs kept other checkpoints fails;
- the launcher's tuning grid: one cell per combination, each with its own
  folders and pretrained evaluation; the pretrained check treats cells as
  separate batches for tuning only;
- the tuning selector: the objective (mean greedy competence of task-directed
  practice and frontier novelty), the tie rule (smaller ratio, then smaller
  learning rate), the early-drop flag, and its refusals;
- the precondition gate: both definitions, undefined Rep counted as 0, both
  failure rules, and its refusals;
- the capacity check: data on every ordered pair, the same evaluation pairs for
  every learner, the one permitted doubling, and the script's outputs.
"""

from dataclasses import asdict
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from collection import FRONTIER, TASK_DIRECTED  # noqa: E402
from episode import EPISODE_STEPS  # noqa: E402
import runs  # noqa: E402
from training import TrainingSettings, run_training  # noqa: E402

T = EPISODE_STEPS
TINY = {"warmup_updates": 30, "demos_per_pair": 2, "budget_steps": 3 * T, "batch_size": 32}


def fake_evaluation(path, settings, which, s_in=0.5, reps=None, quick=False, rollouts=100):
    """An evaluation file shaped like evaluate_run.py's, for the readers under test."""
    pairs = [{"row": r, "is_primary_pair": True, "S_in": s_in,
              "sampled": {"Rep": rep, "successes": 0 if rep is None else 50}} for r, rep in (reps or {}).items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "run": {"which": which, "settings": settings},
        "evaluation": {"quick_smoke_only": quick, "adaptation_run": False, "config": {"sampled_rollouts": rollouts}},
        "report": {"aggregate": {"S_in": s_in}, "pairs": pairs}}))


class TestCheckpoints(unittest.TestCase):
    def test_checkpoints_change_nothing_else(self):
        settings = TrainingSettings(condition=TASK_DIRECTED, **TINY)
        plain = run_training(settings)
        kept = run_training(settings, checkpoint_episodes=[1, 3])
        self.assertEqual(plain["summary"]["final_digest"], kept["summary"]["final_digest"])
        self.assertEqual(sorted(kept["checkpoint_states"]), [1, 3])
        self.assertEqual(set(kept["summary"]["checkpoint_digests"]), {"1", "3"})
        self.assertEqual(kept["summary"]["checkpoint_digests"]["3"], kept["summary"]["final_digest"])
        self.assertNotIn("checkpoint_digests", plain["summary"])
        with self.assertRaises(ValueError):
            run_training(settings, checkpoint_episodes=[4])

    def test_checkpoint_is_saved_loaded_and_evaluated(self):
        import evaluate_run
        import run_training as script
        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "s.json").write_text(json.dumps(TINY))
            script.main(["--condition", FRONTIER, "--settings", str(tmp / "s.json"), "--out", str(tmp / "run"),
                         "--checkpoints", "1,2"])
            policy = runs.load_policy(tmp / "run", "checkpoint_ep1")
            summary = json.loads((tmp / "run" / "summary.json").read_text())
            self.assertEqual(policy.parameter_digest(), summary["checkpoint_digests"]["1"])
            with self.assertRaisesRegex(ValueError, "kept no"):
                runs.load_policy(tmp / "run", "checkpoint_ep3")
            evaluate_run.main(["--run", str(tmp / "run"), "--which", "checkpoint_ep2", "--quick",
                               "--skip-adaptation"])
            result = json.loads((tmp / "run" / "evaluation_checkpoint_ep2.json").read_text())
            self.assertEqual(result["run"]["policy_digest"], summary["checkpoint_digests"]["2"])
            shutil.copy(tmp / "run" / "pretrained.pt", tmp / "run" / "checkpoint_ep1.pt")
            with self.assertRaisesRegex(ValueError, "does not match"):
                runs.load_policy(tmp / "run", "checkpoint_ep1")
        finally:
            shutil.rmtree(tmp)


class TestLauncherGrid(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def plan(self, **overrides):
        plan = {"out": str(self.tmp / "out"), "settings": {**TINY, "purpose": "smoke"}, "seeds": [0],
                "conditions": [TASK_DIRECTED], "evaluate": {"quick": True, "skip_adaptation": True},
                "grid": {"learning_rate": [0.0003, 0.001]}, "checkpoints": [1]}
        plan.update(overrides)
        path = self.tmp / "plan.json"
        path.write_text(json.dumps(plan))
        return path

    def test_grid_jobs(self):
        import launch
        plan = launch.load_plan(self.plan(grid={"learning_rate": [0.0003, 0.001], "updates_per_env_step": [0.5, 1]}))
        out, jobs = launch.build_jobs(plan)
        trains = [j for j in jobs if j.kind == "train"]
        self.assertEqual(len(trains), 4)
        self.assertEqual({j.folder.parts[-5] for j in trains}, {"lr0.0003_r0.5", "lr0.0003_r1", "lr0.001_r0.5", "lr0.001_r1"})
        self.assertEqual(sum(j.kind == "evaluate_pretrained" for j in jobs), 4)
        self.assertEqual(sum("checkpoint_ep1" in j.name for j in jobs), 4)
        for j in jobs:
            if "checkpoint" in j.name:
                self.assertIn("--skip-adaptation", j.command)
        for bad in ({"grid": {"seed": [1, 2]}}, {"grid": {"learning_rate": []}}, {"grid": {"budget_steps": [60]}},
                    {"grid": {"nonsense": [1]}}, {"checkpoints": [0]}):
            with self.assertRaises(SystemExit, msg=bad):
                launch.load_plan(self.plan(**bad))
        with self.assertRaisesRegex(SystemExit, "beyond"):
            launch.build_jobs(launch.load_plan(self.plan(checkpoints=[4])))

    def test_grid_launch_and_checks(self):
        import launch
        self.assertEqual(launch.main([str(self.plan()), "--workers", "4"]), 0)
        out = self.tmp / "out"
        for cell in ("lr0.0003", "lr0.001"):
            run = out / "cells" / cell / "runs" / "weighted" / TASK_DIRECTED / "seed0"
            self.assertTrue((run / "evaluation_checkpoint_ep1.json").exists())
            self.assertTrue((out / "cells" / cell / "pretrained" / "weighted" / "seed0.json").exists())
        # Two learning rates in one seed: a mismatch for a pilot-like batch, but separate cells in tuning.
        folders = sorted(out.glob("cells/*/runs/weighted/*/seed0"))
        report = runs.check_pretrained(folders=folders)
        self.assertFalse(report["passed"])  # purpose smoke: different warm-up settings in one batch
        for folder in folders:
            for name in ("settings.json",):
                settings = json.loads((folder / name).read_text())
                settings["purpose"] = "tuning"
                (folder / name).write_text(json.dumps(settings))
            summary = json.loads((folder / "summary.json").read_text())
            summary["settings"]["purpose"] = "tuning"
            (folder / "summary.json").write_text(json.dumps(summary))
        self.assertTrue(runs.check_pretrained(folders=folders)["passed"])
        # Resuming with other checkpoints fails instead of mixing runs.
        self.assertEqual(launch.main([str(self.plan(checkpoints=[2])), "--workers", "4"]), 1)


def tuning_settings(lr, ratio):
    return {**asdict(TrainingSettings(condition=TASK_DIRECTED, maze="M0", purpose="tuning")),
            "learning_rate": lr, "updates_per_env_step": ratio}


class TestSelectTuning(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.out)

    def cell(self, lr, ratio, s_in, pretrained=0.5, early=0.5, seeds=3, quick=False):
        label = f"lr{lr:g}_r{ratio:g}"
        settings = tuning_settings(lr, ratio)
        for condition in ("task_directed_practice", "frontier_novelty"):
            for seed in range(seeds):
                run = self.out / "cells" / label / "runs" / "weighted" / condition / f"seed{seed}"
                value = s_in[condition] if isinstance(s_in, dict) else s_in
                fake_evaluation(run / "evaluation_final.json", settings, "final", value, quick=quick)
                fake_evaluation(run / "evaluation_checkpoint_ep5.json", settings, "checkpoint_ep5", early)
                fake_evaluation(run / "evaluation_checkpoint_ep25.json", settings, "checkpoint_ep25", 0.0)
        for seed in range(seeds):
            fake_evaluation(self.out / "cells" / label / "pretrained" / "weighted" / f"seed{seed}.json",
                            settings, "pretrained", pretrained)

    def test_objective_and_ties(self):
        import select_tuning
        self.cell(0.001, 1, {"task_directed_practice": 0.9, "frontier_novelty": 0.5})   # mean 0.70
        self.cell(0.0003, 2, 0.705)
        self.cell(0.001, 0.5, 0.695)   # tied within 0.01, smallest ratio
        self.cell(0.0003, 0.5, 0.60)
        report = select_tuning.select(self.out)
        self.assertEqual(report["chosen"]["cell"], "lr0.001_r0.5")
        self.assertAlmostEqual(report["cells"][0]["score"], 0.705)
        self.assertEqual(set(report["tied_with_best"]), {"lr0.001_r1", "lr0.0003_r2", "lr0.001_r0.5"})
        self.assertFalse(report["early_drop"]["triggered"])  # uses the first checkpoint (5), not 25

    def test_learning_rate_breaks_remaining_ties(self):
        import select_tuning
        self.cell(0.001, 1, 0.7)
        self.cell(0.0003, 1, 0.7)
        self.assertEqual(select_tuning.select(self.out)["chosen"]["cell"], "lr0.0003_r1")

    def test_early_drop(self):
        import select_tuning
        self.cell(0.001, 1, 0.8, pretrained=0.7, early=0.5)
        report = select_tuning.select(self.out)
        self.assertAlmostEqual(report["early_drop"]["mean"], 0.2)
        self.assertTrue(report["early_drop"]["triggered"])
        self.assertIn("learning-rate warm-up", report["early_drop"]["next"])

    def test_refusals(self):
        import select_tuning
        with self.assertRaisesRegex(SystemExit, "no tuning cells"):
            select_tuning.select(self.out)
        self.cell(0.001, 1, 0.7)
        self.cell(0.0003, 1, 0.7, seeds=2)
        with self.assertRaisesRegex(SystemExit, "different numbers of seeds"):
            select_tuning.select(self.out)
        missing = self.out / "cells" / "lr0.0003_r1" / "runs" / "weighted" / "frontier_novelty" / "seed0"
        (missing / "evaluation_final.json").unlink()
        with self.assertRaisesRegex(SystemExit, "incomplete"):
            select_tuning.select(self.out)
        shutil.rmtree(self.out / "cells")
        self.cell(0.001, 1, 0.7, quick=True)
        with self.assertRaisesRegex(SystemExit, "quick"):
            select_tuning.select(self.out)


class TestPreconditionCheck(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.out)

    def seeds(self, learner, reps_by_seed, **kwargs):
        settings = {**asdict(TrainingSettings(condition=TASK_DIRECTED, purpose="precondition", budget_steps=0)),
                    "learner": learner}
        for seed, reps in enumerate(reps_by_seed):
            fake_evaluation(self.out / "pretrained" / learner / f"seed{seed}.json", settings, "pretrained",
                            reps=reps, **kwargs)

    def both(self, weighted, plain=None):
        self.seeds("weighted", weighted)
        self.seeds("plain", plain if plain is not None else weighted)

    def test_pass(self):
        import precondition_check
        rows = {1: 0.3, 2: 0.3, 3: 0.4, 5: 0.4, 6: 0.3, 7: 0.3}
        self.both([rows, rows])
        report = precondition_check.check(self.out)
        self.assertTrue(report["passed"])
        self.assertEqual(report["next"], "proceed to the pilot")

    def test_seed_mean_versus_per_seed(self):
        import precondition_check
        good = {1: 0.3, 2: 0.3, 3: 0.4, 5: 0.4, 6: 0.3, 7: 0.3}
        dip = {**good, 1: 0.05}   # one seed's outer pair below the floor; its seed mean is 0.175
        self.both([good, dip])
        self.assertTrue(precondition_check.check(self.out, "seed-mean")["passed"])
        self.assertFalse(precondition_check.check(self.out, "per-seed")["passed"])

    def test_undefined_counts_as_zero(self):
        import precondition_check
        good = {1: 0.3, 2: 0.3, 3: 0.4, 5: 0.4, 6: 0.3, 7: 0.3}
        self.both([good, {**good, 7: None}])
        report = precondition_check.check(self.out)
        self.assertAlmostEqual(report["learners"]["weighted"]["Rep_by_pair"][7], 0.15)
        self.assertEqual(report["learners"]["weighted"]["undefined_counted_as_zero"]["seed1"], [7])

    def test_failure_rules(self):
        import precondition_check
        rows = {1: 0.05, 2: 0.3, 3: 0.45, 5: 0.45, 6: 0.3, 7: 0.3}   # mean 0.31; row 1 below the floor
        self.both([rows])
        report = precondition_check.check(self.out)
        self.assertFalse(report["passed"])
        exclude = report["failure_rules"]["exclude"]
        self.assertEqual(exclude["pairs_excluded_from_primary"], [1])
        self.assertEqual(exclude["primary_pairs_left"], {"north": [2, 3], "south": [5, 6, 7]})
        self.assertIn("proceed to the pilot with these pairs excluded", report["next"])
        self.assertIn("280", precondition_check.check(self.out, failure_rule="add-demos")["next"])

    def test_too_few_pairs_left_or_low_mean_stops(self):
        import precondition_check
        self.both([{1: 0.05, 2: 0.3, 3: 0.05, 5: 0.45, 6: 0.3, 7: 0.3}])   # one north pair left
        self.assertIn("stop the pilot", precondition_check.check(self.out)["next"])
        shutil.rmtree(self.out / "pretrained")
        self.both([{r: 0.15 for r in (1, 2, 3, 5, 6, 7)}])   # every pair above 0.10, mean below 0.25
        report = precondition_check.check(self.out)
        self.assertFalse(report["passed"])
        self.assertIn("stop the pilot", report["next"])

    def test_refusals(self):
        import precondition_check
        rows = {1: 0.3, 2: 0.3, 3: 0.4, 5: 0.4, 6: 0.3, 7: 0.3}
        self.seeds("weighted", [rows])
        with self.assertRaisesRegex(SystemExit, "plain"):
            precondition_check.check(self.out)
        self.seeds("plain", [rows, rows])
        with self.assertRaisesRegex(SystemExit, "different numbers of seeds"):
            precondition_check.check(self.out)
        shutil.rmtree(self.out / "pretrained")
        self.seeds("weighted", [rows], quick=True)
        self.seeds("plain", [rows])
        with self.assertRaisesRegex(SystemExit, "quick"):
            precondition_check.check(self.out)
        shutil.rmtree(self.out / "pretrained")
        self.seeds("weighted", [rows], rollouts=50)
        self.seeds("plain", [rows])
        with self.assertRaisesRegex(SystemExit, "100 sampled"):
            precondition_check.check(self.out)


class TestCapacity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import capacity
        cls.capacity = capacity
        cls.buffer = capacity.expert_buffer("M2", 0)

    def test_data_covers_every_ordered_pair(self):
        from envs.mazes import free_cells
        from envs.pointmaze import MAZES
        n = len(free_cells(MAZES["M2"]))
        self.assertEqual(len(self.capacity.ordered_pairs("M2")), n * (n - 1))
        self.assertEqual(len(self.buffer), n * (n - 1))
        self.assertEqual(len(self.capacity.ordered_pairs("M1")) + len(self.capacity.ordered_pairs("M2")), 1840)
        starts = {tuple(np.floor(o[0, :2]).astype(int)) for o in self.buffer.to_arrays()["observations"]}
        self.assertGreater(len(starts), 1)

    def test_doubling_and_pairs(self):
        c = self.capacity
        tiny = c.CapacitySettings(maze="M2", updates=50, random_pair_episodes=10)
        result, _ = c.run_capacity(tiny, self.buffer)
        self.assertFalse(result["passed"])
        self.assertEqual([a["updates"] for a in result["attempts"]], [50, 100])   # one doubling, no more
        self.assertIn("design question", result["next"])
        stop, _ = c.run_capacity(c.CapacitySettings(maze="M2", updates=50, random_pair_episodes=10,
                                                    on_fail="stop"), self.buffer)
        self.assertEqual(len(stop["attempts"]), 1)
        plain, _ = c.run_capacity(c.CapacitySettings(maze="M2", learner="plain", updates=50,
                                                     random_pair_episodes=10), self.buffer)
        self.assertEqual(plain["gamma"], 1.0)
        with self.assertRaises(ValueError):
            c.CapacitySettings(maze="M0")

    def test_evaluation_pairs_are_shared_and_fixed(self):
        from unittest import mock
        c = self.capacity
        seen = []
        original = c._greedy_success

        def spy(env, policy, start, goal, seed):
            seen.append((start, goal, seed))
            return original(env, policy, start, goal, seed)

        with mock.patch.object(c, "_greedy_success", spy):
            c.evaluate(c.GCSLPolicy(1), c.CapacitySettings(maze="M2", random_pair_episodes=5))
            c.evaluate(c.GCSLPolicy(2), c.CapacitySettings(maze="M2", learner="plain", random_pair_episodes=5))
        self.assertEqual(seen[:5], seen[5:])

    def test_script_and_thresholds(self):
        import run_capacity
        c = self.capacity
        self.assertEqual(c.THRESHOLDS, {"task_pairs": 0.9, "random_pairs": 0.8})
        tmp = Path(tempfile.mkdtemp())
        try:
            run_capacity.main(["--maze", "M2", "--learner", "weighted", "--quick", "--out", str(tmp / "M2w")])
            result = json.loads((tmp / "M2w" / "capacity.json").read_text())
            self.assertTrue(result["quick_smoke_only"])
            self.assertEqual(result["settings"]["updates"], run_capacity.QUICK["updates"])
            self.assertTrue((tmp / "M2w" / "final.pt").exists())
            with self.assertRaises(SystemExit):
                run_capacity.main(["--maze", "M2", "--out", str(tmp / "x")])
            with self.assertRaisesRegex(SystemExit, "results"):
                run_capacity.main(["--all", "--quick", "--out", str(ROOT / "results" / "never")])
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
