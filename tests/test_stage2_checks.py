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


def fake_evaluation(path, settings, which, s_in=0.5, reps=None, quick=False, rollouts=100, digest="d",
                    rep_aggregate=0.0):
    """An evaluation file shaped like evaluate_run.py's, for the readers under test."""
    pairs = [{"row": r, "is_primary_pair": True, "S_in": s_in,
              "sampled": {"Rep": rep, "successes": 0 if rep is None else 50}} for r, rep in (reps or {}).items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "run": {"which": which, "settings": settings, "policy_digest": digest},
        "evaluation": {"quick_smoke_only": quick, "adaptation_run": False, "config": {"sampled_rollouts": rollouts}},
        "report": {"aggregate": {"S_in": s_in, "Rep": rep_aggregate}, "pairs": pairs}}))


def fake_manifest(out, plan):
    out.mkdir(parents=True, exist_ok=True)
    (out / "launch_manifest.json").write_text(json.dumps({"launches": [{"plan": plan}]}))


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
        plan = {"out": str(self.tmp / "out"), "settings": {**TINY, "purpose": "smoke", "maze": "M0"}, "seeds": [0],
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
        for job in trains:   # M22: each run's planned settings match its folder's label
            label = job.folder.parts[-5]
            lr, ratio = label.split("_")
            self.assertEqual(job.settings_values["learning_rate"], float(lr[2:]))
            self.assertEqual(float(job.settings_values["updates_per_env_step"]), float(ratio[1:]))
        for job in jobs:     # M23: each pretrained evaluation reads a run of its own cell
            if job.kind == "evaluate_pretrained":
                cell = job.done.parts[-4]
                run = Path(job.command[job.command.index("--run") + 1])
                self.assertEqual(run.parts[-5], cell)
        for bad in ({"grid": {"seed": [1, 2]}}, {"grid": {"learning_rate": []}}, {"grid": {"budget_steps": [60]}},
                    {"grid": {"nonsense": [1]}}, {"checkpoints": [0]},
                    {"settings": {**TINY, "purpose": "pilot"}},                      # MI2: grid only for tuning, smoke
                    {"settings": {**TINY, "purpose": "tuning", "maze": "M0"}, "grid": {}},   # tuning: both conditions
                    {"settings": {**TINY, "purpose": "smoke", "learning_rate": None}}):       # unset settings
            with self.assertRaises(SystemExit, msg=bad):
                launch.load_plan(self.plan(**bad))
        with self.assertRaisesRegex(SystemExit, "same label"):   # MI1
            launch.build_jobs(launch.load_plan(self.plan(grid={"updates_per_env_step": [1, 1.0]})))
        with self.assertRaisesRegex(ValueError, "M0 with the weighted"):   # MI2: tuning on M1 refused
            TrainingSettings(condition=TASK_DIRECTED, purpose="tuning", maze="M1")
        with self.assertRaisesRegex(SystemExit, "beyond"):
            launch.build_jobs(launch.load_plan(self.plan(checkpoints=[4])))

    def test_grid_launch_and_checks(self):
        import launch
        self.assertEqual(launch.main([str(self.plan()), "--workers", "4"]), 0)
        out = self.tmp / "out"
        manifest = json.loads((out / "launch_manifest.json").read_text())
        self.assertIsNotNone(manifest["launches"][-1]["pretrained_check"])   # audit of 26 Nov, SF1
        for cell in ("lr0.0003", "lr0.001"):
            run = out / "cells" / cell / "runs" / "weighted" / TASK_DIRECTED / "seed0"
            self.assertTrue((run / "evaluation_checkpoint_ep1.json").exists())
            self.assertTrue((out / "cells" / cell / "pretrained" / "weighted" / "seed0.json").exists())
        # Two learning rates in one seed: separate cells for a grid's purposes, a mismatch for any other.
        folders = sorted(out.glob("cells/*/runs/weighted/*/seed0"))
        self.assertTrue(runs.check_pretrained(folders=folders)["passed"])

        def set_purpose(purpose):
            for folder in folders:
                settings = json.loads((folder / "settings.json").read_text())
                settings["purpose"] = purpose
                (folder / "settings.json").write_text(json.dumps(settings))
                summary = json.loads((folder / "summary.json").read_text())
                summary["settings"]["purpose"] = purpose
                (folder / "summary.json").write_text(json.dumps(summary))

        set_purpose("tuning")
        self.assertTrue(runs.check_pretrained(folders=folders)["passed"])
        # The same runs outside a grid's cells are one batch, and their two warm-ups are a mismatch.
        flat = [self.tmp / "flat" / f"run{i}" for i in range(len(folders))]
        for source, target in zip(folders, flat):
            shutil.copytree(source, target)
        self.assertTrue(any("differ in warm-up settings" in p
                            for p in runs.check_pretrained(folders=flat)["problems"]))
        # Tuning cells still must have run the same code (SF1).
        record = json.loads((folders[0] / "run_record.json").read_text())
        record["source_sha256"]["src/gcsl.py"] = "0" * 64
        (folders[0] / "run_record.json").write_text(json.dumps(record))
        report = runs.check_pretrained(folders=folders)
        self.assertTrue(any("different code" in p for p in report["problems"]))
        # Resuming with other checkpoints fails instead of mixing runs (on a clean copy of the launch).
        record["source_sha256"]["src/gcsl.py"] = runs.code_hashes(json.loads(
            (folders[1] / "run_record.json").read_text()))["src/gcsl.py"]
        (folders[0] / "run_record.json").write_text(json.dumps(record))
        set_purpose("smoke")
        self.assertEqual(launch.main([str(self.plan()), "--workers", "4"]), 0)   # unchanged plan resumes
        self.assertEqual(launch.main([str(self.plan(checkpoints=[2])), "--workers", "4"]), 1)
        manifest = json.loads((out / "launch_manifest.json").read_text())
        self.assertTrue(any("kept checkpoints" in (j["log"] or "") for j in manifest["launches"][-1]["jobs"]))


TUNING_PLAN = {"settings": {"purpose": "tuning", "maze": "M0", "budget_steps": 99960}, "seeds": [0, 1, 2],
               "conditions": ["task_directed_practice", "frontier_novelty"], "plain_conditions": [],
               "grid": {"learning_rate": [0.0003, 0.001], "updates_per_env_step": [0.5, 1, 2]},
               "checkpoints": [5, 25]}


class TestSelectTuning(unittest.TestCase):
    """The selector on fake tuning launches: real run folders (settings and summaries), fake evaluations."""

    def setUp(self):
        self.out = Path(tempfile.mkdtemp())
        self.plan = json.loads(json.dumps(TUNING_PLAN))
        fake_manifest(self.out, self.plan)

    def tearDown(self):
        shutil.rmtree(self.out)

    def run_folder(self, label, condition, seed, settings, s_in, early, pretrained, rep=0.0, quick=False,
                   which_final="final"):
        folder = self.out / "cells" / label / "runs" / "weighted" / condition / f"seed{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        digests = {"final": f"F{label}{condition}{seed}", "pretrained": f"P{label}{seed}",
                   "5": f"C5{label}{condition}{seed}", "25": f"C25{label}{condition}{seed}"}
        (folder / "settings.json").write_text(json.dumps(settings))
        (folder / "summary.json").write_text(json.dumps({
            "settings": settings, "final_digest": digests["final"], "pretrained_digest": digests["pretrained"],
            "checkpoint_digests": {"5": digests["5"], "25": digests["25"]}, "budget": {"passed": True}}))
        fake_evaluation(folder / "evaluation_final.json", settings, which_final, s_in, quick=quick,
                        digest=digests["final"], rep_aggregate=rep)
        fake_evaluation(folder / "evaluation_checkpoint_ep5.json", settings, "checkpoint_ep5", early, digest=digests["5"])
        fake_evaluation(folder / "evaluation_checkpoint_ep25.json", settings, "checkpoint_ep25", 0.0, digest=digests["25"])
        fake_evaluation(self.out / "cells" / label / "pretrained" / "weighted" / f"seed{seed}.json", settings,
                        "pretrained", pretrained, digest=digests["pretrained"])

    def grid(self, scores=None, pretrained=0.5, early=0.5, reps=None, **kwargs):
        """Every agreed cell; scores maps a cell label to S_in (or to {condition: S_in}); default 0.5."""
        from launch import cells
        from training import settings_from_dict
        for label, overrides in cells({**self.plan, "grid": self.plan["grid"]}):
            for condition in self.plan["conditions"]:
                for seed in self.plan["seeds"]:
                    settings = asdict(settings_from_dict({**self.plan["settings"], **overrides,
                                                          "condition": condition, "seed": seed,
                                                          "learner": "weighted"}))
                    value = (scores or {}).get(label, 0.5)
                    value = value[condition] if isinstance(value, dict) else value
                    self.run_folder(label, condition, seed, settings, value, early, pretrained,
                                    rep=(reps or {}).get(label, 0.0), **kwargs)

    def test_objective_and_ties(self):
        import select_tuning
        self.grid({"lr0.001_r1": {"task_directed_practice": 0.9, "frontier_novelty": 0.5},   # 0.70
                   "lr0.0003_r2": 0.705, "lr0.001_r0.5": 0.695, "lr0.0003_r0.5": 0.60})
        report = select_tuning.select(self.out)
        self.assertEqual(report["chosen"]["cell"], "lr0.001_r0.5")   # tied within 0.01, smallest ratio
        self.assertAlmostEqual(report["cells"][0]["score"], 0.705)
        self.assertEqual(set(report["tied_with_best"]), {"lr0.001_r1", "lr0.0003_r2", "lr0.001_r0.5"})
        self.assertFalse(report["early_drop"]["triggered"])   # the first checkpoint (5), not 25
        self.assertEqual(report["early_drop_checkpoint"], 5)

    def test_repertoire_never_counts(self):   # audit of 26 Nov, M9
        import select_tuning
        self.grid({"lr0.001_r1": 0.70, "lr0.0003_r1": 0.60}, reps={"lr0.0003_r1": 0.5, "lr0.001_r1": 0.0})
        self.assertEqual(select_tuning.select(self.out)["chosen"]["cell"], "lr0.001_r1")

    def test_learning_rate_breaks_remaining_ties(self):
        import select_tuning
        self.grid({"lr0.001_r1": 0.9, "lr0.0003_r1": 0.9})
        self.assertEqual(select_tuning.select(self.out)["chosen"]["cell"], "lr0.0003_r1")

    def test_early_drop(self):
        import select_tuning
        self.grid({"lr0.001_r1": 0.9}, pretrained=0.7, early=0.5)
        report = select_tuning.select(self.out)
        self.assertAlmostEqual(report["early_drop"]["mean"], 0.2)
        self.assertTrue(report["early_drop"]["triggered"])
        self.assertIn("learning-rate warm-up", report["early_drop"]["next"])
        seeds = report["cells"][0]["early_drop_by_seed"]["frontier_novelty"]
        self.assertEqual({tuple(sorted(d["drops"])) for d in seeds}, {(5, 25)})   # MI5: every checkpoint

    def test_refusals(self):
        import select_tuning
        self.grid()
        shutil.rmtree(self.out / "cells" / "lr0.0003_r0.5")
        with self.assertRaisesRegex(SystemExit, "differ from the plan"):            # SF2: a missing cell
            select_tuning.select(self.out)
        self.grid()
        (self.out / "cells" / "lr0.003_r1").mkdir()
        with self.assertRaisesRegex(SystemExit, "differ from the plan"):            # SF2: an extra cell
            select_tuning.select(self.out)
        shutil.rmtree(self.out / "cells" / "lr0.003_r1")
        run = self.out / "cells" / "lr0.001_r1" / "runs" / "weighted" / "frontier_novelty" / "seed2"
        settings = json.loads((run / "settings.json").read_text())
        wrong = {**settings, "updates_per_env_step": 2.0}
        (run / "settings.json").write_text(json.dumps(wrong))
        summary = json.loads((run / "summary.json").read_text())
        (run / "summary.json").write_text(json.dumps({**summary, "settings": wrong}))
        with self.assertRaisesRegex(SystemExit, "planned settings"):                # SF2: a run in the wrong cell
            select_tuning.select(self.out)
        (run / "settings.json").write_text(json.dumps(settings))
        (run / "summary.json").write_text(json.dumps(summary))
        shutil.copy(self.out / "cells" / "lr0.0003_r1" / "pretrained" / "weighted" / "seed2.json",
                    self.out / "cells" / "lr0.001_r1" / "pretrained" / "weighted" / "seed2.json")
        with self.assertRaisesRegex(SystemExit, "pretrained policy"):               # SF3: another cell's file
            select_tuning.select(self.out)
        shutil.rmtree(self.out / "cells")
        self.grid(which_final="pretrained")
        with self.assertRaisesRegex(SystemExit, "final policy"):                    # SF3: wrong weights
            select_tuning.select(self.out)
        shutil.rmtree(self.out / "cells")
        self.grid(quick=True)
        with self.assertRaisesRegex(SystemExit, "quick"):
            select_tuning.select(self.out)
        shutil.rmtree(self.out / "cells")
        self.grid()
        (self.out / "cells" / "lr0.001_r1" / "runs" / "weighted" / "frontier_novelty" / "seed1" /
         "evaluation_final.json").unlink()
        with self.assertRaisesRegex(SystemExit, "incomplete"):
            select_tuning.select(self.out)
        fake_manifest(self.out, {**self.plan, "grid": {"learning_rate": [0.001], "updates_per_env_step": [1]}})
        with self.assertRaisesRegex(SystemExit, "agreed grid"):
            select_tuning.select(self.out)


SHORTER = (1, 2, 3, 5, 6, 7)


class TestPreconditionCheck(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp())
        self.seed_list = [0, 1]
        fake_manifest(self.out, {"seeds": self.seed_list})

    def tearDown(self):
        shutil.rmtree(self.out)

    def write(self, learner, reps_by_seed, seeds=None, settings_overrides=None, **kwargs):
        for seed, reps in zip(seeds or self.seed_list, reps_by_seed):
            settings = {**asdict(TrainingSettings(condition=TASK_DIRECTED, purpose="precondition", budget_steps=0,
                                                  learner=learner, seed=seed)), **(settings_overrides or {})}
            fake_evaluation(self.out / "pretrained" / learner / f"seed{seed}.json", settings, "pretrained",
                            reps=reps, digest=f"{learner}{seed}", **kwargs)

    def both(self, weighted, plain=None):
        self.write("weighted", weighted)
        self.write("plain", plain if plain is not None else weighted)

    def good(self, change=None):
        return {**{1: 0.3, 2: 0.3, 3: 0.4, 5: 0.4, 6: 0.3, 7: 0.3}, **(change or {})}

    def test_pass_and_fingerprints(self):
        import precondition_check
        self.both([self.good(), self.good()])
        report = precondition_check.check(self.out)
        self.assertTrue(report["passed"])
        self.assertEqual(report["next"], "proceed to the pilot")
        self.assertEqual(report["pretrained_digests"]["plain"], {"seed0": "plain0", "seed1": "plain1"})

    def test_seed_mean_versus_per_seed(self):
        import precondition_check
        self.both([self.good(), self.good({1: 0.05})])   # one seed's outer pair below the floor
        self.assertTrue(precondition_check.check(self.out, "seed-mean")["passed"])
        self.assertFalse(precondition_check.check(self.out, "per-seed")["passed"])

    def test_undefined_counts_as_zero(self):
        import precondition_check
        self.both([self.good(), self.good({7: None})])
        report = precondition_check.check(self.out)
        self.assertAlmostEqual(report["learners"]["weighted"]["Rep_by_pair"][7], 0.15)
        self.assertEqual(report["learners"]["weighted"]["undefined_counted_as_zero"]["seed1"], [7])

    def test_failure_rules(self):
        import precondition_check
        self.both([self.good({1: 0.05, 3: 0.45, 5: 0.45})] * 2)
        report = precondition_check.check(self.out)
        self.assertFalse(report["passed"])
        exclude = report["failure_rules"]["exclude"]
        self.assertEqual(exclude["pairs_excluded_from_primary"], [1])
        self.assertEqual(exclude["primary_pairs_left"], {"north": [2, 3], "south": [5, 6, 7]})
        self.assertIn("proceed to the pilot with these pairs excluded", report["next"])
        self.assertIn("280", precondition_check.check(self.out, failure_rule="add-demos")["next"])

    def test_outer_pairs_excluded_with_kept_mean(self):   # audit of 26 Nov, SF5: case P5
        import precondition_check
        self.both([{1: 0.05, 2: 0.30, 3: 0.35, 5: 0.35, 6: 0.30, 7: 0.05}] * 2)
        report = precondition_check.check(self.out)
        self.assertAlmostEqual(report["learners"]["weighted"]["mean_Rep"], 0.2333, places=3)
        self.assertAlmostEqual(report["failure_rules"]["exclude"]["mean_Rep_over_kept_pairs"]["weighted"], 0.325)
        self.assertIn("proceed to the pilot with these pairs excluded", report["next"])

    def test_one_pair_left_on_a_side_stops(self):   # M17: a passing mean, one north pair left
        import precondition_check
        self.both([{1: 0.05, 2: 0.45, 3: 0.05, 5: 0.45, 6: 0.45, 7: 0.45}] * 2)
        report = precondition_check.check(self.out)
        self.assertGreaterEqual(report["failure_rules"]["exclude"]["mean_Rep_over_kept_pairs"]["weighted"], 0.25)
        self.assertIn("stop the pilot", report["next"])

    def test_either_learner_failing_excludes_the_pair(self):   # M19: only the plain learner fails row 6
        import precondition_check
        self.both([self.good()] * 2, [self.good({6: 0.05})] * 2)
        report = precondition_check.check(self.out)
        self.assertTrue(report["learners"]["weighted"]["passed"])
        self.assertFalse(report["passed"])
        self.assertEqual(report["failure_rules"]["exclude"]["pairs_excluded_from_primary"], [6])

    def test_low_mean_stops(self):
        import precondition_check
        self.both([{r: 0.15 for r in SHORTER}] * 2)   # every pair above 0.10, mean below 0.25
        report = precondition_check.check(self.out)
        self.assertFalse(report["passed"])
        self.assertIn("stop the pilot", report["next"])

    def test_refusals(self):
        import precondition_check
        self.write("weighted", [self.good()] * 2)
        with self.assertRaisesRegex(SystemExit, "plain learner has evaluations for seeds"):
            precondition_check.check(self.out)
        self.write("plain", [self.good()], seeds=[1])
        with self.assertRaisesRegex(SystemExit, "plain learner has evaluations for seeds"):   # SF4: wrong seeds
            precondition_check.check(self.out)
        shutil.rmtree(self.out / "pretrained")
        self.write("weighted", [self.good()] * 2)
        self.write("plain", [self.good()] * 2, settings_overrides={"learning_rate": 3e-4})
        with self.assertRaisesRegex(SystemExit, "different warm-up"):                   # SF4: two warm-ups
            precondition_check.check(self.out)
        shutil.rmtree(self.out / "pretrained")
        self.write("weighted", [self.good()] * 2)
        self.write("plain", [self.good()] * 2, settings_overrides={"learner": "weighted"})
        with self.assertRaisesRegex(SystemExit, "is not the plain learner"):            # SF4: wrong learner
            precondition_check.check(self.out)
        shutil.rmtree(self.out / "pretrained")
        self.write("weighted", [self.good()] * 2)
        self.write("plain", [self.good()] * 2, settings_overrides={"maze": "M0"})
        with self.assertRaisesRegex(SystemExit, "on M1"):                               # SF4: wrong maze
            precondition_check.check(self.out)
        shutil.rmtree(self.out / "pretrained")
        self.write("weighted", [self.good()] * 2)
        self.write("plain", [{**self.good(), 4: 0.5}] * 2)
        with self.assertRaisesRegex(SystemExit, "shorter-route pairs"):                 # MI6: tie row included
            precondition_check.check(self.out)
        shutil.rmtree(self.out / "pretrained")
        self.both([self.good()] * 2)
        tuning = self.out / "selection.json"
        tuning.write_text(json.dumps({"chosen": {"learning_rate": 3e-4}}))
        with self.assertRaisesRegex(SystemExit, "not the tuned"):
            precondition_check.check(self.out, tuning_report=tuning)
        shutil.rmtree(self.out / "pretrained")
        self.write("weighted", [self.good()] * 2, quick=True)
        self.write("plain", [self.good()] * 2)
        with self.assertRaisesRegex(SystemExit, "quick"):
            precondition_check.check(self.out)
        shutil.rmtree(self.out / "pretrained")
        self.write("weighted", [self.good()] * 2, rollouts=50)
        self.write("plain", [self.good()] * 2)
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

    def test_pass_rule(self):   # audit of 26 Nov, M2-M4
        passes = self.capacity.passes
        self.assertTrue(passes({"random_pairs": 0.8}))
        self.assertFalse(passes({"random_pairs": 0.79}))
        self.assertTrue(passes({"random_pairs": 0.8, "task_pairs": 0.9}))
        self.assertFalse(passes({"random_pairs": 0.95, "task_pairs": 0.89}))

    def test_m1_reports_task_pairs(self):
        c = self.capacity
        result = c.evaluate(c.GCSLPolicy(0), c.CapacitySettings(maze="M1", random_pair_episodes=2,
                                                                task_pair_rollouts=1))
        self.assertEqual(sorted(result["task_pairs_by_row"]), [1, 2, 3, 4, 5, 6, 7])
        self.assertIn("task_pairs_worst", result)

    def test_expert_failure_stops_the_check(self):   # M8
        from unittest import mock
        c = self.capacity
        original = c.run_episode
        calls = [0]

        def fail_once(*args, **kwargs):
            calls[0] += 1
            episode = original(*args, **kwargs)
            return {**episode, "success": False} if calls[0] == 3 else episode

        with mock.patch.object(c, "run_episode", fail_once):
            with self.assertRaisesRegex(AssertionError, "expert failed 1 pairs"):
                c.expert_buffer("M2", 0)

    def test_all_refuses_a_used_folder(self):   # SF6
        import run_capacity
        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "M1_weighted").mkdir()
            (tmp / "M1_weighted" / "capacity.json").write_text("{}")
            with self.assertRaisesRegex(SystemExit, "not empty"):
                run_capacity.main(["--all", "--quick", "--out", str(tmp)])
        finally:
            shutil.rmtree(tmp)

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
