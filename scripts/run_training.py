"""Run one training run: one condition, one seed, one learner, one maze.

Run: python scripts/run_training.py --condition task_directed_practice --seed 0 --out OUT
Conditions: demonstrations_only, expert_top_up, random_actions,
task_directed_practice, frontier_novelty; --learner plain for the plain-learner
versions of task-directed practice and frontier novelty.

One process per seed and condition, one CPU thread per process (Stage 2
setting 1). Launch runs in parallel as separate processes. `--budget-steps 0`
stops after warm-up, which is what the precondition check needs.

Writes into OUT: settings.json, run_record.json (commit, dirty flag, source
hashes, machine, package versions, taken at start), episodes.jsonl (one line
per practice episode, written as it finishes), summary.json (budget audit,
data diversity, exposure totals, wall-clock) and pretrained.pt / final.pt (the
policy network's weights). Writing into results/ needs a clean, committed
working tree; anything else (a scratch folder) is always allowed and is not a
filed result. --allow-dirty marks an exploratory run.

This script trains; it does not evaluate. Repertoire, competence and recovery
are measured afterwards with the evaluation suite, whose adaptation settings
come from `training.evaluation_config_for` and `training.adapter_for`.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gcsl import pin_single_thread  # noqa: E402
from provenance import guard_results_write, run_record  # noqa: E402
from training import CONDITIONS, TrainingSettings, run_training, settings_from_dict  # noqa: E402


def parse_settings(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    # No defaults here: an unset flag leaves the settings file's value (or the
    # TrainingSettings default) in place (audit of 4 Nov, SF1).
    parser.add_argument("--seed", type=int)
    parser.add_argument("--maze", choices=("M0", "M1"))
    parser.add_argument("--learner", choices=("weighted", "plain"))
    parser.add_argument("--purpose")
    parser.add_argument("--budget-steps", type=int)
    parser.add_argument("--updates-per-env-step", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--settings", type=Path, help="JSON file of further TrainingSettings fields")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    values = json.loads(args.settings.read_text()) if args.settings else {}
    for name in ("condition", "seed", "maze", "learner", "purpose", "budget_steps", "updates_per_env_step",
                 "learning_rate"):
        flag = getattr(args, name)
        if flag is None:
            continue
        if name in values and values[name] != flag:
            raise SystemExit(f"--{name.replace('_', '-')} {flag} conflicts with {values[name]!r} "
                             "in the settings file")
        values[name] = flag
    return settings_from_dict(values), args


def main(argv=None):
    settings, args = parse_settings(argv)
    assert isinstance(settings, TrainingSettings)
    import torch
    record = run_record(ROOT)  # taken once, at start, so it describes the code that ran
    guard_results_write(ROOT, args.out, args.allow_dirty)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):  # never overwrite a run, finished or crashed (audit of 4 Nov, MI6)
        raise SystemExit(f"{out} is not empty; choose another folder")
    record["allow_dirty"] = args.allow_dirty
    record["torch_threads"] = {"intra_op": torch.get_num_threads(), "inter_op": torch.get_num_interop_threads()}
    (out / "run_record.json").write_text(json.dumps(record, indent=2))
    (out / "settings.json").write_text(json.dumps(asdict(settings), indent=2))

    with open(out / "episodes.jsonl", "w") as episodes:
        def write(row):
            episodes.write(json.dumps(row) + "\n")
            episodes.flush()
        result = run_training(settings, on_episode=write)
    torch.save(result["pretrained_state"], out / "pretrained.pt")
    torch.save(result["final_state"], out / "final.pt")
    (out / "summary.json").write_text(json.dumps(result["summary"], indent=2))
    budget = result["summary"]["budget"]
    print(f"{settings.condition} seed {settings.seed} ({settings.learner}, {settings.maze}): "
          f"{budget['practice_environment_steps']} environment steps, "
          f"{budget['warmup_gradient_steps']} + {budget['practice_gradient_steps']} gradient steps; "
          f"budget audit passed")


if __name__ == "__main__":
    pin_single_thread()  # before torch does any work, so it can still be set
    main()
