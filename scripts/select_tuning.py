"""Pick the tuning cell on M0 and run the early-drop check (spec section 7; v0.12).

Run: python scripts/select_tuning.py OUT [--report REPORT.json]

OUT is a finished tuning launch (scripts/launch.py with purpose "tuning", a
"grid" over learning rate and updates per environment step, and "checkpoints").
This script only reads its evaluations; it runs nothing.

The objective (decision 2 of the 4 Nov audit, still open; Claude's
recommendation is the default). For each grid cell, the score is the mean
greedy competence S_in of the final policies of task-directed practice and
frontier novelty, over the cell's seeds:

    score(cell) = (1 / C) x sum over conditions c of [ (1 / S) x sum over seeds s of S_in(c, s) ]

where C = 2 conditions and S = the number of seeds (3 by default, code default
C9): the mean of the two conditions' seed means. Repertoire (Rep), a primary endpoint, is never used, so tuning cannot
favour a condition by its repertoire. The chosen cell has the highest score;
cells within 0.01 of it count as tied, and among tied cells the smaller update
ratio wins (decision 2), then the smaller learning rate (Claude's tiebreak, for
review).

The early-drop check (spec v0.12 fallback). For every condition and seed, the
check compares greedy S_in of the policy kept after the first checkpoint's
practice episodes with S_in of the shared pretrained policy:

    drop(c, s) = S_in(pretrained, s) - S_in(checkpoint, c, s)

If the mean drop for the chosen cell exceeds the threshold (0.10 by default:
"clearly below" in the spec has no number, so this one is Claude's, for
review), the report says the fallback is triggered: add a linear learning-rate
warm-up of fixed length and run tuning again, on M0, before the pilot.

What it checks before choosing (audit of 26 Nov, SF2 and SF3): the plan the
launcher ran (OUT/launch_manifest.json) is a tuning plan with the agreed grid
(--grid-override to allow another); the cells found are exactly the plan's;
every cell holds the plan's seeds for both conditions; every run's settings are
its cell's planned settings; and every evaluation it reads evaluated that run's
final, checkpoint or pretrained policy (by fingerprint), with the full
protocol. It reports the drop at every checkpoint for every seed; the flag uses
the mean drop at the first checkpoint (MI5: whether that is enough is open).
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

TUNED = ("task_directed_practice", "frontier_novelty")
AGREED_GRID = {"learning_rate": [0.0003, 0.001], "updates_per_env_step": [0.5, 1, 2]}   # spec v0.13


def _evaluation(path, which, digest):
    """S_in of an evaluation file, refused unless it evaluated `which` with fingerprint
    `digest` (audit of 26 Nov, SF3) with the full protocol."""
    result = json.loads(Path(path).read_text())
    if result["evaluation"]["quick_smoke_only"]:
        raise SystemExit(f"{path} is a quick (smoke) evaluation; tuning uses the full protocol")
    if result["run"]["which"] != which or result["run"]["policy_digest"] != digest:
        raise SystemExit(f"{path} did not evaluate this run's {which} policy")
    return float(result["report"]["aggregate"]["S_in"])


def _plan(out):
    manifest = Path(out) / "launch_manifest.json"
    if not manifest.exists():
        raise SystemExit(f"{manifest} is missing: the selector reads the plan the launcher ran")
    return json.loads(manifest.read_text())["launches"][-1]["plan"]


def select(out, tie=0.01, drop_threshold=0.10, conditions=TUNED, grid_override=False):
    from launch import cells as plan_cells
    from runs import load_run
    from training import settings_from_dict
    out = Path(out)
    plan = _plan(out)
    if plan["settings"].get("purpose") != "tuning":
        raise SystemExit("the launch is not a tuning launch")
    if not grid_override and {k: sorted(v) for k, v in plan["grid"].items()} != AGREED_GRID:
        raise SystemExit(f"the plan's grid {plan['grid']} is not the agreed grid {AGREED_GRID} "
                         "(--grid-override to allow it)")
    if sorted(plan["conditions"]) != sorted(conditions) or plan.get("plain_conditions"):
        raise SystemExit("a tuning plan runs task-directed practice and frontier novelty on the weighted learner")
    expected = dict(plan_cells(plan))
    found = sorted(p.name for p in (out / "cells").iterdir() if p.is_dir()) if (out / "cells").exists() else []
    if sorted(expected) != found:  # audit of 26 Nov, SF2: exactly the planned cells
        raise SystemExit(f"cells found {found} differ from the plan's {sorted(expected)}")
    rows = []
    for label, overrides in plan_cells(plan):
        cell = out / "cells" / label
        scores, drops, missing = {}, {}, []
        for condition in conditions:
            for seed in plan["seeds"]:
                folder = cell / "runs" / "weighted" / condition / f"seed{seed}"
                final = folder / "evaluation_final.json"
                if not (folder / "summary.json").exists() or not final.exists():
                    missing.append(str(final))
                    continue
                settings, summary, _ = load_run(folder)
                planned = settings_from_dict({**plan["settings"], **overrides, "condition": condition,
                                              "seed": seed, "learner": "weighted"})
                if settings != planned:  # SF2: every run is the plan's, in its own cell
                    raise SystemExit(f"{folder} was not run with its cell's planned settings")
                scores.setdefault(condition, []).append(_evaluation(final, "final", summary["final_digest"]))
                pretrained = cell / "pretrained" / "weighted" / f"seed{seed}.json"
                before = _evaluation(pretrained, "pretrained", summary["pretrained_digest"])
                per_checkpoint = {}
                for n in plan["checkpoints"]:
                    which = f"checkpoint_ep{n}"
                    digest = (summary.get("checkpoint_digests") or {}).get(str(n))
                    per_checkpoint[n] = before - _evaluation(folder / f"evaluation_{which}.json", which, digest)
                drops.setdefault(condition, []).append({"seed": seed, "drops": per_checkpoint})
        if missing:
            raise SystemExit(f"incomplete cell {label}, nothing chosen: {missing}")
        # MI4: the mean of the two conditions' seed means, as the objective states.
        by_condition = {c: sum(v) / len(v) for c, v in scores.items()}
        first = plan["checkpoints"][0] if plan["checkpoints"] else None
        first_drops = [d["drops"][first] for c in conditions for d in drops[c]] if first else []
        rows.append({
            "cell": label, "learning_rate": overrides.get("learning_rate", plan["settings"].get("learning_rate")),
            "updates_per_env_step": overrides.get("updates_per_env_step",
                                                  plan["settings"].get("updates_per_env_step", 1.0)),
            "S_in_by_condition": by_condition, "score": sum(by_condition.values()) / len(by_condition),
            "early_drop_mean": sum(first_drops) / len(first_drops) if first_drops else None,
            "early_drop_by_seed": drops,   # MI5: every checkpoint, every seed
        })
    best = max(r["score"] for r in rows)
    tied = [r for r in rows if best - r["score"] <= tie + 1e-12]
    chosen = min(tied, key=lambda r: (r["updates_per_env_step"], r["learning_rate"]))
    if chosen["early_drop_mean"] is None:
        raise SystemExit("the plan kept no checkpoints, so the early-drop check cannot run")
    triggered = chosen["early_drop_mean"] > drop_threshold
    return {
        "objective": "mean over task-directed practice and frontier novelty of each condition's mean greedy "
                     "S_in over seeds (decision 2)",
        "tie_margin": tie, "early_drop_threshold": drop_threshold, "early_drop_checkpoint": plan["checkpoints"][0],
        "cells": sorted(rows, key=lambda r: -r["score"]),
        "tied_with_best": [r["cell"] for r in tied],
        "chosen": {"cell": chosen["cell"], "learning_rate": chosen["learning_rate"],
                   "updates_per_env_step": chosen["updates_per_env_step"], "score": chosen["score"]},
        "early_drop": {"mean": chosen["early_drop_mean"], "triggered": triggered,
                       "next": ("spec v0.12 fallback: add a linear learning-rate warm-up of one fixed length, "
                                "identical everywhere, and run tuning on M0 again before the pilot")
                       if triggered else "none"},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path, help="the tuning launch's output folder")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--tie", type=float, default=0.01)
    parser.add_argument("--drop-threshold", type=float, default=0.10)
    parser.add_argument("--grid-override", action="store_true", help="allow a grid other than the agreed one")
    args = parser.parse_args(argv)
    report = select(args.out, args.tie, args.drop_threshold, grid_override=args.grid_override)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    for r in report["cells"]:
        drop = "n/a" if r["early_drop_mean"] is None else f"{r['early_drop_mean']:+.3f}"
        print(f"{r['cell']:>14}  score {r['score']:.3f}  early drop {drop}")
    c = report["chosen"]
    print(f"chosen: learning rate {c['learning_rate']:g}, {c['updates_per_env_step']:g} updates per step "
          f"(cell {c['cell']}); early-drop fallback {'TRIGGERED' if report['early_drop']['triggered'] else 'not triggered'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
