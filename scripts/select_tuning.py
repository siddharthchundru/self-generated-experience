"""Pick the tuning cell on M0 and run the early-drop check (spec section 7; v0.12).

Run: python scripts/select_tuning.py OUT [--report REPORT.json]

OUT is a finished tuning launch (scripts/launch.py with purpose "tuning", a
"grid" over learning rate and updates per environment step, and "checkpoints").
This script only reads its evaluations; it runs nothing.

The objective (decision 2 of the 4 Nov audit, still open; Claude's
recommendation is the default). For each grid cell, the score is the mean
greedy competence S_in of the final policies of task-directed practice and
frontier novelty, over the cell's seeds:

    score(cell) = (1 / (C x S)) x sum over conditions c and seeds s of S_in(c, s)

where C = 2 conditions and S = the number of seeds (3 by default, code default
C9). Repertoire (Rep), a primary endpoint, is never used, so tuning cannot
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

Refuses a launch whose cells are incomplete, whose runs are not purpose
"tuning", or whose evaluations were quick (smoke-only) ones.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TUNED = ("task_directed_practice", "frontier_novelty")


def _s_in(path):
    result = json.loads(Path(path).read_text())
    if result["evaluation"]["quick_smoke_only"]:
        raise SystemExit(f"{path} is a quick (smoke) evaluation; tuning uses the full protocol")
    return float(result["report"]["aggregate"]["S_in"]), result["run"]["settings"]


def select(out, tie=0.01, drop_threshold=0.10, conditions=TUNED):
    out = Path(out)
    cells = sorted(p for p in (out / "cells").iterdir() if p.is_dir()) if (out / "cells").exists() else []
    if not cells:
        raise SystemExit(f"{out} has no tuning cells (OUT/cells/<cell>/); was the plan's grid set?")
    rows = []
    for cell in cells:
        scores, drops, settings, missing = {}, {}, None, []
        for condition in conditions:
            folders = sorted((cell / "runs" / "weighted" / condition).glob("seed*"))
            if not folders:
                missing.append(f"{condition}: no runs")
            for folder in folders:
                final = folder / "evaluation_final.json"
                if not final.exists():
                    missing.append(str(final))
                    continue
                s_in, run_settings = _s_in(final)
                if run_settings["purpose"] != "tuning" or run_settings["maze"] != "M0":
                    raise SystemExit(f"{folder} is not a tuning run on M0")
                if settings is None:
                    settings = run_settings
                scores.setdefault(condition, []).append(s_in)
                checkpoints = sorted(folder.glob("evaluation_checkpoint_ep*.json"),
                                     key=lambda p: int(p.stem.rsplit("ep", 1)[1]))
                pretrained = cell / "pretrained" / "weighted" / f"{folder.name}.json"
                if checkpoints and pretrained.exists():
                    early, _ = _s_in(checkpoints[0])
                    before, _ = _s_in(pretrained)
                    drops.setdefault(condition, []).append(before - early)
                elif not checkpoints:
                    missing.append(f"{folder}: no checkpoint evaluation")
                else:
                    missing.append(str(pretrained))
        values = [v for c in conditions for v in scores.get(c, [])]
        all_drops = [v for c in conditions for v in drops.get(c, [])]
        rows.append({
            "cell": cell.name, "complete": not missing, "missing": missing,
            "learning_rate": settings["learning_rate"] if settings else None,
            "updates_per_env_step": settings["updates_per_env_step"] if settings else None,
            "seeds": {c: len(scores.get(c, [])) for c in conditions},
            "S_in_by_condition": {c: (sum(v) / len(v) if v else None) for c, v in scores.items()},
            "score": sum(values) / len(values) if values else None,
            "early_drop_mean": sum(all_drops) / len(all_drops) if all_drops else None,
            "early_drop_by_condition": {c: sum(v) / len(v) for c, v in drops.items()},
        })
    incomplete = [r["cell"] for r in rows if not r["complete"]]
    if incomplete:
        raise SystemExit(f"incomplete cells, nothing chosen: {incomplete}")
    seed_counts = {tuple(sorted(r["seeds"].items())) for r in rows}
    if len(seed_counts) != 1:
        raise SystemExit("cells have different numbers of seeds; equal trials per condition is required")
    best = max(r["score"] for r in rows)
    tied = [r for r in rows if best - r["score"] <= tie + 1e-12]
    chosen = min(tied, key=lambda r: (r["updates_per_env_step"], r["learning_rate"]))
    triggered = chosen["early_drop_mean"] is not None and chosen["early_drop_mean"] > drop_threshold
    return {
        "objective": "mean greedy S_in of task-directed practice and frontier novelty over seeds (decision 2)",
        "tie_margin": tie, "early_drop_threshold": drop_threshold,
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
    args = parser.parse_args(argv)
    report = select(args.out, args.tie, args.drop_threshold)
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
