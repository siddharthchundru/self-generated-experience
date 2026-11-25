"""The precondition check on the pretrained policies (spec section 7; Stage 2 gate).

Run: python scripts/precondition_check.py OUT [--report REPORT.json]

OUT is a finished launch of purpose "precondition": for every seed, warm-up only
(budget 0) for both learners, with the pretrained policy evaluated without
adaptation (plans/precondition_m1.json). This script reads those evaluations; it
runs nothing.

Why it matters. Repertoire collapse presupposes that both routes were there to
begin with. If the pretrained policy already takes one route almost always, a
null on H2 would only mean there was nothing to preserve (spec section 3).

The rule. Rep is the minority-route share among a pair's successful sampled
rollouts (0 to 0.5). The thresholds are the spec's: mean Rep at least 0.25 over
the shorter-route pairs, and Rep at least 0.10 on every shorter-route pair, for
each learner. The definition (decision 6 of the 4 Nov audit, open; Claude's
recommendation is the default) is per pair, averaged across the pilot seeds:

    Rep_bar(p) = (1 / S) x sum over seeds s of Rep(p, s)

where S is the number of seeds and Rep(p, s) comes from 100 sampled rollouts on
pair p. The check passes when the mean of Rep_bar(p) over the shorter-route
pairs is at least 0.25 and every Rep_bar(p) is at least 0.10. The alternative
("per-seed", --definition per-seed) applies both thresholds to every seed
separately, 6 seeds x 6 pairs x 2 learners = 72 checks, which can fail by noise.

A pair whose Rep is undefined in a seed (fewer than 20 successes out of 100)
counts as 0 there (Claude's choice, for review): a pretrained policy that cannot
reach the goal on a pair has shown no repertoire there, and failing safe is
better than averaging over the seeds where it happened to succeed.

If the check fails (decision 7, open): the report says what each pre-specified
failure rule would do, and marks the one chosen with --failure-rule (default
"exclude", Claude's lean):
- exclude: a shorter-route pair below 0.10 for either learner leaves the primary
  endpoints (still reported as secondary), provided at least two pairs remain on
  each side (north-shorter and south-shorter); otherwise the pilot stops and a
  reformulation is documented. A failing mean is not fixed by exclusion: it
  stops the pilot.
- add-demos: add demonstrations with the same split, up to 280 on M1
  (40 per pair), and run the check again.
The script never changes settings itself.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from envs.mazes import RINGS  # noqa: E402

MEAN_THRESHOLD, PAIR_THRESHOLD = 0.25, 0.10
LEARNERS = ("weighted", "plain")


def _pairs(path):
    result = json.loads(Path(path).read_text())
    if result["evaluation"]["quick_smoke_only"]:
        raise SystemExit(f"{path} is a quick (smoke) evaluation; the check needs the full protocol")
    if result["run"]["which"] != "pretrained":
        raise SystemExit(f"{path} does not evaluate a pretrained policy")
    if result["evaluation"]["config"]["sampled_rollouts"] != 100:
        raise SystemExit(f"{path} did not use 100 sampled rollouts per pair")
    settings = result["run"]["settings"]
    rep = {}
    for pair in result["report"]["pairs"]:
        if pair["is_primary_pair"]:
            value = pair["sampled"]["Rep"]
            rep[pair["row"]] = {"Rep": 0.0 if value is None else float(value), "undefined": value is None,
                                "successes": pair["sampled"]["successes"]}
    return settings, rep


def check(out, definition="seed-mean", failure_rule="exclude"):
    out = Path(out)
    learners, problems = {}, []
    maze = None
    for learner in LEARNERS:
        files = sorted((out / "pretrained" / learner).glob("seed*.json"))
        if not files:
            problems.append(f"no pretrained evaluations for the {learner} learner")
            continue
        per_seed = {}
        for file in files:
            settings, rep = _pairs(file)
            if settings["purpose"] != "precondition":
                raise SystemExit(f"{file} comes from a {settings['purpose']} run, not a precondition run")
            maze = maze or settings["maze"]
            per_seed[file.stem] = rep
        rows = sorted(next(iter(per_seed.values())))
        if definition == "seed-mean":
            bars = {r: sum(per_seed[s][r]["Rep"] for s in per_seed) / len(per_seed) for r in rows}
            failing = [r for r in rows if bars[r] < PAIR_THRESHOLD]
            mean = sum(bars.values()) / len(bars)
            mean_ok = mean >= MEAN_THRESHOLD
        else:
            failing = sorted({r for s in per_seed for r in rows if per_seed[s][r]["Rep"] < PAIR_THRESHOLD})
            means = {s: sum(per_seed[s][r]["Rep"] for r in rows) / len(rows) for s in per_seed}
            mean, mean_ok = min(means.values()), all(m >= MEAN_THRESHOLD for m in means.values())
            bars = {r: sum(per_seed[s][r]["Rep"] for s in per_seed) / len(per_seed) for r in rows}
        learners[learner] = {
            "seeds": len(per_seed), "Rep_by_pair": bars, "mean_Rep": mean, "mean_ok": mean_ok,
            "pairs_below_floor": failing, "passed": mean_ok and not failing,
            "undefined_counted_as_zero": {s: [r for r in rows if per_seed[s][r]["undefined"]] for s in per_seed},
        }
    if problems:
        raise SystemExit("; ".join(problems))
    if len({v["seeds"] for v in learners.values()}) != 1:
        raise SystemExit("the two learners were evaluated on different numbers of seeds")
    passed = all(v["passed"] for v in learners.values())
    tie = RINGS[maze]["tie_row"]
    failing = sorted({r for v in learners.values() for r in v["pairs_below_floor"]})
    all_rows = sorted(next(iter(learners.values()))["Rep_by_pair"])
    remaining = [r for r in all_rows if r not in failing]
    north, south = [r for r in remaining if r < tie], [r for r in remaining if r > tie]
    means_ok = all(v["mean_ok"] for v in learners.values())
    exclude_ok = means_ok and len(north) >= 2 and len(south) >= 2
    rules = {
        "exclude": {"pairs_excluded_from_primary": failing, "primary_pairs_left": {"north": north, "south": south},
                    "outcome": ("proceed to the pilot with these pairs excluded from the primary endpoints"
                                if exclude_ok else "stop the pilot and document a reformulation")},
        "add-demos": {"outcome": "add demonstrations with the same split, up to 280 on M1 (40 per pair), "
                                 "then run the check again"},
    }
    return {
        "definition": definition, "thresholds": {"mean": MEAN_THRESHOLD, "every_pair": PAIR_THRESHOLD},
        "maze": maze, "learners": learners, "passed": passed,
        "next": "proceed to the pilot" if passed else rules[failure_rule]["outcome"],
        "failure_rule": failure_rule, "failure_rules": None if passed else rules,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--definition", choices=("seed-mean", "per-seed"), default="seed-mean")
    parser.add_argument("--failure-rule", choices=("exclude", "add-demos"), default="exclude")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = check(args.out, args.definition, args.failure_rule)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    for learner, v in report["learners"].items():
        pairs = ", ".join(f"row {r}: {x:.2f}" for r, x in sorted(v["Rep_by_pair"].items()))
        print(f"{learner}: mean Rep {v['mean_Rep']:.3f} ({pairs}) -> {'pass' if v['passed'] else 'FAIL'}")
    print("precondition check", "passed" if report["passed"] else "FAILED", "-", report["next"])
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
