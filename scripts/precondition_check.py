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
  each side (north-shorter and south-shorter) and the mean Rep over the pairs
  kept is at least 0.25 for each learner; otherwise the pilot stops and a
  reformulation is documented. (Taking the mean over the kept pairs is Claude's
  reading, for review: the outer pairs the spec expects to fail would otherwise
  pull the six-pair mean under 0.25 almost automatically; audit of 26 Nov, SF5.)
- add-demos: add demonstrations with the same split, up to 280 on M1
  (40 per pair), and run the check again.
The script never changes settings itself.

Before it scores anything it checks its inputs (audit of 26 Nov, SF4 and MI6):
the plan the launcher ran (OUT/launch_manifest.json) gives the seeds, and both
learners must have exactly those; each file must be its folder's learner and
its name's seed, a warm-up-only precondition run on M1, list exactly M1's six
shorter-route pairs (never the tie row), and share one warm-up with every other
file apart from learner and seed. With --tuning-report, the learning rate must
be the tuned one. The report records each seed's pretrained fingerprint, which
the pilot's runs must reproduce.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from envs.mazes import RINGS  # noqa: E402
from eval.protocol import task_pairs  # noqa: E402

MEAN_THRESHOLD, PAIR_THRESHOLD = 0.25, 0.10
LEARNERS = ("weighted", "plain")


def _pairs(path, learner, seed):
    result = json.loads(Path(path).read_text())
    if result["evaluation"]["quick_smoke_only"]:
        raise SystemExit(f"{path} is a quick (smoke) evaluation; the check needs the full protocol")
    if result["run"]["which"] != "pretrained":
        raise SystemExit(f"{path} does not evaluate a pretrained policy")
    if result["evaluation"]["config"]["sampled_rollouts"] != 100:
        raise SystemExit(f"{path} did not use 100 sampled rollouts per pair")
    settings = result["run"]["settings"]
    # Audit of 26 Nov, SF4: the file is this learner's and this seed's warm-up-only run on M1.
    if (settings["purpose"] != "precondition" or settings["learner"] != learner or settings["seed"] != seed
            or settings["maze"] != "M1" or settings["budget_steps"] != 0):
        raise SystemExit(f"{path} is not the {learner} learner's seed-{seed} precondition run on M1 (budget 0)")
    rep = {}
    for pair in result["report"]["pairs"]:
        if pair["is_primary_pair"]:
            value = pair["sampled"]["Rep"]
            rep[pair["row"]] = {"Rep": 0.0 if value is None else float(value), "undefined": value is None,
                                "successes": pair["sampled"]["successes"]}
    shorter = sorted(pair.row for pair in task_pairs("M1") if pair.shorter)   # MI6: never the tie row
    if sorted(rep) != shorter:
        raise SystemExit(f"{path} lists primary pairs {sorted(rep)}, not M1's shorter-route pairs {shorter}")
    return settings, rep, result["run"]["policy_digest"]


def _warmup_without(settings):
    from training import settings_from_dict
    key = settings_from_dict(settings).warmup_key()
    return {k: v for k, v in key.items() if k not in ("learner", "seed")}


def check(out, definition="seed-mean", failure_rule="exclude", tuning_report=None):
    out = Path(out)
    manifest = out / "launch_manifest.json"
    if not manifest.exists():
        raise SystemExit(f"{manifest} is missing: the check reads the plan the launcher ran")
    plan = json.loads(manifest.read_text())["launches"][-1]["plan"]
    planned_seeds = sorted(plan["seeds"])
    learners, warmups, digests = {}, [], {}
    for learner in LEARNERS:
        files = {int(f.stem[4:]): f for f in (out / "pretrained" / learner).glob("seed*.json")}
        if sorted(files) != planned_seeds:   # SF4: exactly the plan's seeds, for both learners
            raise SystemExit(f"the {learner} learner has evaluations for seeds {sorted(files)}, "
                             f"not the plan's {planned_seeds}")
        per_seed = {}
        for seed, file in sorted(files.items()):
            settings, rep, digest = _pairs(file, learner, seed)
            warmups.append(_warmup_without(settings))
            digests.setdefault(learner, {})[seed] = digest
            per_seed[seed] = rep
        rows = sorted(next(iter(per_seed.values())))
        bars = {r: sum(per_seed[s][r]["Rep"] for s in per_seed) / len(per_seed) for r in rows}
        if definition == "seed-mean":
            failing = [r for r in rows if bars[r] < PAIR_THRESHOLD]
            mean = sum(bars.values()) / len(bars)
            mean_ok = mean >= MEAN_THRESHOLD
        else:
            failing = sorted({r for s in per_seed for r in rows if per_seed[s][r]["Rep"] < PAIR_THRESHOLD})
            means = {s: sum(per_seed[s][r]["Rep"] for r in rows) / len(rows) for s in per_seed}
            mean, mean_ok = min(means.values()), all(m >= MEAN_THRESHOLD for m in means.values())
        learners[learner] = {
            "seeds": len(per_seed), "Rep_by_pair": bars, "mean_Rep": mean, "mean_ok": mean_ok,
            "pairs_below_floor": failing, "passed": mean_ok and not failing,
            "undefined_counted_as_zero": {f"seed{s}": [r for r in rows if per_seed[s][r]["undefined"]]
                                          for s in per_seed},
        }
    if any(w != warmups[0] for w in warmups):   # SF4: one warm-up, apart from learner and seed
        raise SystemExit("the evaluations come from different warm-up settings (learning rate, batch, D0 size "
                         "or warm-up rule differ)")
    if tuning_report is not None:
        chosen = json.loads(Path(tuning_report).read_text())["chosen"]
        if warmups[0]["learning_rate"] != chosen["learning_rate"]:
            raise SystemExit(f"the precondition runs used learning rate {warmups[0]['learning_rate']}, "
                             f"not the tuned {chosen['learning_rate']}")
    passed = all(v["passed"] for v in learners.values())
    tie = RINGS["M1"]["tie_row"]
    failing = sorted({r for v in learners.values() for r in v["pairs_below_floor"]})
    all_rows = sorted(next(iter(learners.values()))["Rep_by_pair"])
    remaining = [r for r in all_rows if r not in failing]
    north, south = [r for r in remaining if r < tie], [r for r in remaining if r > tie]
    # SF5 (a decision for Siddharth): under exclusion the mean is taken over the pairs kept, so excluding the
    # outer pairs that the spec expects to fail does not also fail the mean automatically.
    kept_means = {learner: (sum(v["Rep_by_pair"][r] for r in remaining) / len(remaining) if remaining else 0.0)
                  for learner, v in learners.items()}
    exclude_ok = len(north) >= 2 and len(south) >= 2 and all(m >= MEAN_THRESHOLD for m in kept_means.values())
    rules = {
        "exclude": {"pairs_excluded_from_primary": failing, "primary_pairs_left": {"north": north, "south": south},
                    "mean_Rep_over_kept_pairs": kept_means,
                    "outcome": ("proceed to the pilot with these pairs excluded from the primary endpoints"
                                if exclude_ok else "stop the pilot and document a reformulation")},
        "add-demos": {"outcome": "add demonstrations with the same split, up to 280 on M1 (40 per pair), "
                                 "then run the check again"},
    }
    return {
        "definition": definition, "thresholds": {"mean": MEAN_THRESHOLD, "every_pair": PAIR_THRESHOLD},
        "maze": "M1", "seeds": planned_seeds, "warmup": warmups[0], "learners": learners, "passed": passed,
        # SF4: the pilot's runs must start from these exact policies (warm-up is deterministic).
        "pretrained_digests": {l: {f"seed{s}": d for s, d in v.items()} for l, v in digests.items()},
        "next": "proceed to the pilot" if passed else rules[failure_rule]["outcome"],
        "failure_rule": failure_rule, "failure_rules": None if passed else rules,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--definition", choices=("seed-mean", "per-seed"), default="seed-mean")
    parser.add_argument("--failure-rule", choices=("exclude", "add-demos"), default="exclude")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--tuning-report", type=Path, help="refuse runs whose learning rate is not the tuned one")
    args = parser.parse_args(argv)
    report = check(args.out, args.definition, args.failure_rule, args.tuning_report)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    for learner, v in report["learners"].items():
        pairs = ", ".join(f"row {r}: {x:.2f}" for r, x in sorted(v["Rep_by_pair"].items()))
        print(f"{learner}: mean Rep {v['mean_Rep']:.3f} ({pairs}) -> {'pass' if v['passed'] else 'FAIL'}")
    print("precondition check", "passed" if report["passed"] else "FAILED", "-", report["next"])
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
