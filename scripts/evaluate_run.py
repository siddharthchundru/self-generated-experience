"""Evaluate one finished run's policy with the evaluation suite (spec section 8).

Run: python scripts/evaluate_run.py --run RUN_FOLDER [--which final|pretrained]

Everything the suite needs comes from the run folder, not from flags: the
policy is rebuilt from final.pt (or pretrained.pt) and checked against the
fingerprint in summary.json; the evaluation settings and the common adaptation
procedure are built from the run's settings.json with
`training.evaluation_config_for` and `training.adapter_for`, so the adapter's
gamma, learning rate, batch size, update ratio and k are the run's, and the
suite refuses any mismatch. Works for training runs and re-learning runs alike.

Measures (the suite's, unchanged): S_in, Rep with its minimum-success rule,
minority-route success rate, per-route quality, A_0 and best-of-m under the P1
block, and A_k after the common adaptation procedure under P1 and P1-own.

Options:
- --which checkpoint_ep<N> evaluates the weights a run kept after N practice
  episodes (run_training.py --checkpoints), for the early-drop check in tuning.
- --which pretrained evaluates the policy after warm-up. Every condition in a
  seed shares it, so evaluate it from one run per seed and learner (the
  launcher does).
- --skip-adaptation leaves out the two adaptation runs (A_k is reported as not
  run). For the precondition check, which needs Rep of the pretrained policy
  only.
- --quick shrinks rollout counts and k for a smoke test. It changes the
  measures (Rep's minimum successes, k), so it is refused for every purpose
  except smoke, and marked in the output.

Writes OUT (default RUN_FOLDER/evaluation_<which>.json; never overwritten),
holding this script's run record, the run's identity and the suite's full
report. Writing into results/ needs a clean, committed tree.
"""

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval.suite import evaluate_policy  # noqa: E402
from episode import EPISODE_STEPS  # noqa: E402
from gcsl import pin_single_thread  # noqa: E402
from provenance import guard_results_write, run_record  # noqa: E402
from runs import load_policy, load_run, recorded_digest  # noqa: E402
from training import K_PLACEHOLDER, adapter_for, evaluation_config_for  # noqa: E402

QUICK = {"sampled_rollouts": 10, "minimum_successes": 2, "greedy_rollouts": 2, "bootstrap_resamples": 200,
         "adaptation_steps": EPISODE_STEPS}


def build(settings, quick=False, skip_adaptation=False):
    """(evaluation config, adapter or None) for a run's settings."""
    if quick and settings.purpose != "smoke":  # audit of 24 Nov, SF4: --quick changes the measures
        raise SystemExit("--quick is for smoke runs only; every other purpose uses the full protocol")
    if settings.purpose in ("pilot", "main") and settings.adaptation_steps == K_PLACEHOLDER:
        raise SystemExit("k = 960 is a placeholder; set k by the spec section 8 rule before the pilot")
    config = evaluation_config_for(settings)
    if quick:
        config = replace(config, **QUICK)
    adapter = None if skip_adaptation else adapter_for(settings)
    return config, adapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--which", default="final", help="final, pretrained or checkpoint_ep<N>")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--skip-adaptation", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)

    settings, summary, source_record = load_run(args.run)
    out = args.out or args.run / f"evaluation_{args.which}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; evaluations are never overwritten")
    config, adapter = build(settings, args.quick, args.skip_adaptation)
    policy = load_policy(args.run, args.which)
    evaluated_digest = policy.parameter_digest()  # of the policy handed to the suite (audit of 24 Nov, M8)
    if evaluated_digest != recorded_digest(summary, args.which):
        raise AssertionError("the policy about to be evaluated is not the run's")
    record = run_record(ROOT)
    guard_results_write(ROOT, out, args.allow_dirty)
    import torch
    record["allow_dirty"] = args.allow_dirty
    record["torch_threads"] = {"intra_op": torch.get_num_threads(), "inter_op": torch.get_num_interop_threads()}

    report = evaluate_policy(policy, config, adapter)
    if report["parameter_digest"] is None:
        raise AssertionError("a learned policy must be fingerprinted by the suite")
    result = {
        **record,
        "run": {"folder": str(args.run.resolve()), "kind": summary.get("kind", "training"),
                "init": summary.get("init"), "which": args.which, "settings": asdict(settings),
                "policy_digest": evaluated_digest,
                "training_commit": source_record.get("git_commit"),
                "same_commit_as_training": source_record.get("git_commit") == record["git_commit"]},
        "evaluation": {"quick_smoke_only": args.quick, "adaptation_run": adapter is not None,
                       "config": asdict(config)},
        "report": report,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(f"{out.name}.{os.getpid()}.partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    try:  # appears whole or not at all, and never replaces an existing file (audit of 24 Nov, SF3)
        os.link(partial, out)
    except FileExistsError:
        raise SystemExit(f"{out} was written by another evaluation meanwhile; this one is discarded") from None
    finally:
        partial.unlink()
    aggregate = report["aggregate"]
    print(json.dumps({"run": str(args.run), "which": args.which, "S_in": aggregate["S_in"],
                      "Rep": aggregate["Rep"], "A_0": aggregate["A_0"], "A_k": aggregate["A_k"],
                      "seconds": round(report["elapsed_seconds"], 1)}))


if __name__ == "__main__":
    pin_single_thread()
    main()
