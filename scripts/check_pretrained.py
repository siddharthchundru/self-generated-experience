"""Check that every run in a seed started from the same pretrained policy.

Run: python scripts/check_pretrained.py ROOT [--out REPORT.json]

Looks at every run folder under ROOT (training and re-learning runs). Training
runs with the same warm-up settings must have recorded the same pretrained
fingerprint, and the pretrained.pt on disk must still match it; runs that share
a seed, maze, learner and purpose must share their warm-up settings and their
commit. Re-learning runs with --init same must reproduce their source's
pretrained policy. See `src/runs.py` for the reasons.

Exits with status 1 if any problem is found. Unfinished runs are listed but do
not fail the check. The launcher runs this check after every launch.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from runs import check_pretrained  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, help="also write the report here")
    parser.add_argument("--no-file-check", action="store_true",
                        help="trust the recorded fingerprints without reloading pretrained.pt")
    args = parser.parse_args(argv)
    report = check_pretrained(args.root, verify_files=not args.no_file_check)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['training_runs']} training runs, {report['relearning_runs']} re-learning runs, "
          f"{len(report['warmup_groups'])} warm-up groups, {len(report['unfinished'])} unfinished")
    for line in report["warnings"]:
        print("warning:", line)
    for line in report["problems"]:
        print("PROBLEM:", line)
    print("passed" if report["passed"] else "FAILED")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
