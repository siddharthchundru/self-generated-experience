"""Compare independently supplied human labels with the classifier, after labelling.

Run: python scripts/check_route_labels.py --answers-file /path/to/answer-line.txt
The answer key is computed from the exact rounded positions displayed in the
sheet, not from unseen original positions. Uncertain labels are reported
separately, never silently counted as agreement. No labels are invented.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from routes import trajectory_route


def parse_answers(text, expected_ids):
    labels = {}
    for token in text.split():
        match = re.fullmatch(r"(\d+)([NSX])(\?)?", token)
        if not match:
            raise ValueError(f"invalid or unfinished label: {token}")
        ident = int(match[1])
        if ident in labels:
            raise ValueError(f"duplicate label {ident}")
        labels[ident] = {"route": {"N": "north", "S": "south", "X": "neither"}[match[2]],
                         "unsure": bool(match[3])}
    if set(labels) != set(expected_ids):
        raise ValueError(f"label IDs differ from sheet: missing={sorted(set(expected_ids)-set(labels))}; "
                         f"extra={sorted(set(labels)-set(expected_ids))}")
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results/stage1/route_handcheck/human_comparison.json")
    args = parser.parse_args()
    data_file = ROOT / "results/stage1/route_handcheck/trajectories.json"
    data = json.loads(data_file.read_text())
    text = args.answers_file.read_text().strip()
    human = parse_answers(text, [d["id"] for d in data])
    comparisons = [{"id": d["id"], "human": human[d["id"]]["route"],
                    "unsure": human[d["id"]]["unsure"],
                    "classifier": trajectory_route(d["positions"])} for d in data]
    certain = [d for d in comparisons if not d["unsure"]]
    disagreements = [d for d in certain if d["human"] != d["classifier"]]
    result = {"trajectory_sha256": hashlib.sha256(data_file.read_bytes()).hexdigest(),
              "classifier_source_sha256": hashlib.sha256((ROOT / "src/routes.py").read_bytes()).hexdigest(),
              "maze_source_sha256": hashlib.sha256((ROOT / "src/envs/mazes.py").read_bytes()).hexdigest(),
              "human_answer_line": text, "labelled": len(data), "certain": len(certain),
              "uncertain_ids": [d["id"] for d in comparisons if d["unsure"]],
              "certain_agreement": (len(certain)-len(disagreements))/len(certain) if certain else None,
              "certain_disagreements": disagreements,
              "confusion_counts": dict(Counter(f"{d['human']} -> {d['classifier']}" for d in certain)),
              "comparisons": comparisons,
              "ready_for_review": len(certain) >= 50 and not disagreements,
              "status": "review_required_before_classifier_PR"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("comparisons", "human_answer_line")}, indent=2))


if __name__ == "__main__":
    main()
