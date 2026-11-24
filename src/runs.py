"""Reading finished run folders, and checking runs against each other.

What this file does, in one sentence: it loads what `run_training.py` and
`run_relearning.py` write (settings, summary, policy weights, final buffer),
checks each file against the fingerprints the run recorded, and checks that
every run sharing a warm-up started from the same pretrained policy.

A run folder is finished when summary.json exists: the scripts write it last.
The weights in pretrained.pt and final.pt are loaded into a fresh policy and
fingerprinted again, so a file swapped or edited after the run is refused
rather than evaluated.

The pretrained-policy check (spec section 7, paired seeds). Warm-up depends
only on the settings in `TrainingSettings.warmup_key()` (seed, maze, learner,
learning rate, batch size, D0 size and the warm-up rule), never on the
condition. So every training run with the same warm-up key must record the
same pretrained fingerprint, and the file on disk must match it. Runs in one
launch that share a seed, maze and learner must also share a warm-up key:
otherwise the conditions in that seed did not start from the same policy, and
a paired comparison between them would compare different starting points.
Runs made from different commits fail the check too, because the practice
code could differ even when warm-up matched. Runs from a dirty working tree
(exploratory, `--allow-dirty`) are listed as warnings. Unfinished runs are
listed, not failed: whether a crashed run matters is the launcher's question.
"""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from gcsl import GCSLPolicy, TrajectoryBuffer
from training import settings_from_dict

WEIGHTS = {"pretrained": "pretrained.pt", "final": "final.pt"}


def is_finished(folder):
    return (Path(folder) / "summary.json").exists()


def load_run(folder):
    """(settings, summary, run_record) of a finished run folder."""
    folder = Path(folder)
    if not is_finished(folder):
        raise FileNotFoundError(f"{folder} has no summary.json: the run is unfinished or crashed")
    settings = settings_from_dict(json.loads((folder / "settings.json").read_text()))
    summary = json.loads((folder / "summary.json").read_text())
    record_file = folder / "run_record.json"
    record = json.loads(record_file.read_text()) if record_file.exists() else {}
    return settings, summary, record


def load_policy(folder, which="final"):
    """The policy saved as `which` ("pretrained" or "final"), checked against the
    fingerprint recorded in summary.json."""
    if which not in WEIGHTS:
        raise ValueError(f"which must be one of {tuple(WEIGHTS)}")
    folder = Path(folder)
    _, summary, _ = load_run(folder)
    policy = GCSLPolicy(0, name=f"{folder.name}:{which}")
    state = torch.load(folder / WEIGHTS[which], map_location="cpu", weights_only=True)
    policy.network.load_state_dict(state)
    if policy.parameter_digest() != summary[f"{which}_digest"]:
        raise ValueError(f"{folder / WEIGHTS[which]} does not match the fingerprint in summary.json")
    return policy


def load_buffer(folder):
    """The run's final buffer, checked against the digest in summary.json."""
    folder = Path(folder)
    _, summary, _ = load_run(folder)
    with np.load(folder / "final_buffer.npz") as arrays:
        buffer = TrajectoryBuffer.from_arrays({k: arrays[k] for k in arrays.files})
    if buffer.digest() != summary["final_buffer_digest"]:
        raise ValueError(f"{folder / 'final_buffer.npz'} does not match the digest in summary.json")
    return buffer


def find_runs(root):
    """Every run folder under `root` (a folder holding settings.json), finished or not."""
    return sorted(p.parent for p in Path(root).rglob("settings.json"))


def check_pretrained(root=None, verify_files=True, folders=None):
    """Check that runs sharing a warm-up share a pretrained policy, for every run
    under `root` or for the run folders listed in `folders`. Returns a report;
    `report["passed"]` is False if any problem was found.

    Training runs are grouped by warm-up key (must share one pretrained
    fingerprint) and by (seed, maze, learner, purpose) (must share one warm-up
    key and one commit). Re-learning runs with init "same" must reproduce their
    source's pretrained fingerprint; "fresh" re-learners are listed but not
    grouped, since they start from different weights by design.
    """
    if (root is None) == (folders is None):
        raise ValueError("give either a root folder or a list of run folders")
    problems, warnings, unfinished, runs = [], [], [], []
    for folder in (find_runs(root) if folders is None else [Path(f) for f in folders]):
        if not is_finished(folder):
            unfinished.append(str(folder))
            continue
        settings, summary, record = load_run(folder)
        entry = {"folder": str(folder), "kind": summary.get("kind", "training"),
                 "condition": settings.condition, "seed": settings.seed, "maze": settings.maze,
                 "learner": settings.learner, "purpose": settings.purpose,
                 "warmup_key": json.dumps(settings.warmup_key(), sort_keys=True),
                 "pretrained_digest": summary["pretrained_digest"],
                 "commit": record.get("git_commit"), "dirty": record.get("working_tree_dirty"),
                 "init": summary.get("init")}
        if summary.get("warmup_key") != settings.warmup_key():
            problems.append(f"{folder}: summary's warm-up key differs from its settings")
        if summary["budget"].get("passed") is not True:
            problems.append(f"{folder}: budget audit did not pass")
        if verify_files:
            try:
                load_policy(folder, "pretrained")
            except ValueError as error:
                problems.append(str(error))
        if entry["kind"] == "relearning" and entry["init"] == "same" \
                and summary["pretrained_digest"] != summary["source"]["pretrained_digest"]:
            problems.append(f"{folder}: init='same' re-learner did not reproduce its source's pretrained policy")
        runs.append(entry)

    training = [r for r in runs if r["kind"] == "training"]
    by_key, by_seed = defaultdict(list), defaultdict(list)
    for r in training:
        by_key[r["warmup_key"]].append(r)
        by_seed[(r["seed"], r["maze"], r["learner"], r["purpose"])].append(r)
    groups = []
    for key, members in by_key.items():
        digests = sorted({m["pretrained_digest"] for m in members})
        groups.append({"warmup_key": json.loads(key), "runs": len(members), "pretrained_digests": digests,
                       "conditions": sorted(m["condition"] for m in members)})
        if len(digests) != 1:
            problems.append(f"runs with warm-up key {key} recorded {len(digests)} different pretrained "
                            f"policies: {[m['folder'] for m in members]}")
    for (seed, maze, learner, purpose), members in by_seed.items():
        label = f"seed {seed}, {maze}, {learner} learner, {purpose}"
        if len({m["warmup_key"] for m in members}) != 1:
            problems.append(f"{label}: runs differ in warm-up settings, so they did not start from the same "
                            f"policy: {[m['folder'] for m in members]}")
        if len({m["commit"] for m in members}) != 1:
            problems.append(f"{label}: runs come from different commits: "
                            f"{sorted({str(m['commit']) for m in members})}")
        if any(m["dirty"] for m in members):
            warnings.append(f"{label}: at least one run was made from a dirty working tree (exploratory)")
    return {"passed": not problems, "problems": problems, "warnings": warnings, "unfinished": unfinished,
            "training_runs": len(training), "relearning_runs": len(runs) - len(training),
            "warmup_groups": groups}
