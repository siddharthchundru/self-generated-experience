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
Runs that ran different code fail the check too, because the practice code
could differ even when warm-up matched. Code is compared by the source hashes
each run records for src/ and the run scripts; the commit is only a label, so
a commit that touches only notes does not fail a resumed launch, and two dirty
runs with different code do fail (audit of 24 Nov, MI1). A pilot or main run
with no source hashes is a problem. Re-learners in a seed with the same init
must share one pretrained policy too (MI2). Runs from a dirty working tree
(exploratory, `--allow-dirty`) are listed as warnings, as are runs of
different purposes in one seed that started from different policies (MI3).
An unreadable file is reported as a problem, never a crash (SF3). Unfinished
runs are listed, not failed: whether a crashed run matters is the launcher's
question.
"""

from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import re

import numpy as np
import torch

from gcsl import GCSLPolicy, TrajectoryBuffer
from training import settings_from_dict

WEIGHTS = {"pretrained": "pretrained.pt", "final": "final.pt"}
#: Settings a tuning grid varies (spec section 7): runs that differ in them are different cells, not a mismatch.


def grid_cell(folder):
    """The launch-grid cell a run folder belongs to (the label after "cells" in its path), or ""."""
    parts = Path(folder).resolve().parts
    return parts[parts.index("cells") + 1] if "cells" in parts[:-1] else ""


def is_finished(folder):
    return (Path(folder) / "summary.json").exists()


def load_run(folder):
    """(settings, summary, run_record) of a finished run folder."""
    folder = Path(folder)
    if not is_finished(folder):
        raise FileNotFoundError(f"{folder} has no summary.json: the run is unfinished or crashed")
    settings = settings_from_dict(json.loads((folder / "settings.json").read_text()))
    summary = json.loads((folder / "summary.json").read_text())
    if summary.get("settings") != asdict(settings):  # audit of 24 Nov, SF1
        raise ValueError(f"{folder}: settings.json differs from the settings the run recorded in summary.json")
    record_file = folder / "run_record.json"
    record = json.loads(record_file.read_text()) if record_file.exists() else {}
    return settings, summary, record


def code_hashes(record):
    """The run's source hashes for the code that decides training (src/ and the run
    scripts), used to tell whether two runs ran the same code (audit of 24 Nov, MI1)."""
    hashes = record.get("source_sha256") or {}
    return {path: digest for path, digest in hashes.items()
            if path.startswith("src/") or path in ("scripts/run_training.py", "scripts/run_relearning.py")}


CHECKPOINT = re.compile(r"^checkpoint_ep(\d+)$")


def recorded_digest(summary, which):
    """The fingerprint summary.json records for `which`: "pretrained", "final" or
    "checkpoint_ep<N>" (a weights snapshot kept after N practice episodes)."""
    match = CHECKPOINT.match(which)
    if match:
        digest = (summary.get("checkpoint_digests") or {}).get(match.group(1))
        if digest is None:
            raise ValueError(f"the run kept no {which}")
        return digest
    if which not in WEIGHTS:
        raise ValueError(f"which must be one of {tuple(WEIGHTS)} or checkpoint_ep<N>")
    return summary[f"{which}_digest"]


def load_policy(folder, which="final"):
    """The policy saved as `which` ("pretrained", "final" or "checkpoint_ep<N>"),
    checked against the fingerprint recorded in summary.json."""
    folder = Path(folder)
    _, summary, _ = load_run(folder)
    expected = recorded_digest(summary, which)
    path = folder / (WEIGHTS[which] if which in WEIGHTS else f"{which}.pt")
    policy = GCSLPolicy(0, name=f"{folder.name}:{which}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    policy.network.load_state_dict(state)
    if policy.parameter_digest() != expected:
        raise ValueError(f"{path} does not match the fingerprint in summary.json")
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
    key and the same code). Re-learning runs with init "same" must reproduce
    their source's pretrained fingerprint; re-learners sharing seed, maze,
    learner, purpose and init must share one pretrained fingerprint.
    """
    if (root is None) == (folders is None):
        raise ValueError("give either a root folder or a list of run folders")
    problems, warnings, unfinished, runs = [], [], [], []
    for folder in (find_runs(root) if folders is None else [Path(f) for f in folders]):
        if not is_finished(folder):
            unfinished.append(str(folder))
            continue
        try:  # an unreadable file is a problem to report, not a crash (audit of 24 Nov, SF3)
            settings, summary, record = load_run(folder)
            if verify_files:
                load_policy(folder, "pretrained")
        except Exception as error:  # noqa: BLE001
            problems.append(f"{folder}: could not be checked ({type(error).__name__}: {error})")
            continue
        entry = {"folder": str(folder), "kind": summary.get("kind", "training"),
                 "condition": settings.condition, "seed": settings.seed, "maze": settings.maze,
                 "learner": settings.learner, "purpose": settings.purpose,
                 "warmup_key": json.dumps(settings.warmup_key(), sort_keys=True),
                 "pretrained_digest": summary["pretrained_digest"],
                 "commit": record.get("git_commit"), "dirty": record.get("working_tree_dirty"),
                 "code": json.dumps(code_hashes(record), sort_keys=True), "init": summary.get("init"),
                 # A launch grid (tuning, or a smoke test) varies settings on purpose: each cell, a folder
                 # OUT/cells/<label>/, is its own batch. Runs outside a cell are never split this way.
                 "cell": grid_cell(folder)}
        if entry["kind"] == "training" and summary.get("warmup_key") != settings.warmup_key():
            problems.append(f"{folder}: summary's warm-up key differs from its settings")
        if summary["budget"].get("passed") is not True:
            problems.append(f"{folder}: budget audit did not pass")
        if not code_hashes(record):
            message = f"{folder}: no source hashes recorded, so its code cannot be compared"
            (problems if settings.purpose in ("pilot", "main") else warnings).append(message)
        if entry["kind"] == "relearning" and entry["init"] == "same" \
                and summary["pretrained_digest"] != summary["source"]["pretrained_digest"]:
            problems.append(f"{folder}: init='same' re-learner did not reproduce its source's pretrained policy")
        runs.append(entry)

    training = [r for r in runs if r["kind"] == "training"]
    by_key, by_seed, by_seed_any_purpose, relearners = (defaultdict(list) for _ in range(4))
    for r in training:
        by_key[r["warmup_key"]].append(r)
        by_seed[(r["seed"], r["maze"], r["learner"], r["purpose"], r["cell"])].append(r)
        by_seed_any_purpose[(r["seed"], r["maze"], r["learner"])].append(r)
    for r in runs:
        if r["kind"] == "relearning":  # audit of 24 Nov, MI2: re-learners in a seed are paired too
            relearners[(r["seed"], r["maze"], r["learner"], r["purpose"], r["cell"], r["init"])].append(r)
    groups = []
    for key, members in by_key.items():
        digests = sorted({m["pretrained_digest"] for m in members})
        groups.append({"warmup_key": json.loads(key), "runs": len(members), "pretrained_digests": digests,
                       "conditions": sorted(m["condition"] for m in members)})
        if len(digests) != 1:
            problems.append(f"runs with warm-up key {key} recorded {len(digests)} different pretrained "
                            f"policies: {[m['folder'] for m in members]}")
    for (seed, maze, learner, purpose, cell), members in by_seed.items():
        label = f"seed {seed}, {maze}, {learner} learner, {purpose}" + (f", grid cell {cell}" if cell else "")
        if len({m["warmup_key"] for m in members}) != 1:
            problems.append(f"{label}: runs differ in warm-up settings, so they did not start from the same "
                            f"policy: {[m['folder'] for m in members]}")
        if any(m["dirty"] for m in members):
            warnings.append(f"{label}: at least one run was made from a dirty working tree (exploratory)")
    # Code is compared by source hashes over each purpose's whole batch, across tuning cells too (audit of
    # 26 Nov, SF1); the commit is only a label (audit of 24 Nov, MI1).
    by_code = defaultdict(list)
    for r in training:
        by_code[(r["seed"], r["maze"], r["learner"], r["purpose"])].append(r)
    for (seed, maze, learner, purpose), members in by_code.items():
        if len({m["code"] for m in members}) != 1:
            problems.append(f"seed {seed}, {maze}, {learner} learner, {purpose}: runs ran different code "
                            f"(source hashes differ; commits {sorted({str(m['commit']) for m in members})})")
    for (seed, maze, learner), members in by_seed_any_purpose.items():
        if len({m["purpose"] for m in members}) > 1 and len({m["warmup_key"] for m in members}) > 1:
            warnings.append(f"seed {seed}, {maze}, {learner} learner: runs of different purposes "
                            f"({sorted({m['purpose'] for m in members})}) started from different policies; "
                            "they are not paired with each other")  # audit of 24 Nov, MI3
    for (seed, maze, learner, purpose, _, init), members in relearners.items():
        if len({m["pretrained_digest"] for m in members}) != 1:
            problems.append(f"seed {seed}, {maze}, {learner} learner, {purpose}: re-learners with init "
                            f"'{init}' started from different pretrained policies: {[m['folder'] for m in members]}")
    return {"passed": not problems, "problems": problems, "warnings": warnings, "unfinished": unfinished,
            "training_runs": len(training), "relearning_runs": len(runs) - len(training),
            "warmup_groups": groups}
