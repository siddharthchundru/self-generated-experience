"""Record where a result came from, and refuse to file results from a dirty tree.

What this file does, in one sentence: `run_record(root)` returns the commit,
whether the working tree had uncommitted changes, a hash of every source file,
the machine and the package versions; `guard_results_write` stops a script
before it writes into `results/` unless the tree is clean and committed.

Why it matters (audit of 28 Oct, finding S4): Stage 1 results recorded a commit
that was never pushed, from a tree with uncommitted edits. The source hashes
showed the numbers were still trustworthy, but that is not a pattern to repeat
in Stage 2, where every result must be traceable to code a reader can check
out. A deliberate exploratory run can still write into `results/` with an
explicit `--allow-dirty`, and the record then says the tree was dirty.

"Dirty" means uncommitted changes to anything *outside* `results/`: results
written by earlier runs from the same clean commit do not count, or every run
after the first would be refused (audit of 30 Oct, SF1). Scripts take the
record once, at start, so it describes the code that actually ran.

Also fixes audit finding M11: outside a git checkout the record says so
(commit None) instead of crashing; writing into `results/` then needs
`--allow-dirty`, because the commit cannot be recorded.
"""

import hashlib
import importlib.metadata
from pathlib import Path
import platform
import subprocess

PACKAGES = ("numpy", "gymnasium", "gymnasium-robotics", "mujoco", "torch")


def git_state(root: Path) -> dict:
    """Commit hash and dirty flag of the checkout at `root`; both None outside git."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                         stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "status", "--porcelain", "--", ".",
                                          ":(exclude,icase)results"], cwd=root, text=True,
                                         stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_commit": None, "working_tree_dirty": None}
    return {"git_commit": commit, "working_tree_dirty": bool(status.strip())}


def _package_versions() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def run_record(root: Path) -> dict:
    """Everything needed to tie a result file to the code and machine that made it."""
    return {
        **git_state(root),
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for folder in (root / "src", root / "scripts")
                          for p in sorted(folder.rglob("*.py"))},
        "machine": {"system": platform.system(), "release": platform.release(),
                    "architecture": platform.machine(), "python": platform.python_version()},
        "packages": _package_versions(),
    }


def guard_results_write(root: Path, output: Path, allow_dirty: bool) -> dict:
    """Raise SystemExit if `output` is inside `root/results` and the tree is not clean.

    Output written anywhere else (a temporary folder, a scratch directory) is not
    filed as a result and is always allowed. Returns the git state.
    """
    state = git_state(root)
    # Case-folded, because macOS file systems treat RESULTS/ and results/ as one folder.
    target = str(Path(output).resolve()).casefold()
    results = str((root / "results").resolve()).casefold()
    inside_results = target == results or target.startswith(results + "/")
    clean = state["git_commit"] is not None and state["working_tree_dirty"] is False
    if inside_results and not clean and not allow_dirty:
        reason = ("not a git checkout" if state["git_commit"] is None
                  else "the working tree has uncommitted changes")
        raise SystemExit(f"refusing to write {output}: {reason}. Commit first, write "
                         "outside results/, or pass --allow-dirty for an exploratory run.")
    return state


CLAIM = ".claimed"


def claim_output_folder(folder: Path) -> None:
    """Create `folder` and claim it for one process, or raise SystemExit.

    The claim is a marker file created with O_EXCL, so of two processes racing
    for one folder exactly one succeeds (audit of 24 Nov, SF3). A folder that
    already holds anything, finished or crashed, is never reused.
    """
    import os
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(folder / CLAIM, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        raise SystemExit(f"{folder} is claimed by another run; choose another folder") from None
    if any(p.name != CLAIM for p in folder.iterdir()):
        raise SystemExit(f"{folder} is not empty; choose another folder")


def write_text_atomic(path: Path, text: str) -> None:
    """Write `text` so that `path` appears whole or not at all (audit of 24 Nov, MI4):
    summary.json marks a run as finished, so a half-written one must never exist."""
    import os
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text)
    os.replace(partial, path)
