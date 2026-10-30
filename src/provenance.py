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
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True,
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
    inside_results = Path(output).resolve().is_relative_to((root / "results").resolve())
    clean = state["git_commit"] is not None and state["working_tree_dirty"] is False
    if inside_results and not clean and not allow_dirty:
        reason = ("not a git checkout" if state["git_commit"] is None
                  else "the working tree has uncommitted changes")
        raise SystemExit(f"refusing to write {output}: {reason}. Commit first, write "
                         "outside results/, or pass --allow-dirty for an exploratory run.")
    return state
