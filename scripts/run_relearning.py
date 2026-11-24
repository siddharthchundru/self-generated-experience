"""Run the re-learning control on one finished training run (spec section 7, H4).

Run: python scripts/run_relearning.py --run RUN_FOLDER --out OUT [--init fresh|same]

A new learner trains on RUN_FOLDER's final buffer with the run's gradient-step
count: warm-up on D0 for the run's recorded warm-up length, an Adam reset, then
N x ratio updates on the whole final buffer. No environment is used. See
`src/relearning.py` for what the control can and cannot show, and for the two
`--init` choices (default "fresh", decision 13 of the 4 Nov audit, open).

Writes into OUT: settings.json (the source run's, unchanged, so the evaluation
script reads the re-learner like any run), relearning.json (source folder,
init, source fingerprints), run_record.json, chunks.jsonl (one line per T x
ratio updates), pretrained.pt, final.pt, and summary.json last. Pilot and main
re-learning is refused unless the code (source hashes of src/ and the run
scripts) is the source run's. Writing into
results/ needs a clean, committed tree; --allow-dirty marks an exploratory run.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gcsl import pin_single_thread  # noqa: E402
from provenance import claim_output_folder, guard_results_write, run_record, write_text_atomic  # noqa: E402
from relearning import INITS, run_relearning  # noqa: E402
from runs import code_hashes, load_buffer, load_run  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="a finished training run folder")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--init", choices=INITS, default="fresh")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    import torch

    settings, summary, source_record = load_run(args.run)
    buffer = load_buffer(args.run)
    record = run_record(ROOT)
    guard_results_write(ROOT, args.out, args.allow_dirty)
    # Same code as the source run? A learner change between the original and its
    # re-learner would confound H4 (audit of 24 Nov, MI5).
    same_code = bool(code_hashes(source_record)) and code_hashes(source_record) == code_hashes(record)
    if not same_code and settings.purpose in ("pilot", "main"):
        raise SystemExit("the re-learner's code differs from the source run's (source hashes of src/ and the "
                         "run scripts); pilot and main re-learning must run the code that trained the source")
    if not same_code:
        print("warning: the re-learner's code differs from the source run's", file=sys.stderr)
    claim_output_folder(args.out)
    record["allow_dirty"] = args.allow_dirty
    record["torch_threads"] = {"intra_op": torch.get_num_threads(), "inter_op": torch.get_num_interop_threads()}
    (args.out / "run_record.json").write_text(json.dumps(record, indent=2))
    (args.out / "settings.json").write_text(json.dumps(asdict(settings), indent=2))
    (args.out / "relearning.json").write_text(json.dumps({
        "source_run": str(args.run.resolve()), "init": args.init,
        "source_commit": source_record.get("git_commit"),
        "source_pretrained_digest": summary["pretrained_digest"],
        "source_final_digest": summary["final_digest"],
        "source_final_buffer_digest": summary["final_buffer_digest"],
        "same_code_as_source": same_code,
    }, indent=2))

    with open(args.out / "chunks.jsonl", "w") as chunks:
        def write(row):
            chunks.write(json.dumps(row) + "\n")
            chunks.flush()
        result = run_relearning(settings, summary, buffer, init=args.init, on_chunk=write)
    torch.save(result["pretrained_state"], args.out / "pretrained.pt")
    torch.save(result["final_state"], args.out / "final.pt")
    result["summary"]["source"]["folder"] = str(args.run.resolve())
    result["summary"]["source"]["same_code"] = same_code
    write_text_atomic(args.out / "summary.json", json.dumps(result["summary"], indent=2))  # written last
    budget = result["summary"]["budget"]
    print(f"re-learning ({args.init}) of {settings.condition} seed {settings.seed}: "
          f"{budget['warmup_gradient_steps']} + {budget['practice_gradient_steps']} gradient steps, "
          f"0 environment steps; budget audit passed")


if __name__ == "__main__":
    pin_single_thread()
    main()
