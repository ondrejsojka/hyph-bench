#!/usr/bin/env python3
"""Regenerate one hand-tuned PATGEN profile on a recorded run's training split.

Trains every level of a `profiles/*.in` file on `<run-dir>/<dataset>/splits/data.train.wlh`
with the recorded PATGEN binary, then scores the resulting pattern set on the recorded
held-out test split under the current scoring rule (`Hyphenator.score`) and writes a JSON
evidence record. Used for profiles that are not part of the reported comparison, such as
the untruncated eight-level `wortliste8`, whose numbers the manuscript quotes as historical
context.

Usage:
    uv run python -m scripts.regenerate_hand_profile \
        --dataset de/wortliste --profile profiles/wortliste8.in \
        --patgen /home/dev/patgen-10x \
        --output results/gpopt260828_analysis/wortliste8_regeneration.json
"""

import argparse
import json
import uuid
from pathlib import Path

from .analyze_gpopt260828 import train_hand_profile
from .analyze_heldout_results import (
    aggregate,
    f17,
    parse_profile,
    per_line_counts,
    precision,
    recall,
)
from .dataset_utls import find_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, help="dataset id, e.g. de/wortliste")
    parser.add_argument("--profile", required=True, help="path to a profiles/*.in file")
    parser.add_argument("--run-dir", default="results/gpopt260828")
    parser.add_argument("--patgen", default="patgen")
    parser.add_argument("--output", required=True, help="JSON evidence record to write")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) / args.dataset
    train_path = run_dir / "splits/data.train.wlh"
    test_path = run_dir / "splits/data.test.wlh"
    for path in (train_path, test_path):
        if not path.exists():
            raise SystemExit(
                f"missing {path}; regenerate the recorded split with "
                f"`python -m scripts.analyze_gpopt260828 --write-splits`"
            )

    profile_path = Path(args.profile)
    _, translate_path = find_dataset(args.dataset)

    pattern_path, stats, scorer = train_hand_profile(
        args.patgen,
        profile_path,
        str(train_path),
        translate_path,
        f"_regen_{profile_path.stem}_{uuid.uuid4().hex[:8]}",
    )
    try:
        counts = aggregate(per_line_counts(str(test_path), pattern_path, translate_path))
    finally:
        scorer.clean()

    record = {
        "profile": profile_path.as_posix(),
        "dataset": args.dataset,
        "levels": len(parse_profile(profile_path)),
        "train_split": train_path.as_posix(),
        "test_split": test_path.as_posix(),
        "patgen": args.patgen,
        "f17": f17(counts["good"], counts["bad"], counts["missed"]),
        "precision": precision(counts["good"], counts["bad"]),
        "recall": recall(counts["good"], counts["missed"]),
        **counts,
        **stats,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    print(f"record -> {output}")


if __name__ == "__main__":
    main()
