#!/usr/bin/env python3
"""
Merge multiple .wlh hyphenation dictionaries into a single file.

A .wlh file holds one word per line; syllable breaks are marked with ASCII
hyphens (e.g. ``wel-kom``). Two entries refer to the "same word" iff they are
identical once all hyphens are removed. When two or more *different* input
files disagree on how such a word should be hyphenated, the word is excluded
from the merged output and every conflicting variant is written to a separate
collisions file (annotated with the source file it came from), so the
disagreement can be reviewed by hand.

Disagreements *within a single file* are ignored: a dictionary that lists the
same word twice with conflicting hyphenations is collapsed to its first
occurrence before being merged. Only cross-file conflicts are treated as
collisions.

Usage:
    python3 merge_wlh.py FILE [FILE ...] -o merged.wlh [-c collisions.wlh]

If ``-c/--collisions`` is omitted, it defaults to ``<output>.collisions``.
Words are emitted in the order they were first encountered across the inputs,
which keeps the merge stable and diff-friendly when re-run.
"""

import argparse
import sys
from pathlib import Path


def iter_entries(path: Path):
    """Yield (word_key, hyphenated_form) for each non-empty line in *path*.

    ``word_key`` is the word with all hyphens removed and is used as the
    identity for collision detection; ``hyphenated_form`` is the line as
    written in the source file (whitespace stripped).
    """
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            entry = raw.strip()
            if not entry:
                continue
            yield entry.replace("-", ""), entry


def load_file(path: Path) -> dict[str, str]:
    """Return ``key -> first hyphenated form`` for a single .wlh file.

    Within-file duplicates are resolved by keeping the first occurrence; any
    later line with the same key (whether identical or differently hyphenated)
    is silently dropped. This makes each file's contribution to the merge
    self-consistent before we start comparing across files.
    """
    entries: dict[str, str] = {}
    for key, form in iter_entries(path):
        entries.setdefault(key, form)
    return entries


def merge(inputs: list[Path]) -> tuple[dict[str, str], dict[str, dict[str, Path]]]:
    """Merge *inputs* and return (kept, collisions).

    * ``kept`` maps word_key -> agreed-upon hyphenated form, in insertion order.
    * ``collisions`` maps word_key -> {hyphenated_form: first_source_path}, also
      in insertion order, for every key where two or more inputs disagreed.

    Each file is first collapsed via :func:`load_file`, so only cross-file
    disagreements can produce a collision. A key that ends up in ``collisions``
    is removed from ``kept``; once a word is known to be ambiguous across
    files, no later file can rescue it.
    """
    kept: dict[str, str] = {}
    kept_source: dict[str, Path] = {}
    collisions: dict[str, dict[str, Path]] = {}

    for path in inputs:
        for key, form in load_file(path).items():
            if key in collisions:
                # Already disputed across files — record any new variant.
                collisions[key].setdefault(form, path)
                continue

            previous = kept.get(key)
            if previous is None:
                kept[key] = form
                kept_source[key] = path
            elif previous != form:
                # First cross-file disagreement: promote both variants into
                # the collisions bucket and drop the entry from the merged set.
                collisions[key] = {
                    previous: kept_source[key],
                    form: path,
                }
                del kept[key]
                del kept_source[key]
            # else: identical hyphenation in another file → silent dedupe.

    return kept, collisions


def write_merged(path: Path, kept: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for form in kept.values():
            fh.write(form)
            fh.write("\n")


def write_collisions(path: Path, collisions: dict[str, dict[str, Path]]) -> None:
    """Write one block per disputed word.

    Format::

        # <word_key>
        <variant1>\t<source1>
        <variant2>\t<source2>
        <blank line>

    The leading ``#`` comment makes the file easy to grep and the blank line
    separates entries so blocks can be visually scanned.
    """
    with path.open("w", encoding="utf-8") as fh:
        for key, variants in collisions.items():
            fh.write(f"# {key}\n")
            for form, source in variants.items():
                fh.write(f"{form}\t{source}\n")
            fh.write("\n")

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge any number of .wlh hyphenation dictionaries into one file. "
            "Words whose hyphenation disagrees across inputs are diverted to a "
            "separate collisions file and excluded from the merged output."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input .wlh files to merge (two or more is the useful case).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Path to write the merged .wlh file.",
    )
    parser.add_argument(
        "-c",
        "--collisions",
        type=Path,
        default=None,
        help="Path to write disputed words to (default: <output>.collisions).",
    )
    args = parser.parse_args()

    missing = [p for p in args.inputs if not p.is_file()]
    if missing:
        for p in missing:
            print(f"Error: input file not found: {p}", file=sys.stderr)
        return 1

    collisions_path = args.collisions or args.output.with_suffix(
        args.output.suffix + ".collisions"
    )

    kept, collisions = merge(args.inputs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    collisions_path.parent.mkdir(parents=True, exist_ok=True)

    write_merged(args.output, kept)
    write_collisions(collisions_path, collisions)

    print(
        f"Merged {len(args.inputs)} file(s): "
        f"{len(kept)} unique words → {args.output}, "
        f"{len(collisions)} collisions → {collisions_path}.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
