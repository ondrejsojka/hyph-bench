#!/usr/bin/env python3
"""Build a union annotation file across all annotators.

For each line, output the canonical word with a hyphen at every position where
at least one annotator placed one. Each hyphen is preceded by a Unicode
superscript number indicating how many annotators placed a hyphen at that
position.
"""

import os
import sys

ANNOTATORS = [
    "claude",
    "claude_hyphenation_stressed",
    "claude_hyphenation_stressed_repeat",
    "gemini_first",
    "gemini_hyphenation_stressed",
    "gemini_second",
    "human_1",
    "human_2",
]

CANONICAL_PRIORITY = ["human_1", "human_2", "gemini_first", "gemini_second"]

SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def hyphen_positions(rhs: str) -> set[int]:
    """Return the set of character indices (in the unhyphenated word) before
    which a hyphen appears in `rhs`."""
    positions: set[int] = set()
    idx = 0
    for ch in rhs:
        if ch == "-":
            positions.add(idx)
        else:
            idx += 1
    return positions


def main() -> None:
    base = os.path.join(os.path.dirname(__file__), "..", "annotation_results")
    base = os.path.abspath(base)

    files: dict[str, list[str]] = {}
    for name in ANNOTATORS:
        with open(os.path.join(base, name), encoding="utf-8") as f:
            files[name] = [ln.rstrip("\n") for ln in f]

    n_lines = max(len(v) for v in files.values())
    out_lines: list[str] = []

    for i in range(n_lines):
        # Pick canonical word for this line: prefer human, fall back to others
        # whose LHS == RHS-without-hyphens (clean entries).
        canonical: str | None = None
        for name in CANONICAL_PRIORITY + ANNOTATORS:
            if i >= len(files[name]):
                continue
            line = files[name][i]
            if "=" not in line:
                continue
            lhs, rhs = line.split("=", 1)
            if lhs == rhs.replace("-", ""):
                canonical = lhs
                break
        if canonical is None:
            # Last resort: take LHS of first file
            line = files[ANNOTATORS[0]][i]
            canonical = line.split("=", 1)[0] if "=" in line else line

        # Count hyphen votes per position (1..len(canonical)-1)
        counts: dict[int, int] = {}
        for name in ANNOTATORS:
            if i >= len(files[name]):
                continue
            line = files[name][i]
            if "=" not in line:
                continue
            _, rhs = line.split("=", 1)
            unhyphenated = rhs.replace("-", "")
            if unhyphenated != canonical:
                # Skip annotators whose word doesn't match canonical: their
                # positions can't be aligned reliably.
                continue
            for pos in hyphen_positions(rhs):
                if 0 < pos < len(canonical):
                    counts[pos] = counts.get(pos, 0) + 1

        # Build output: insert "<sup>-" before each marked position
        out_chars: list[str] = []
        for j, ch in enumerate(canonical):
            if j in counts:
                out_chars.append(str(counts[j]).translate(SUPERSCRIPT))
                out_chars.append("-")
            out_chars.append(ch)
        out_lines.append("".join(out_chars))

    out_path = os.path.join(base, "union")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"Wrote {len(out_lines)} lines to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
