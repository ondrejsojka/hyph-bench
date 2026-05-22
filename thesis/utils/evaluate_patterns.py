#!/usr/bin/env python3
"""Evaluate hyphenation pattern files against human-annotated truth files.

For each pattern file, hyphenates a plain word list and computes Cohen's kappa
against each truth file. Outputs a LaTeX tabularx table.

Usage:
    python evaluate_patterns.py words.wl \\
        --truth human1.wl human2.wl \\
        --patterns uk.pat uk2.pat \\
        [--output table.tex]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from hyphenate import Hyphenator


def load_words(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def hyphen_positions(word: str) -> set[int]:
    positions: set[int] = set()
    idx = 0
    for ch in word:
        if ch == "-":
            positions.add(idx)
        else:
            idx += 1
    return positions


def load_truth(path):
    """Returns list of (bare_word, hyphen_position_set)."""
    result = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            bare = word.replace("-", "")
            result.append((bare, hyphen_positions(word)))
    return result


def load_pattern_tokens(path):
    """Extract raw pattern tokens from a .tex, .dic, or plain pattern file."""
    ext = os.path.splitext(path)[1].lower()
    tokens = []

    if ext == ".tex":
        in_block = False
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not in_block:
                    if "\\patterns{" in line:
                        in_block = True
                        after = line[line.index("\\patterns{") + len("\\patterns{"):]
                        if "}" in after:
                            tokens.extend(after[:after.index("}")].split())
                            in_block = False
                        else:
                            tokens.extend(after.split())
                else:
                    if "}" in line:
                        tokens.extend(line[:line.index("}")].split())
                        break
                    tokens.extend(line.split("%")[0].split())

    elif ext == ".dic":
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[1:]:  # first line is encoding declaration
            tokens.extend(line.strip().split())

    else:
        with open(path, encoding="utf-8") as f:
            for line in f:
                tokens.extend(line.strip().split())

    return tokens


def make_hyphenator(path):
    """Build a Hyphenator from a .tex, .dic, or plain pattern file.

    Returns (hyphenator, pattern_count).
    """
    tokens = load_pattern_tokens(path)
    h = Hyphenator.__new__(Hyphenator)
    h.tree = {}
    for token in tokens:
        h._insert_pattern(token)
    return h, len(tokens)


def hyphenate_all(words, pattern_file):
    """Returns (list of (bare_word, hyphen_position_set), pattern_count)."""
    h, n_patterns = make_hyphenator(pattern_file)
    result = []
    for word in words:
        pieces = h.hyphenate_word(word)
        result.append((word, hyphen_positions("-".join(pieces))))
    return result, n_patterns


def cohen_kappa(a, b) -> float:
    tt = ff = tf = ft = 0
    for (wa, pa), (wb, pb) in zip(a, b):
        if wa != wb or not wa:
            continue
        for pos in range(1, len(wa)):
            in_a = pos in pa
            in_b = pos in pb
            if in_a and in_b:
                tt += 1
            elif in_a and not in_b:
                ft += 1
            elif not in_a and in_b:
                tf += 1
            else:
                ff += 1
    total = tt + ff + tf + ft
    if total == 0:
        return float("nan")
    p_h = ((tt + tf) / total) * ((tt + ft) / total)
    p_n = ((ff + tf) / total) * ((ff + ft) / total)
    p_e = p_h + p_n
    if 1 - p_e == 0:
        return float("nan")
    return ((tt + ff) / total - p_e) / (1 - p_e)


def print_table(pattern_files, truth_files, kappas, pattern_counts, file=None):
    import math
    p = lambda *a: print(*a, file=file)
    truth_names = [os.path.basename(t).replace("_", "\\_") for t in truth_files]
    # columns: name | n_patterns | kappa_1 | eff_1 | kappa_2 | eff_2 | ...
    n_truth = len(truth_files)
    col_spec = "l|r|" + "|".join("XX" for _ in truth_files)
    kappa_headers = " & ".join(
        f"$\\kappa$ ({name}) & $\\kappa / \\log_{{10}} n$" for name in truth_names
    )
    p("\\begin{table}[h]")
    p("  \\centering")
    p(f"  \\begin{{tabularx}}{{\\textwidth}}{{{col_spec}}}")
    p(f"    Patterns & $n$ & {kappa_headers} \\\\")
    p("    \\hline")
    for pat_file, row, n in zip(pattern_files, kappas, pattern_counts):
        pat_name = os.path.basename(pat_file).replace("_", "\\_")
        log_n = math.log10(n) if n > 0 else float("nan")
        cells = []
        for k in row:
            eff = k / log_n if log_n and not math.isnan(k) else float("nan")
            cells.append(f"{k:.4f} & {eff:.4f}")
        p(f"    {pat_name} & {n} & {' & '.join(cells)} \\\\")
    p("  \\end{tabularx}")
    p("  \\caption{}")
    p("  \\label{tab:pattern-evaluation}")
    p("\\end{table}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pattern files against truth hyphenations using Cohen's kappa."
    )
    parser.add_argument("words", help="plain word list to hyphenate (one word per line)")
    parser.add_argument("--truth", nargs="+", required=True, metavar="FILE",
                        help="hyphenated truth files (one hyphenated word per line)")
    parser.add_argument("--patterns", nargs="+", required=True, metavar="FILE",
                        help="patgen pattern files to evaluate")
    parser.add_argument("--output", metavar="FILE",
                        help="write LaTeX table to FILE instead of stdout")
    args = parser.parse_args()

    words = load_words(args.words)
    truths = [load_truth(t) for t in args.truth]

    kappas = []
    pattern_counts = []
    for pat_file in args.patterns:
        hyphenated, n_patterns = hyphenate_all(words, pat_file)
        kappas.append([cohen_kappa(hyphenated, truth) for truth in truths])
        pattern_counts.append(n_patterns)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            print_table(args.patterns, args.truth, kappas, pattern_counts, file=f)
        print(f"Table written to {args.output}")
    else:
        print_table(args.patterns, args.truth, kappas, pattern_counts)


if __name__ == "__main__":
    main()