#!/usr/bin/env python3
"""Stricter test of whether claude_new_prompt is indistinguishable from humans.

Bootstraps over words to compute a confidence interval on the gap between
human-human and model-human Cohen's kappa. Also reports per-word divergence
to find where claude_new_prompt diverges from BOTH human annotators.
"""

import os
import sys
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "..", "annotation_results")
BASE = os.path.abspath(BASE)

ANNOTATORS = [
    "claude",
    "claude_hyphenation_stressed",
    "claude_hyphenation_stressed_repeat",
    "claude_new_prompt",
    "gemini_first",
    "gemini_second",
    "gemini_hyphenation_stressed",
    "human_1",
    "human_2",
]


def hyphen_positions(rhs: str) -> set[int]:
    positions: set[int] = set()
    idx = 0
    for ch in rhs:
        if ch == "-":
            positions.add(idx)
        else:
            idx += 1
    return positions


def load(name: str) -> list[tuple[str, set[int], int]]:
    """Returns list of (canonical_word, hyphen_position_set, n_internal_positions)."""
    out = []
    with open(os.path.join(BASE, name), encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "=" not in line:
                out.append(("", set(), 0))
                continue
            lhs, rhs = line.split("=", 1)
            unhyph = rhs.replace("-", "")
            # length-1 = number of internal "between-character" positions
            out.append((unhyph, hyphen_positions(rhs), max(0, len(unhyph) - 1)))
    return out


def kappa_from_counts(tt: int, ff: int, tf: int, ft: int) -> float:
    total = tt + ff + tf + ft
    if total == 0:
        return float("nan")
    p_h = ((tt + tf) / total) * ((tt + ft) / total)
    p_n = ((ff + tf) / total) * ((ff + ft) / total)
    p_o = (tt + ff) / total
    p_e = p_h + p_n
    if 1 - p_e == 0:
        return float("nan")
    return (p_o - p_e) / (1 - p_e)


def per_word_counts(a: list[tuple[str, set[int], int]],
                    b: list[tuple[str, set[int], int]]):
    """For each line where canonical words match, return (tt, ff, tf, ft, n_disagree)."""
    n = min(len(a), len(b))
    out = []
    for i in range(n):
        wa, pa, lena = a[i]
        wb, pb, lenb = b[i]
        if wa != wb or wa == "" or lena == 0:
            out.append(None)
            continue
        # Iterate over internal positions 1..len(wa)-1
        tt = ff = tf = ft = 0
        for pos in range(1, lena + 1):
            in_a = pos in pa
            in_b = pos in pb
            if in_a and in_b:
                tt += 1
            elif in_a and not in_b:
                ft += 1  # a hyphenates, b doesn't (matching original convention)
            elif (not in_a) and in_b:
                tf += 1
            else:
                ff += 1
        n_disagree = tf + ft
        out.append((tt, ff, tf, ft, n_disagree))
    return out


def kappa_pair(a, b) -> float:
    counts = per_word_counts(a, b)
    tt = sum(c[0] for c in counts if c is not None)
    ff = sum(c[1] for c in counts if c is not None)
    tf = sum(c[2] for c in counts if c is not None)
    ft = sum(c[3] for c in counts if c is not None)
    return kappa_from_counts(tt, ff, tf, ft)


def kappa_pair_from_perword(counts) -> float:
    tt = sum(c[0] for c in counts if c is not None)
    ff = sum(c[1] for c in counts if c is not None)
    tf = sum(c[2] for c in counts if c is not None)
    ft = sum(c[3] for c in counts if c is not None)
    return kappa_from_counts(tt, ff, tf, ft)


def bootstrap_gap(counts_target, counts_baseline, n_boot: int = 10000, seed: int = 42):
    """Bootstrap word indices, compute (baseline_kappa - target_kappa) per resample."""
    rng = np.random.default_rng(seed)
    valid_idx = [i for i, c in enumerate(counts_target)
                 if c is not None and counts_baseline[i] is not None]
    n = len(valid_idx)
    target_arr = np.array([counts_target[i] for i in valid_idx], dtype=np.int64)
    base_arr = np.array([counts_baseline[i] for i in valid_idx], dtype=np.int64)

    gaps = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        t = target_arr[idx].sum(axis=0)
        bs = base_arr[idx].sum(axis=0)
        kt = kappa_from_counts(int(t[0]), int(t[1]), int(t[2]), int(t[3]))
        kb = kappa_from_counts(int(bs[0]), int(bs[1]), int(bs[2]), int(bs[3]))
        gaps[b] = kb - kt
    return gaps


def main():
    data = {name: load(name) for name in ANNOTATORS}

    h1 = data["human_1"]
    h2 = data["human_2"]
    cnp = data["claude_new_prompt"]

    counts_hh = per_word_counts(h1, h2)
    counts_ch1 = per_word_counts(cnp, h1)
    counts_ch2 = per_word_counts(cnp, h2)

    k_hh = kappa_pair_from_perword(counts_hh)
    k_ch1 = kappa_pair_from_perword(counts_ch1)
    k_ch2 = kappa_pair_from_perword(counts_ch2)

    print("=" * 70)
    print("POINT ESTIMATES (Cohen's kappa, pooled over all positions)")
    print("=" * 70)
    print(f"  human_1 vs human_2          : {k_hh:.4f}")
    print(f"  claude_new_prompt vs human_1: {k_ch1:.4f}")
    print(f"  claude_new_prompt vs human_2: {k_ch2:.4f}")
    print()

    print("=" * 70)
    print("BOOTSTRAP CI ON GAP = (human-human kappa) - (model-human kappa)")
    print("Resampling over words, n_boot = 10000")
    print("=" * 70)
    for label, ctgt in [("claude_new_prompt vs h1", counts_ch1),
                        ("claude_new_prompt vs h2", counts_ch2)]:
        gaps = bootstrap_gap(ctgt, counts_hh)
        mean = gaps.mean()
        lo, hi = np.percentile(gaps, [2.5, 97.5])
        p_below_zero = (gaps <= 0).mean()
        print(f"  {label}:")
        print(f"    gap mean = {mean:+.4f}, 95% CI = [{lo:+.4f}, {hi:+.4f}]")
        print(f"    P(gap <= 0) = {p_below_zero:.4f}  "
              f"(model >= humans on this resample)")
    print()

    # All other models for context
    print("=" * 70)
    print("SAME GAP TEST FOR ALL OTHER MODELS (avg over h1, h2)")
    print("=" * 70)
    for name in ANNOTATORS:
        if name in ("human_1", "human_2"):
            continue
        m = data[name]
        c1 = per_word_counts(m, h1)
        c2 = per_word_counts(m, h2)
        k1 = kappa_pair_from_perword(c1)
        k2 = kappa_pair_from_perword(c2)
        avg = (k1 + k2) / 2
        gap = k_hh - avg
        # combined bootstrap
        rng = np.random.default_rng(42)
        valid = [i for i, c in enumerate(c1)
                 if c is not None and c2[i] is not None and counts_hh[i] is not None]
        a1 = np.array([c1[i] for i in valid], dtype=np.int64)
        a2 = np.array([c2[i] for i in valid], dtype=np.int64)
        ah = np.array([counts_hh[i] for i in valid], dtype=np.int64)
        n = len(valid)
        gaps = np.empty(2000)
        for b in range(2000):
            idx = rng.integers(0, n, size=n)
            s1 = a1[idx].sum(axis=0); s2 = a2[idx].sum(axis=0); sh = ah[idx].sum(axis=0)
            k_m = (kappa_from_counts(int(s1[0]), int(s1[1]), int(s1[2]), int(s1[3])) +
                   kappa_from_counts(int(s2[0]), int(s2[1]), int(s2[2]), int(s2[3]))) / 2
            k_h = kappa_from_counts(int(sh[0]), int(sh[1]), int(sh[2]), int(sh[3]))
            gaps[b] = k_h - k_m
        lo, hi = np.percentile(gaps, [2.5, 97.5])
        print(f"  {name:40s} avg_kappa={avg:.4f}  gap={gap:+.4f}  "
              f"95% CI=[{lo:+.4f}, {hi:+.4f}]")
    print()

    # Per-word divergence: words where claude_new_prompt disagrees with BOTH humans
    print("=" * 70)
    print("TOP 20 WORDS WHERE claude_new_prompt DIVERGES FROM BOTH HUMANS")
    print("(metric: |cnp Δ h1| + |cnp Δ h2|; tiebreak by |h1 Δ h2|)")
    print("=" * 70)
    rows = []
    for i in range(min(len(h1), len(h2), len(cnp))):
        if counts_ch1[i] is None or counts_ch2[i] is None or counts_hh[i] is None:
            continue
        word = h1[i][0]
        d_ch1 = counts_ch1[i][4]
        d_ch2 = counts_ch2[i][4]
        d_hh = counts_hh[i][4]
        rows.append((d_ch1 + d_ch2, d_hh, word, i, d_ch1, d_ch2))
    rows.sort(key=lambda r: (-r[0], r[1]))
    print(f"  {'cnp~h':>5} {'h~h':>4}  word")
    for total, d_hh, word, _, d1, d2 in rows[:20]:
        print(f"  {total:>5} {d_hh:>4}  {word}  (cnp~h1={d1}, cnp~h2={d2})")
    print()

    # How often does claude_new_prompt agree with at least one human entirely?
    n_valid = sum(1 for i in range(len(counts_hh))
                  if counts_hh[i] is not None and counts_ch1[i] is not None
                  and counts_ch2[i] is not None)
    n_perfect_h1 = sum(1 for i in range(len(counts_ch1))
                       if counts_ch1[i] is not None and counts_ch1[i][4] == 0)
    n_perfect_h2 = sum(1 for i in range(len(counts_ch2))
                       if counts_ch2[i] is not None and counts_ch2[i][4] == 0)
    n_perfect_hh = sum(1 for i in range(len(counts_hh))
                       if counts_hh[i] is not None and counts_hh[i][4] == 0)
    n_perfect_either = sum(1 for i in range(len(counts_ch1))
                           if counts_ch1[i] is not None and counts_ch2[i] is not None
                           and (counts_ch1[i][4] == 0 or counts_ch2[i][4] == 0))
    print("=" * 70)
    print("PER-WORD PERFECT AGREEMENT (no position disagreements on word)")
    print("=" * 70)
    print(f"  comparable words: {n_valid}")
    print(f"  human_1 == human_2          : {n_perfect_hh:>4}  ({n_perfect_hh/n_valid:.1%})")
    print(f"  cnp == human_1              : {n_perfect_h1:>4}  ({n_perfect_h1/n_valid:.1%})")
    print(f"  cnp == human_2              : {n_perfect_h2:>4}  ({n_perfect_h2/n_valid:.1%})")
    print(f"  cnp == h1 OR cnp == h2      : {n_perfect_either:>4}  ({n_perfect_either/n_valid:.1%})")


if __name__ == "__main__":
    main()
