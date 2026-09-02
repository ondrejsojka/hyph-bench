#!/usr/bin/env python3
"""Held-out scoring helpers shared by the reported-result analyses.

Per-line Good/Bad/Missed counting, their aggregation, and the F_{1/7} score used
by `scripts.analyze_gpopt260828` and `scripts.analyze_gpopt260828_selectors`.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .hyphenator.hyphenator import Hyphenator


def f17(good: int, bad: int, missed: int, beta: float = 1 / 7) -> float:
    if good == 0:
        return 0.0
    precision = good / (good + bad) if good + bad else 0.0
    recall = good / (good + missed) if good + missed else 0.0
    if precision == 0 or recall == 0:
        return 0.0
    beta_sq = beta * beta
    return (1 + beta_sq) * precision * recall / ((beta_sq * precision) + recall)


def precision(good: int, bad: int) -> float:
    return good / (good + bad) if good + bad else 0.0


def recall(good: int, missed: int) -> float:
    return good / (good + missed) if good + missed else 0.0


def parse_profile(path: Path) -> List[Tuple[int, int, int, int, int]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [int(x) for x in line.split()]
            if len(parts) != 5:
                raise ValueError(f"Expected 5 values in {path}: {line}")
            rows.append(tuple(parts))
    return rows


def per_line_counts(test_path: str, pattern_path: str, translate_path: str) -> np.ndarray:
    hyphenator = Hyphenator(pattern_path, hyphenation_mark="-", translate_file=translate_path)
    rows = []
    with open(test_path, encoding="utf-8") as handle:
        for line in handle:
            rows.append(hyphenator.score(line.strip()))
    return np.asarray(rows, dtype=np.int64)


def aggregate(arr: np.ndarray) -> Dict[str, int]:
    good, bad, missed = arr.sum(axis=0)
    return {"good": int(good), "bad": int(bad), "missed": int(missed)}
