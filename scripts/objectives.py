"""
Pluggable objective functions for patgen GP optimization.

Each objective implements a score() method that takes patgen results
and returns a score where higher is better.
"""

from abc import ABC, abstractmethod


class ObjectiveFunction(ABC):
    """Base class for optimization objectives."""

    @abstractmethod
    def score(self, good: int, bad: int, missed: int,
              trie_nodes: int = 0, n_patterns: int = 0) -> float:
        """
        Compute score from patgen results. Higher is better.

        Args:
            good: True positives (correct hyphenation points)
            bad: False positives (wrong hyphenation points)
            missed: False negatives (missed hyphenation points)
            trie_nodes: Pattern trie size (for size optimization)
            n_patterns: Number of patterns generated
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return human-readable name for this objective."""
        pass


class F17Score(ObjectiveFunction):
    """
    F_1/7 score - weights precision 7x more than recall.

    This is useful when false positives (bad hyphenations) are much
    worse than false negatives (missed hyphenations), which is typically
    the case for typographic hyphenation.

    F_beta = (1 + beta^2) * precision * recall / (beta^2 * precision + recall)
    With beta = 1/7, precision is weighted 7x more than recall.
    """

    def __init__(self, beta: float = 1/7):
        self.beta = beta

    def score(self, good: int, bad: int, missed: int, **kwargs) -> float:
        if good == 0:
            return 0.0

        precision = good / (good + bad) if (good + bad) > 0 else 0.0
        recall = good / (good + missed) if (good + missed) > 0 else 0.0

        if precision == 0 or recall == 0:
            return 0.0

        beta_sq = self.beta ** 2
        return (1 + beta_sq) * precision * recall / ((beta_sq * precision) + recall)

    @property
    def name(self) -> str:
        return f"F_1/{int(1/self.beta)}"


class BoundedBadMinSize(ObjectiveFunction):
    """
    Hard constraint on bad hyphenations, then minimize pattern size.

    This objective is useful for identifying outliers in the dataset:
    - If bad > threshold: return 0 (infeasible configuration)
    - Else: return score inversely proportional to pattern count

    This can help find minimal pattern sets that still achieve
    acceptable error rates.
    """

    def __init__(self, bad_threshold: int = 500):
        self.bad_threshold = bad_threshold

    def score(self, good: int, bad: int, missed: int,
              n_patterns: int = 0, **kwargs) -> float:
        if bad > self.bad_threshold:
            return 0.0  # Infeasible

        # Normalize: fewer patterns = higher score
        # Using 1/(1+n/1000) to map to (0,1] range
        # With 1000 patterns -> 0.5, 10000 patterns -> ~0.09
        return 1.0 / (1 + n_patterns / 1000)

    @property
    def name(self) -> str:
        return f"BoundedBad({self.bad_threshold})"


class WeightedScore(ObjectiveFunction):
    """
    Weighted combination of good/bad/missed with customizable weights.

    Score = good_weight * good - bad_weight * bad - missed_weight * missed
    Normalized to [0, 1] range.
    """

    def __init__(self, good_weight: float = 1.0, bad_weight: float = 7.0,
                 missed_weight: float = 1.0):
        self.good_weight = good_weight
        self.bad_weight = bad_weight
        self.missed_weight = missed_weight

    def score(self, good: int, bad: int, missed: int, **kwargs) -> float:
        total = good + bad + missed
        if total == 0:
            return 0.0

        # Raw weighted score
        raw = (self.good_weight * good -
               self.bad_weight * bad -
               self.missed_weight * missed)

        # Normalize to approximately [0, 1]
        # Max possible is good_weight * total, min is -bad_weight * total
        max_score = self.good_weight * total
        min_score = -self.bad_weight * total

        return (raw - min_score) / (max_score - min_score)

    @property
    def name(self) -> str:
        return f"Weighted(g={self.good_weight},b={self.bad_weight},m={self.missed_weight})"


class PrecisionRecallCurve(ObjectiveFunction):
    """
    Score based on distance to ideal point (precision=1, recall=1).

    This gives a balanced view of both metrics without strong weighting.
    """

    def score(self, good: int, bad: int, missed: int, **kwargs) -> float:
        if good == 0:
            return 0.0

        precision = good / (good + bad) if (good + bad) > 0 else 0.0
        recall = good / (good + missed) if (good + missed) > 0 else 0.0

        # Distance to ideal point (1, 1) - closer is better
        distance = ((1 - precision) ** 2 + (1 - recall) ** 2) ** 0.5

        # Convert to score where higher is better (max distance is sqrt(2))
        return 1 - distance / (2 ** 0.5)

    @property
    def name(self) -> str:
        return "PRCurve"


def get_objective(name: str, **kwargs) -> ObjectiveFunction:
    """
    Factory function to get objective by name.

    Args:
        name: One of 'f17', 'bounded_bad', 'weighted', 'pr_curve'
        **kwargs: Additional arguments for the objective

    Returns:
        ObjectiveFunction instance
    """
    objectives = {
        'f17': F17Score,
        'bounded_bad': BoundedBadMinSize,
        'weighted': WeightedScore,
        'pr_curve': PrecisionRecallCurve,
    }

    if name not in objectives:
        raise ValueError(f"Unknown objective: {name}. Available: {list(objectives.keys())}")

    return objectives[name](**kwargs)
