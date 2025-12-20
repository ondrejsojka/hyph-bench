"""
Gaussian Process optimizer for patgen parameters.

Optimizes a 4-dimensional parameter space: bad_weight for each of 4 levels.
Uses Upper Confidence Bound (UCB) acquisition for exploration-exploitation balance.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from typing import List, Tuple, Optional, Dict, Any
import pickle
import os

from .objectives import ObjectiveFunction


class GPOptimizer:
    """
    Gaussian Process optimizer for patgen parameters.

    Optimizes 4 parameters: bad_weight for each of 4 levels.
    Uses exploration (UCB) and exploitation strategy.

    Parameter space:
        bad_1, bad_2, bad_3, bad_4 in [1, 9] (integers)

    Fixed values:
        good_weight = 1 (all layers)
        threshold = configurable (default 5)
        pat_start, pat_finish from profile
    """

    def __init__(self, objective: ObjectiveFunction, seed: int = 42,
                 param_range: Tuple[int, int] = (1, 9),
                 min_samples_for_gp: int = 5):
        self.objective = objective
        self.rng = np.random.RandomState(seed)
        self.param_range = param_range
        self.min_samples_for_gp = min_samples_for_gp

        # 4-dimensional Matern kernel (good for integer parameters)
        kernel = Matern(
            nu=2.5,
            length_scale=[1.0] * 4,
            length_scale_bounds=(1e-3, 1e3)
        ) + WhiteKernel(
            noise_level=0.1,
            noise_level_bounds=(1e-4, 1)
        )

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=seed,
            n_restarts_optimizer=2
        )

        self.X: List[List[int]] = []      # Parameter history
        self.y: List[float] = []          # Score history
        self.results: List[Dict] = []     # Full results history

    def _random_params(self) -> Tuple[int, int, int, int]:
        """Generate random parameter set within range."""
        low, high = self.param_range
        return tuple(self.rng.randint(low, high + 1, size=4))

    def generate_coarse_grid(self, step: int = 4) -> List[Tuple[int, ...]]:
        """
        Generate coarse grid of parameters for warmup phase.

        Args:
            step: Step size for grid (default 4 gives [1,5,9] for range [1,9])

        Returns:
            List of parameter tuples covering the grid
        """
        from itertools import product
        low, high = self.param_range
        values = list(range(low, high + 1, step))
        if high not in values:
            values.append(high)
        grid = list(product(values, repeat=4))
        self.rng.shuffle(grid)  # Randomize order
        return [tuple(g) for g in grid]

    def _predict(self, params: Tuple[int, ...]) -> Tuple[float, float]:
        """
        Get GP prediction and uncertainty for a parameter set.

        Args:
            params: (bad_1, bad_2, bad_3, bad_4) tuple

        Returns:
            (mean, std) prediction
        """
        if len(self.X) < self.min_samples_for_gp:
            return 0.5, 1.0  # Prior before enough data

        x = np.array([list(params)])
        mean, std = self.gp.predict(x, return_std=True)
        return float(mean[0]), float(std[0])

    def update(self, params: Tuple[int, ...], good: int, bad: int,
               missed: int, n_patterns: int = 0, trie_nodes: int = 0) -> float:
        """
        Update GP with new observation.

        Args:
            params: (bad_1, bad_2, bad_3, bad_4) tuple
            good: True positives
            bad: False positives
            missed: False negatives
            n_patterns: Total pattern count
            trie_nodes: Trie node count

        Returns:
            The computed score
        """
        score = self.objective.score(
            good, bad, missed,
            n_patterns=n_patterns,
            trie_nodes=trie_nodes
        )

        self.X.append(list(params))
        self.y.append(score)
        self.results.append({
            'params': params,
            'good': good,
            'bad': bad,
            'missed': missed,
            'n_patterns': n_patterns,
            'trie_nodes': trie_nodes,
            'score': score
        })

        # Refit GP after enough observations
        if len(self.X) >= self.min_samples_for_gp:
            self.gp.fit(np.array(self.X), np.array(self.y))

        return score

    def suggest_batch(self, n: int = 5, n_candidates: int = 5000,
                      ucb_kappa: float = 2.0) -> List[Tuple[int, ...]]:
        """
        Suggest diverse batch of promising parameters.

        Uses Upper Confidence Bound (UCB) acquisition function:
            UCB = mean + kappa * std

        Args:
            n: Number of suggestions to return
            n_candidates: Number of random candidates to evaluate
            ucb_kappa: Exploration weight (higher = more exploration)

        Returns:
            List of parameter tuples
        """
        if len(self.X) < self.min_samples_for_gp:
            # Random sampling before GP is fitted
            return [self._random_params() for _ in range(n)]

        # Generate candidates
        candidates = [self._random_params() for _ in range(n_candidates)]
        X_cand = np.array([list(c) for c in candidates])

        # UCB acquisition: mean + kappa * std
        mean, std = self.gp.predict(X_cand, return_std=True)
        ucb = mean + ucb_kappa * std

        # Select diverse set from top quartile
        top_indices = np.where(ucb >= np.percentile(ucb, 75))[0]

        selected = []
        selected_set = set()

        # Always include best UCB
        best_idx = np.argmax(ucb)
        selected.append(candidates[best_idx])
        selected_set.add(candidates[best_idx])

        # Add diverse samples from top quartile using max-min distance
        while len(selected) < n and len(top_indices) > 0:
            max_min_dist = -1
            best_idx = -1

            for idx in top_indices:
                if candidates[idx] in selected_set:
                    continue

                # L1 distance to all selected
                dists = [np.sum(np.abs(np.array(candidates[idx]) - np.array(s)))
                         for s in selected]
                min_dist = min(dists)

                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = idx

            if best_idx != -1:
                selected.append(candidates[best_idx])
                selected_set.add(candidates[best_idx])
                top_indices = top_indices[top_indices != best_idx]
            else:
                break

        # Fill remaining with random if needed
        while len(selected) < n:
            new_params = self._random_params()
            if new_params not in selected_set:
                selected.append(new_params)
                selected_set.add(new_params)

        return selected

    def exploit_best(self, n: int = 5, n_candidates: int = 10000) -> List[Tuple[int, ...]]:
        """
        Return parameters with highest predicted scores (pure exploitation).

        Args:
            n: Number of suggestions
            n_candidates: Candidates to evaluate

        Returns:
            List of parameter tuples sorted by predicted score (best first)
        """
        if len(self.X) < self.min_samples_for_gp:
            return self.suggest_batch(n)

        candidates = [self._random_params() for _ in range(n_candidates)]
        X_cand = np.array([list(c) for c in candidates])
        mean = self.gp.predict(X_cand)

        # Get unique top predictions
        sorted_indices = np.argsort(mean)[::-1]
        selected = []
        selected_set = set()

        for idx in sorted_indices:
            if candidates[idx] not in selected_set:
                selected.append(candidates[idx])
                selected_set.add(candidates[idx])
                if len(selected) >= n:
                    break

        return selected

    def best_so_far(self) -> Optional[Dict]:
        """Return best result observed so far."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r['score'])

    def get_predictions_for_params(self, params_list: List[Tuple[int, ...]]) -> List[Tuple[float, float]]:
        """
        Get predictions for a list of parameter sets.

        Returns:
            List of (mean, std) tuples
        """
        return [self._predict(p) for p in params_list]

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics of optimization so far."""
        if not self.results:
            return {'n_observations': 0}

        scores = [r['score'] for r in self.results]
        best = self.best_so_far()

        return {
            'n_observations': len(self.results),
            'best_score': best['score'],
            'best_params': best['params'],
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'best_good': best['good'],
            'best_bad': best['bad'],
            'best_missed': best['missed'],
        }

    def save(self, path: str):
        """Save optimizer state to file."""
        state = {
            'gp': self.gp,
            'X': self.X,
            'y': self.y,
            'results': self.results,
            'param_range': self.param_range,
            'objective_name': self.objective.name,
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str, objective: ObjectiveFunction) -> 'GPOptimizer':
        """
        Load optimizer state from file.

        Args:
            path: Path to saved state
            objective: Objective function to use

        Returns:
            GPOptimizer with loaded state
        """
        opt = cls(objective)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                state = pickle.load(f)
            opt.gp = state['gp']
            opt.X = state['X']
            opt.y = state['y']
            opt.results = state['results']
            if 'param_range' in state:
                opt.param_range = state['param_range']
        return opt

    def get_history_dataframe(self):
        """
        Return optimization history as a pandas DataFrame.

        Requires pandas to be installed.
        """
        import pandas as pd

        rows = []
        for i, r in enumerate(self.results):
            row = {
                'iteration': i,
                'bad_1': r['params'][0],
                'bad_2': r['params'][1],
                'bad_3': r['params'][2],
                'bad_4': r['params'][3],
                'good': r['good'],
                'bad': r['bad'],
                'missed': r['missed'],
                'n_patterns': r['n_patterns'],
                'trie_nodes': r['trie_nodes'],
                'score': r['score'],
            }
            rows.append(row)

        return pd.DataFrame(rows)
