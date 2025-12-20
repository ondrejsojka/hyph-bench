#!/usr/bin/env python3
"""
Gaussian Process optimization for patgen parameters.

This script optimizes the bad_weight parameters for a 4-level patgen run
using Gaussian Process regression with Upper Confidence Bound acquisition.

Usage:
    python -m scripts.optimize --lang pl --iterations 50
    python -m scripts.optimize --lang uk --objective bounded_bad --bad-threshold 500
    python -m scripts.optimize --lang pl --resume --iterations 20

The optimizer searches over a 4-dimensional space:
    bad_1, bad_2, bad_3, bad_4 in [1, 9]

With fixed values:
    good_weight = 1 (all layers)
    threshold = 5 (configurable)
    pat_start, pat_finish from profile
"""

import argparse
import os
import sys
from typing import Tuple, Optional, List

from .gp_optimizer import GPOptimizer
from .objectives import get_objective
from .hyperparameters.score import PatgenScorer
from .hyperparameters.sample import Sample


# Default pattern ranges per level (from base.in profile)
DEFAULT_PAT_RANGES = [
    (1, 4),   # Level 1
    (2, 5),   # Level 2
    (2, 6),   # Level 3
    (2, 7),   # Level 4
]


def find_dataset(lang: str, data_dir: str = None) -> Tuple[str, str]:
    """
    Find wordlist and translate file for a language.

    Args:
        lang: Language code (e.g., 'pl', 'uk', 'cs')
        data_dir: Optional override for data directory

    Returns:
        (wordlist_path, translate_path) tuple

    Raises:
        FileNotFoundError: If no dataset found
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', lang)

    # Check wiktionary subdirectory first (preferred)
    wikt_dir = os.path.join(data_dir, 'wiktionary')
    if os.path.exists(wikt_dir):
        for f in sorted(os.listdir(wikt_dir)):
            if f.endswith('_dis.wlh'):
                wl = os.path.join(wikt_dir, f)
                tr = wl + '.tra'
                if os.path.exists(tr):
                    return os.path.abspath(wl), os.path.abspath(tr)

    # Check data directory directly
    for f in sorted(os.listdir(data_dir)):
        if f.endswith('.wlh'):
            wl = os.path.join(data_dir, f)
            tr = wl + '.tra'
            if os.path.exists(tr):
                return os.path.abspath(wl), os.path.abspath(tr)

    raise FileNotFoundError(f"No dataset found for language: {lang} in {data_dir}")


def parse_profile(profile_path: str) -> List[Tuple[int, int]]:
    """
    Parse a profile file to get pat_start/pat_finish per level.

    Profile format (per line): pat_start pat_finish good_weight bad_weight threshold

    Args:
        profile_path: Path to profile file

    Returns:
        List of (pat_start, pat_finish) tuples
    """
    pat_ranges = []
    with open(profile_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                pat_ranges.append((int(parts[0]), int(parts[1])))
    return pat_ranges


def run_patgen_multilevel(scorer: PatgenScorer, bad_weights: Tuple[int, ...],
                          pat_ranges: List[Tuple[int, int]],
                          good_weight: int = 1,
                          threshold: int = 5) -> dict:
    """
    Run full multi-level patgen with given bad weights.

    Args:
        scorer: PatgenScorer instance
        bad_weights: (bad_1, bad_2, bad_3, bad_4) tuple
        pat_ranges: List of (pat_start, pat_finish) per level
        good_weight: Good weight for all levels
        threshold: Threshold for all levels

    Returns:
        Dict with good, bad, missed, n_patterns, trie_nodes
    """
    prev_id = 0
    total_patterns = 0
    final_stats = None

    n_levels = min(len(bad_weights), len(pat_ranges))

    for level in range(1, n_levels + 1):
        bad_wt = bad_weights[level - 1]
        ps, pf = pat_ranges[level - 1]

        sample = Sample({
            'level': level,
            'prev': prev_id,
            'pat_start': ps,
            'pat_finish': pf,
            'good_weight': good_weight,
            'bad_weight': bad_wt,
            'threshold': threshold
        })

        scorer.score(sample)
        prev_id = sample.run_id
        total_patterns += sample.stats.get('level_patterns', 0)
        final_stats = sample.stats

    return {
        'good': final_stats['tp'],
        'bad': final_stats['fp'],
        'missed': final_stats['fn'],
        'n_patterns': total_patterns,
        'trie_nodes': final_stats['trie_nodes']
    }


def main():
    parser = argparse.ArgumentParser(
        description='GP optimization for patgen parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument('--lang', required=True,
                        help='Language code (e.g., pl, uk, cs, de)')

    # Objective configuration
    parser.add_argument('--objective', default='f17',
                        choices=['f17', 'bounded_bad', 'weighted', 'pr_curve'],
                        help='Objective function (default: f17)')
    parser.add_argument('--bad-threshold', type=int, default=500,
                        help='Bad threshold for bounded_bad objective (default: 500)')
    parser.add_argument('--beta', type=float, default=1/7,
                        help='Beta for F-score (default: 1/7 for F_1/7)')

    # Optimization parameters
    parser.add_argument('--iterations', type=int, default=50,
                        help='Number of optimization iterations (default: 50)')
    parser.add_argument('--batch-size', type=int, default=5,
                        help='Suggestions per iteration (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--ucb-kappa', type=float, default=2.0,
                        help='UCB exploration weight (default: 2.0)')
    parser.add_argument('--coarse-grid', action='store_true',
                        help='Run coarse grid warmup before GP optimization')
    parser.add_argument('--grid-step', type=int, default=4,
                        help='Grid step size (default: 4 gives [1,5,9])')

    # Patgen parameters
    parser.add_argument('--threshold', type=int, default=5,
                        help='Patgen threshold for all levels (default: 5)')
    parser.add_argument('--good-weight', type=int, default=1,
                        help='Patgen good_weight for all levels (default: 1)')
    parser.add_argument('--profile', type=str,
                        help='Profile file for pat_start/pat_finish')

    # Data paths
    parser.add_argument('--wordlist', type=str,
                        help='Override wordlist path')
    parser.add_argument('--translate', type=str,
                        help='Override translate file path')
    parser.add_argument('--patgen', type=str, default='patgen',
                        help='Path to patgen binary (default: patgen)')

    # Output and state
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results (default: results)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from saved state')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # Setup objective
    if args.objective == 'f17':
        objective = get_objective('f17', beta=args.beta)
    elif args.objective == 'bounded_bad':
        objective = get_objective('bounded_bad', bad_threshold=args.bad_threshold)
    else:
        objective = get_objective(args.objective)

    print(f"Objective: {objective.name}")

    # Find dataset
    if args.wordlist and args.translate:
        wl_path, tr_path = args.wordlist, args.translate
    else:
        wl_path, tr_path = find_dataset(args.lang)

    print(f"Wordlist: {wl_path}")
    print(f"Translate: {tr_path}")

    # Parse profile for pat_ranges
    if args.profile:
        pat_ranges = parse_profile(args.profile)
    else:
        pat_ranges = DEFAULT_PAT_RANGES

    print(f"Pattern ranges: {pat_ranges}")

    # Setup scorer
    scorer = PatgenScorer(args.patgen, wl_path, tr_path, verbose=args.verbose)

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    state_path = os.path.join(args.output_dir, f'{args.lang}_gp_state.pkl')
    csv_path = os.path.join(args.output_dir, f'{args.lang}_history.csv')

    # Determine min samples for GP based on coarse grid
    if args.coarse_grid:
        grid = GPOptimizer(objective, seed=args.seed).generate_coarse_grid(args.grid_step)
        min_samples = len(grid)
        print(f"Coarse grid enabled: {min_samples} grid points")
    else:
        min_samples = 5

    # Setup optimizer
    if args.resume and os.path.exists(state_path):
        optimizer = GPOptimizer.load(state_path, objective)
        optimizer.min_samples_for_gp = min_samples
        print(f"Resumed from {len(optimizer.X)} observations")
    else:
        optimizer = GPOptimizer(objective, seed=args.seed, min_samples_for_gp=min_samples)
        print("Starting fresh optimization")

    # Coarse grid warmup phase
    if args.coarse_grid and len(optimizer.X) < min_samples:
        grid = optimizer.generate_coarse_grid(args.grid_step)
        remaining = [g for g in grid if list(g) not in optimizer.X]
        print(f"\n{'=' * 60}")
        print(f"COARSE GRID WARMUP: {len(remaining)} points to evaluate")
        print(f"{'=' * 60}")

        for i, params in enumerate(remaining):
            print(f"  Grid [{i+1}/{len(remaining)}]: bad_weights={params}")

            scorer.reset()
            results = run_patgen_multilevel(
                scorer, params, pat_ranges,
                good_weight=args.good_weight,
                threshold=args.threshold
            )
            score = optimizer.update(params, **results)

            print(f"    good={results['good']}, bad={results['bad']}, "
                  f"missed={results['missed']}, patterns={results['n_patterns']}, "
                  f"score={score:.4f}")

            # Save state periodically
            if (i + 1) % 10 == 0:
                optimizer.save(state_path)
                best = optimizer.best_so_far()
                print(f"\n  [Checkpoint] Best so far: {best['params']} -> {best['score']:.4f}\n")

        optimizer.save(state_path)
        best = optimizer.best_so_far()
        print(f"\n{'=' * 60}")
        print(f"GRID WARMUP COMPLETE")
        print(f"Best from grid: {best['params']} -> {best['score']:.4f}")
        print(f"  good={best['good']}, bad={best['bad']}, missed={best['missed']}")
        print(f"{'=' * 60}")

    # Main optimization loop
    try:
        for iteration in range(args.iterations):
            print(f"\n{'=' * 60}")
            print(f"Iteration {iteration + 1}/{args.iterations}")

            # Get suggestions
            suggestions = optimizer.suggest_batch(
                args.batch_size,
                ucb_kappa=args.ucb_kappa
            )

            for params in suggestions:
                print(f"  Testing: bad_weights={params}")

                # Reset scorer for clean run
                scorer.reset()

                # Run patgen
                results = run_patgen_multilevel(
                    scorer, params, pat_ranges,
                    good_weight=args.good_weight,
                    threshold=args.threshold
                )

                # Update optimizer
                score = optimizer.update(params, **results)

                print(f"    good={results['good']}, bad={results['bad']}, "
                      f"missed={results['missed']}, patterns={results['n_patterns']}, "
                      f"score={score:.4f}")

            # Report best so far
            best = optimizer.best_so_far()
            if best:
                print(f"\n  Best so far: {best['params']} -> {best['score']:.4f}")
                print(f"    good={best['good']}, bad={best['bad']}, missed={best['missed']}")

            # Save state after each iteration
            optimizer.save(state_path)

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving state...")
        optimizer.save(state_path)

    # Final exploitation phase
    print(f"\n{'=' * 60}")
    print("Final exploitation phase")

    best_params = optimizer.exploit_best(n=3)

    for params in best_params:
        scorer.reset()
        results = run_patgen_multilevel(
            scorer, params, pat_ranges,
            good_weight=args.good_weight,
            threshold=args.threshold
        )
        score = optimizer.update(params, **results)

        print(f"  {params}: good={results['good']}, bad={results['bad']}, "
              f"patterns={results['n_patterns']}, score={score:.4f}")

    # Final report
    best = optimizer.best_so_far()
    print(f"\n{'=' * 60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Best parameters: {best['params']}")
    print(f"  bad_1={best['params'][0]}, bad_2={best['params'][1]}, "
          f"bad_3={best['params'][2]}, bad_4={best['params'][3]}")
    print(f"Results:")
    print(f"  good={best['good']}, bad={best['bad']}, missed={best['missed']}")
    print(f"  n_patterns={best['n_patterns']}, trie_nodes={best['trie_nodes']}")
    print(f"  score={best['score']:.4f}")

    # Save final state and history
    optimizer.save(state_path)
    print(f"\nState saved to: {state_path}")

    # Export history to CSV
    try:
        df = optimizer.get_history_dataframe()
        df.to_csv(csv_path, index=False)
        print(f"History saved to: {csv_path}")
    except ImportError:
        print("(pandas not available, skipping CSV export)")

    # Cleanup
    scorer.clean()


if __name__ == '__main__':
    main()
