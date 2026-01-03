#!/usr/bin/env python3
"""
Cross-validation for optimized patgen parameters.
Uses NFoldCrossValidator from scripts.train_test.
"""

import argparse
import os
import sys
from typing import List, Tuple

from .train_test import NFoldCrossValidator, extract_files
from .hyperparameters import combine, score, sample, metaheuristic
from .optimize import parse_profile, DEFAULT_PAT_RANGES, find_dataset


def create_dynamic_profile(params: List[int], pat_ranges: List[Tuple[int, int]], 
                           good_weight: int = 1, output_path: str = "dynamic.in"):
    """
    Create a profile file from optimized parameters.
    """
    if len(params) == 5:
        bad_weights = params[:4]
        threshold = params[4]
    else:
        bad_weights = params
        threshold = 1
        
    with open(output_path, "w") as f:
        for i, (ps, pf) in enumerate(pat_ranges):
            bw = bad_weights[i] if i < len(bad_weights) else bad_weights[-1]
            f.write(f"{ps} {pf} {good_weight} {bw} {threshold}\n")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Cross-validation for optimized parameters')
    parser.add_argument('--lang', required=True, help='Language code')
    parser.add_argument('--params', type=int, nargs='+', required=True,
                        help='Best parameters found (4 bad_weights + 1 threshold)')
    parser.add_argument('--profile', type=str, help='Base profile for pat_ranges')
    parser.add_argument('--nfold', type=int, default=10, help='Number of folds')
    parser.add_argument('--good-weight', type=int, default=1, help='Good weight')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Find dataset
    wl_path, tr_path = find_dataset(args.lang)
    print(f"Dataset: {wl_path}")
    
    # Get pattern ranges
    if args.profile:
        pat_ranges = parse_profile(args.profile)
    else:
        pat_ranges = DEFAULT_PAT_RANGES
        
    # Create temporary dynamic profile
    profile_path = f"results/{args.lang}_dynamic.in"
    create_dynamic_profile(args.params, pat_ranges, args.good_weight, profile_path)
    print(f"Created dynamic profile: {profile_path}")
    
    # Setup validator
    scorer = score.PatgenScorer("patgen", "", tr_path, verbose=args.verbose)
    sampler = sample.FileSampler(profile_path)
    meta = metaheuristic.NoMetaheuristic(scorer, sampler)
    combiner = combine.SimpleCombiner(meta, verbose=args.verbose)
    
    validator = NFoldCrossValidator(combiner, tr_path, args.nfold)
    print(f"Running {args.nfold}-fold cross-validation...")
    validator.validate(wl_path, verbose=args.verbose)
    
    # Report results
    lang_name = os.path.basename(os.path.dirname(wl_path))
    ds_name = os.path.basename(wl_path).replace("_dis.wlh", "").replace(".wlh", "")
    
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    print(validator.report(lang=lang_name, name=ds_name, profile="dynamic", tabular=True))
    print("="*60)
    
    # Cleanup (optional, keeping profile for reference)
    # os.remove(profile_path)

if __name__ == "__main__":
    main()
