# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

hyph-bench is a benchmark dataset and tooling for generating hyphenation patterns using **patgen** (the TeX pattern generator). The main output is an ACL 2026 paper (`~/aclpaper.tex`) demonstrating Gaussian Process optimization for automatically finding optimal patgen parameters.

**Paper status (2026-03-12)**: Submitted to ACL 2026 ARR cycle. First round of reviews received (scores: 2, 1.5, 2). Preparing revisions for resubmission or Findings acceptance. Reviews are in `review1.txt`, `review2.txt`, `review3.txt`.

**Key result achieved**: GP optimization (profile GPopt4(p)) beats hand-tuned baselines on 11/18 datasets. Primary dataset cssk/cshyphen: F_{1/7}=0.9992 (+0.0006 over best baseline), trie size 18888 (-25% smaller).

**Primary dataset for paper**: cssk/cshyphen (Czech+Slovak, weighted, 835k entries).

### Reviewer feedback to address for resubmission

1. **Add budget-matched HPO baselines** (Random Search, TPE) - Reviewers ZGsw, 7caL
2. **Report variance/significance** - run GP with multiple seeds, report std - Reviewer ZGsw
3. **Ablate trie penalty** - adaptive vs fixed trie_normalizer - Reviewer ZGsw
4. **Analyze failure cases** (de/wortliste: 0.9452 vs 0.9775) - Reviewers 1LBL, ZGsw
5. **Clarify CV protocol** - not nested CV, state explicitly - Reviewer ZGsw
6. **Back efficiency claims with numbers** or remove "1000x" - Reviewer ZGsw

## Key Commands

```bash
# Run GP optimization (settings used for paper results)
uv run python -m scripts.optimize --lang cssk/cshyphen \
  --iterations 100 --batch-size 5 --objective f17_trie \
  --good-weight 3 --max-bad-weight 30 --max-threshold 1 \
  --ucb-kappa 2.5 --trie-weight 0.0005 --trie-normalizer 25000

# Run 10-fold cross-validation with specific parameters
uv run python -m scripts.cross_validate --lang cssk --params 5 1 6 9 1

# Direct cross-validation with a profile
uv run python scripts/train_test.py -t -v -n 10 -p profiles/base.in data/cssk/cshyphen

# Generate disambiguated wordlists and translate files
make disambiguate
make translate


# Run all 18 datasets sequentially (exact paper settings)
bash run_optimizations.sh

# Visualize GP regression surfaces
uv run python scripts/visualize_gp_2d.py --lang cssk --iteration 100
```

## Architecture

### Patgen Integration

Patgen is the TeX hyphenation pattern generator. It takes:
- A wordlist with hyphenation points marked by `-`
- A translate file (`.tr`) defining the character set and hyphen min values
- Parameters per level: `pat_start pat_finish good_weight bad_weight threshold`

Patgen runs in 4 levels (odd levels add patterns, even levels inhibit). Each level can have different pattern length ranges and weights.

### GP Optimization (`scripts/optimize.py`)

Optimizes 5 parameters: 4 bad_weights (one per level) + 1 shared threshold. Good_weight is fixed at 3 (paper setting).

The optimizer uses:
- `GPOptimizer` (sklearn GP with Matern kernel + UCB acquisition)
- `PatgenScorer` (runs patgen and parses output for good/bad/missed counts)
- Pluggable objectives in `scripts/objectives.py`:
  - `f17` - F_{1/7} score (default, weights precision 7x over recall)
  - `f17_trie` - F_{1/7} with trie size penalty for better generalization
  - `bounded_bad` - Hard constraint on bad count, minimize patterns
  - `min_size` - Minimize pattern count under bad threshold
  - `weighted` - Customizable weighted combination
  - `pr_curve` - Distance to ideal precision/recall

**Trie size optimization**: Use `--objective f17_trie` to balance F_{1/7} with trie size. Smaller tries tend to generalize better. Configure with `--trie-weight` (default: 0.0005) and `--trie-normalizer` (default: 25000).

### Dataset Structure

```
data/<lang>/<dataset>/
  *.wlhamb       - Ambiguous hyphenated wordlist source
  *.wlh          - Hyphenated wordlist (one word per line, hyphens mark break points)
  *.tr           - Translate file for patgen
```

**cssk/cshyphen** is special: its tracked source is `data/cssk/cshyphen/cssk_cshyphen.wlhamb`, a Czech+Slovak combined corpus with weights indicating frequency/confidence.

### Profile Format

Profiles in `profiles/` define patgen parameters per level:
```
pat_start pat_finish good_weight bad_weight threshold
```
Four lines for four levels.

## Important Quirks

- **Patgen handles conflicting hyphenations** in the wordlist gracefully by averaging - no need to deduplicate
- **F_{1/7}** weights precision 7x more than recall (bad hyphenations are worse than missed ones in typography)
- **Trie node count** correlates with generalization - smaller tries often generalize better
- The optimizer saves state to `results/<lang>_gp_state.pkl` and can resume with `--resume`
- Cross-validation creates temporary train/test splits in `data/<lang>/<dataset>/test/`

## Results Tracking

- `final_results_table.md` - Complete GP vs baseline comparison for all 18 datasets
- `OVERNIGHT_SUMMARY.md` - Documents the overnight run that produced paper results
- `tuning_notes.md` - Empirical findings on trie penalty, search bounds, kappa tuning
- `run_optimizations.sh` - Exact script used to produce paper results (100 iter, f17_trie)
- `results/<lang>_gp_state.pkl` - Saved optimizer states (can resume)
- `results/<lang>_history.csv` - Full optimization histories
- `optimization_logs/` - Full patgen logs from paper runs

## Paper-Specific Notes

The ACL paper (`~/aclpaper.tex`) has:
- Dataset descriptions and statistics (Tables 1, 2)
- Patgen parameter explanation (Table 3)
- Baseline profile definitions (Table 4)
- Cross-validation results with GP column (Tables 5, 6)
- GP visualizations in appendix (from `scripts/visualize_gp_2d.py`)
