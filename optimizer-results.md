# GP Optimizer Results for Patgen Hyphenation Parameters

## Executive Summary

The Gaussian Process optimizer was tested on the Polish (pl) dataset to find optimal `bad_weight` parameters for 4-level patgen runs. **The existing preset parameters significantly outperformed the GP-optimized parameters**, revealing important limitations of the optimization approach.

| Configuration | bad_weights | F_1/7 Score | Good | Bad | Missed | Patterns |
|--------------|-------------|-------------|------|-----|--------|----------|
| **Preset (base.in)** | (1, 2, 3, 4) | **0.9993** | 311,602 | **219** | 380 | 7,653 |
| GP Best (grid) | (9, 1, 9, 1) | 0.9940 | 305,558 | 1,758 | 6,424 | 2,477 |

**Key finding**: The preset achieves 8x fewer bad hyphenations (219 vs 1,758) and 17x fewer missed hyphenations (380 vs 6,424).

---

## Test Configuration

- **Dataset**: Polish Wiktionary (`pl_enwiktionary_251001.wlh_dis.wlh`)
- **Pattern ranges**: [(2,4), (3,5), (4,7), (5,9)] for levels 1-4
- **Objective**: F_1/7 (precision weighted 7x more than recall)
- **Parameter space**: `bad_weight` in [1-9] for each of 4 levels (9^4 = 6,561 combinations)
- **GP warmup**: Coarse grid with values [1, 5, 9] = 81 combinations
- **GP iterations**: 10 iterations with batch size 5

---

## Detailed Results

### Preset Baseline (base.in)

The `base.in` profile uses incrementally increasing bad_weights:

```
Level 1: pat_start=2, pat_finish=4, good=1, bad=1, threshold=1
Level 2: pat_start=3, pat_finish=5, good=1, bad=2, threshold=1
Level 3: pat_start=4, pat_finish=7, good=1, bad=3, threshold=1
Level 4: pat_start=5, pat_finish=9, good=1, bad=4, threshold=1
```

**Results**:
- Good (true positives): 311,602
- Bad (false positives): 219
- Missed (false negatives): 380
- Total patterns: 7,653
- Trie nodes: 10,554
- **F_1/7 Score: 0.999287**
- Precision: 99.93%
- Recall: 99.88%

### GP Optimizer Results

#### Coarse Grid Warmup (81 points)

The coarse grid searched values [1, 5, 9] for each parameter. Best results from grid:

| Rank | bad_weights | F_1/7 | Good | Bad | Missed |
|------|-------------|-------|------|-----|--------|
| 1 | (9, 1, 9, 1) | 0.9940 | 305,558 | 1,758 | 6,424 |
| 2 | (9, 1, 9, 5) | 0.9939 | 305,572 | 1,784 | 6,410 |
| 3 | (9, 1, 9, 9) | 0.9938 | 305,574 | 1,802 | 6,408 |
| 4 | (9, 1, 5, 1) | 0.9936 | 306,011 | 1,898 | 5,971 |
| 5 | (9, 1, 5, 5) | 0.9935 | 306,027 | 1,924 | 5,955 |

**Observation**: All top results have `bad_1=9` and `bad_2=1`, suggesting high penalty on level 1 and low on level 2. However, these configurations have significantly more errors than the preset.

#### GP Optimization Phase (10 iterations)

After the grid warmup, the GP attempted to refine the search but **got stuck on the local optimum (9, 1, 9, 1)**. The GP kept re-testing this point across multiple iterations without finding improvements.

Sample GP-tested configurations:
- (9, 1, 8, 4): score=0.9939
- (9, 2, 8, 1): score=0.9936
- (9, 1, 7, 7): score=0.9937
- (7, 1, 2, 3): score=0.9919

None exceeded the grid best of 0.9940.

---

## Analysis

### Why Did the Optimizer Fail to Find the Optimal Region?

1. **Coarse Grid Missed the Sweet Spot**: The grid values [1, 5, 9] completely skipped the optimal region around [1-4]. The preset uses values 1, 2, 3, 4 - all below the lowest grid step of 5 for later levels.

2. **GP Got Trapped in Local Optimum**: Once the grid established (9, 1, 9, 1) as the best, the GP's UCB acquisition function kept exploiting this region rather than exploring toward lower values.

3. **Misleading Objective Landscape**: The F_1/7 score heavily weights precision. Configurations with high `bad_1` (like 9) aggressively reject hyphenation points, achieving low bad counts at the cost of many missed hyphenations.

### Score vs Quality Trade-off

| Metric | Preset (1,2,3,4) | GP Best (9,1,9,1) | Difference |
|--------|------------------|-------------------|------------|
| F_1/7 Score | 0.9993 | 0.9940 | -0.53% |
| Bad (FP) | 219 | 1,758 | +703% |
| Missed (FN) | 380 | 6,424 | +1,590% |
| Patterns | 7,653 | 2,477 | -68% |

The GP configuration produces fewer patterns but at a significant quality cost.

---

## Recommendations

### 1. Use Finer Grid Steps
Replace step=4 (giving [1,5,9]) with step=2 (giving [1,3,5,7,9]) to capture the low-value region:
```bash
uv run python -m scripts.optimize --lang pl --coarse-grid --grid-step 2
```

### 2. Include Known Good Points
Seed the optimizer with the preset values:
```python
# Add to optimizer initialization
optimizer.update((1,2,3,4), ...)  # Preset baseline
```

### 3. Increase Exploration
The UCB kappa of 2.0 wasn't enough to escape the local optimum. Consider:
- Higher kappa (3.0-5.0) for more exploration
- Random restarts
- Multi-start optimization

### 4. Consider Alternative Objectives
- **bounded_bad**: Hard constraint on bad count, then minimize patterns
- **pr_curve**: Balanced precision-recall distance

---

## Conclusion

**The preset parameters (1, 2, 3, 4) remain the best choice for Polish hyphenation**, achieving near-perfect precision (99.93%) and recall (99.88%) with only 219 false positives and 380 false negatives.

The GP optimizer's failure to find this optimum highlights the importance of:
1. Proper grid initialization that covers the likely optimal region
2. Including domain knowledge (existing good solutions) in the search
3. Sufficient exploration to escape local optima

The optimizer infrastructure is sound and can be improved with the recommendations above. For production use, consider seeding with preset values and using finer grid steps.

---

## Appendix: Running the Optimizer

```bash
# Basic optimization with F_1/7
uv run python -m scripts.optimize --lang pl --iterations 50

# With coarse grid warmup
uv run python -m scripts.optimize --lang pl --coarse-grid --iterations 20

# With finer grid
uv run python -m scripts.optimize --lang pl --coarse-grid --grid-step 2 --iterations 20

# Resume interrupted run
uv run python -m scripts.optimize --lang pl --resume --iterations 20
```

## Test Environment

- Date: 2025-12-07
- Platform: Linux (Fedora 43)
- Python: 3.14
- Dataset size: ~312,000 hyphenation points
- Evaluation time: ~13 seconds per configuration
