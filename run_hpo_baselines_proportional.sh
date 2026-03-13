#!/usr/bin/env bash
# Re-run HPO comparison (GP vs Random vs TPE) with dataset-proportional
# trie normalizer (N = |D|, number of wordlist lines per dataset).
#
# This addresses the interaction between Task 1 (HPO baselines) and Task 2
# (trie penalty ablation): the fixed N=25,000 used in Task 1 miscalibrates
# the objective for datasets whose size differs substantially from 25k.
#
# IMPORTANT: No --reuse-existing-gp here. All three methods run fresh
# under the corrected objective for a fair comparison.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Expected virtualenv python at ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  DATASETS=(
    "cssk/cshyphen"
    "cs/cshyphen_cstenten"
    "de/wortliste"
    "th/orchid"
    "el/wiktionary"
    "es/wiktionary"
    "ms/wiktionary"
  )
else
  DATASETS=("$@")
fi

exec "${PYTHON_BIN}" -m scripts.compare_hpo_methods \
  --datasets "${DATASETS[@]}" \
  --methods gp random tpe \
  --objective f17_trie \
  --iterations 100 \
  --batch-size 1 \
  --good-weight 3 \
  --max-bad-weight 30 \
  --max-threshold 1 \
  --ucb-kappa 2.5 \
  --trie-weight 0.0005 \
  --proportional-normalizer \
  --nfold 10 \
  --output-dir results/hpo_baselines_proportional
