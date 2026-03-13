#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Expected virtualenv python at ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  DATASETS=(
    "ms/wiktionary"
    "el/wiktionary"
    "th/orchid"
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
  --trie-normalizer 25000 \
  --nfold 10 \
  --reuse-existing-gp \
  --output-dir results/hpo_baselines
