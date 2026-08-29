# hyph-bench

`hyph-bench` generates TeX-compatible hyphenation patterns and evaluates their accuracy and trie size. The accompanying paper uses Gaussian-process optimization to replace manual PATGEN profile tuning with a reproducible train, validation, and held-out test workflow.

The repository contains:

- curated and Wiktionary-derived hyphenated word lists;
- PATGEN preprocessing and evaluation tools;
- GP, TPE, and Random Search optimizers;
- the per-level Gaussian-process paper protocol and dated `gpopt260828` batch runner;
- generated histories, selected profiles, and analysis artifacts.

## Requirements

You need:

- Python 3.10 or newer;
- [uv](https://docs.astral.sh/uv/);
- `patgen` from a recent TeX Live installation.

On Debian and Ubuntu, the `texlive-binaries` package provides `patgen`. Other TeX Live distributions usually install it with the core binaries.

Install the Python environment:

```bash
uv sync
```

Check the PATGEN path:

```bash
command -v patgen
```

The paper runs use `/home/dev/patgen-10x`, a local name for a TeX Live 2024 PATGEN build with the upstream Web2C capacity settings: a 10,000,000-entry pattern trie, a 5,000,000-entry count trie, and 40,800 output operations. A recent TeX Live source build applies these values through `texk/web2c/patgen.ch`. Smaller datasets work with the standard packaged binary. Large datasets may require the higher-capacity build.

Pass a non-default binary with `--patgen /path/to/patgen` or set `PATGEN_BIN` when a batch script supports it.

## Run a smoke optimization

This command exercises the full train, validation, selection, held-out test, and pattern-export path on the Thai ORCHID dataset:

```bash
uv run python -m scripts.optimize_validation \
  --lang th/orchid \
  --patgen "$(command -v patgen)" \
  --iterations 1 \
  --batch-size 1 \
  --objective f17_trie \
  --good-weight 3 \
  --max-bad-weight 30 \
  --max-threshold 1 \
  --ucb-kappa 2.5 \
  --trie-weight 0.0005 \
  --output-dir /tmp/hyph-bench-smoke \
  --export-final-patterns
```

The command writes:

- deterministic 8/1/1 splits under `/tmp/hyph-bench-smoke/th/orchid/splits/`;
- `gpoptval4_history.csv` with every validation evaluation;
- `gpoptval4_state.pkl` for resuming a run;
- `gpoptval4_final.pat`, selected on validation data and evaluated once on the held-out test split.

A one-iteration smoke run does not reproduce a paper score. Use the full protocol below for reported results.

## Understand the data format

A dataset lives under `data/<language>/<name>/`. It contains:

- a `.wlh` word list with one entry per line;
- hyphens at allowed break positions, for example `hy-phen-a-tion`;
- a matching `.tra` file that defines PATGEN characters and left and right hyphen minima.

Some source datasets use:

- `*_dis.wlh` for disambiguated entries;
- `*.wlhw` for weighted entries;
- `*_expanded.wlh` after expanding weights.

Generate a translate file for a new word list:

```bash
uv run python -m scripts.make_tr data/xx/example/example.wlh
```

This writes `data/xx/example/example.wlh.tra` with left and right minima of 2. Override them when the orthography requires different values:

```bash
uv run python -m scripts.make_tr \
  data/xx/example/example.wlh \
  --left_hyphen_min 2 \
  --right_hyphen_min 3
```

## Optimize patterns for a new dataset

You can place the files under `data/<language>/<name>/` and use `--lang`, or pass explicit paths. Explicit paths still require a label for the output directory:

```bash
uv run python -m scripts.optimize_validation \
  --lang xx/example \
  --wordlist /absolute/path/example.wlh \
  --translate /absolute/path/example.wlh.tra \
  --patgen "$(command -v patgen)" \
  --iterations 30 \
  --batch-size 5 \
  --objective f17_trie \
  --good-weight 3 \
  --max-bad-weight 30 \
  --max-threshold 1 \
  --ucb-kappa 2.5 \
  --trie-weight 0.0005 \
  --output-dir results/gpoptval4 \
  --export-final-patterns
```

The optimizer trains patterns on 80% of the entries, selects parameters on 10%, and reports the selected profile once on the remaining 10%. The split uses the input line index modulo 10 and is deterministic. Do not reorder the word list between related runs.

The objective is

$$
F_{1/7} - 0.0005\frac{\text{trie nodes}}{|D|}.
$$

$F_{1/7}$ weights precision more strongly than recall because an incorrect hyphen is usually worse than a missed optional break. Dividing trie size by the dataset size keeps the compactness penalty comparable across datasets.

### Optimize shared parameters

The shared-parameter search optimizes four level-specific bad weights, one shared threshold, and one shared good weight:

```bash
uv run python -m scripts.optimize_shared_parameters \
  --lang xx/example \
  --wordlist /absolute/path/example.wlh \
  --translate /absolute/path/example.wlh.tra \
  --patgen "$(command -v patgen)" \
  --output-dir results/shared_parameter_search \
  --iterations 30 \
  --batch-size 5 \
  --seed 42 \
  --ucb-kappa 2.5 \
  --objective f17_trie \
  --trie-weight 0.0005 \
  --export-final-patterns
```

The default bounds are:

- each `bad_wt` in $[1,30]$;
- `threshold` in $\{1,2\}$;
- `good_wt` in $\{1,2,3,4,5\}$.

PATGEN evaluates five candidates in parallel during each iteration. A full run therefore uses five worker processes and 153 evaluations: 150 search evaluations plus three final exploitation evaluations.

## Apply generated patterns

Apply a generated `.pat` file to a plain word list:

```bash
uv run python -m scripts.hyphenate_wordlist \
  --wordlist words.txt \
  --patterns results/shared_parameter_search/xx/example/wider_final.pat \
  --translate /absolute/path/example.wlh.tra \
  --output words.hyphenated.txt
```

The command uses the repository's Liang-pattern implementation. It does not substitute a language-specific dictionary from another library.

## Reproduce the paper experiments

The final per-level search uses the following protocol for every manuscript dataset:

- deterministic 8/1/1 train, validation, and held-out test split;
- four independently selected weight ratios, one per PATGEN level;
- four independently selected thresholds, one per PATGEN level;
- weight ratios in `{1/5, 1/4, 1/3, 1/2, 1, ..., 30}`;
- thresholds in `[1,42]`;
- pattern ranges `[(1,4), (2,5), (2,6), (2,7)]`;
- 30 GP iterations with batches of 5, followed by three exploitation evaluations;
- seed 42 and UCB $\kappa=2.5$;
- proportional trie normalization by $|D|$ and `trie_weight=0.0005`;
- selection by the best observed validation objective;
- one held-out test evaluation after selection.

The dated repository identifier for the final 17-dataset run is `gpopt260828`.
The paper describes it generically as the per-level GP search and does not use
the dated identifier as a method name.

Run the full matrix or fill missing datasets:

```bash
PATGEN_BIN=/path/to/high-capacity/patgen \
  bash scripts/run_paper2_gpopt260828.sh
```

By default, the runner writes to `results/gpopt260828/`. Each complete dataset
contains:

- `run_config.json`, recording the exact command and search space;
- `final_history.csv`, with 153 evaluated profiles;
- `selected_profile.json`, with validation selection and held-out metrics;
- `final_patterns.pat`, the selected deployable pattern set;
- deterministic split files and optimizer state.

The result directory is gitignored by default. For a paper release, add only
the reviewed lightweight configurations, histories, selected profiles,
patterns, and aggregate evidence; do not publish machine-specific optimizer
pickle state.

Historical GPTopt8 artifacts remain under `results/gptopt8/`. Its launcher
pins `PAPER2_WEIGHT_SPACE=legacy`, preserving the original fractional choices
`{1/3, 1/2}` and threshold range `[1,5]`. The dated final runner uses the
extended default space. Do not combine histories from the two spaces.

Related scripts:

| Purpose | Command or script |
|---|---|
| Final per-level GP search | `python -m scripts.paper2_final_search` |
| Final 17-dataset queue | `scripts/run_paper2_gpopt260828.sh` |
| Historical GPTopt8 queue | `scripts/run_paper2_gptopt8.sh` |
| GP, TPE, Random comparison | `python -m scripts.compare_hpo_methods` |

The budget-matched GP/TPE/Random experiment uses a separate restricted
four-parameter space. Its results must not be presented as repeated runs or
uncertainty estimates for the final per-level search.

## Preprocess the bundled datasets

The Makefile provides bulk preprocessing targets:

```bash
make prepare_wikt       # Extract wikt_dump.zip into wikt_dump/
make process_wikt       # Convert JSONL dumps to .wlh files
make disambiguate_all   # Remove conflicting annotations
make translate_all      # Generate PATGEN .tra files
make stats_all_datasets # Report dataset statistics
```

`make process_wikt` reads a large archive and can require substantial disk space and time. You do not need to rerun it to use the curated files already present under `data/`.

## Repository layout

- `data/`: hyphenated word lists and translate files.
- `profiles/`: hand-tuned PATGEN baselines.
- `scripts/`: preprocessing, optimization, evaluation, and reporting code.
- `results/shared_parameter_search/`: shared-parameter histories and selected patterns.
- `results/shared_parameter_analysis/`: camera-ready aggregate evidence.
- `thesis/`: earlier thesis-specific experiments; these are not the canonical paper workflow.

## Dataset licenses

The repository combines datasets with different licenses. Preserve the attribution and license of each source when redistributing data or generated derivatives.

| Dataset | License | Source note |
|---|---|---|
| `cs/cshyphen_cstenten` | CC BY-NC-SA 3.0 | Czech–Slovak curated data |
| `cs/cshyphen_ujc` | MIT | Czech curated data |
| `cssk/cshyphen` | MIT | Weighted Czech–Slovak data |
| `de/wortliste` | MIT | German curated data |
| `is/hyphenation-is` | CC BY 4.0 | Icelandic curated data |
| `th/orchid` | CC BY-SA 4.0 | Licensed in 2025 from the public-domain ORCHID source |
| Wiktionary-derived datasets | CC BY-SA 4.0 | `cs`, `de`, `el`, `es`, `it`, `ms`, `nl`, `pl`, `pt`, `ru`, and `tr` |
| `uk/wiktionary` | CC BY-SA 4.0 | Prepared for the cited Ukrainian hyphenation thesis |
| `uk/dict_uk` | GPL-3.0 | Derived from [brown-uk/dict_uk](https://github.com/brown-uk/dict_uk) |

The original software in this repository is available under the [MIT License](LICENSE). Dataset files retain the separate licenses and attribution requirements listed above; the software license does not override those terms.

## Known scope boundaries

- `scripts.optimize_validation` and `scripts.optimize_shared_parameters` define the canonical held-out optimization workflows.
- `scripts.optimize` performs in-sample optimization and serves older experiments. Do not use it to reproduce held-out camera-ready results.
- `thesis/` contains older workflows and may require thesis-specific inputs that are not part of the paper artifact.
