# Hyph-bench: Benchmark Dataset of Hyphenated Words for Generating Hyphenation Patterns

version 1.0

Raw data and scripts for creating and evaluating hyphenated word lists in several languages.

## Repository structure

### data/
Directory with initial versions of non-Wiktionary datasets

### Makefile
Definition of helpful batch commands (`*` = `wikt` for Wiktionary datasets / `other` for other datasets):
`process_wikt`: unzip Wiktionary dump files and process them into initial word lists, which are stored in `data/`
`prepare_*`: perform initial preprocessing
`disambiguate_*`: eliminate ambiguous hyphenations
`translate_*`: create translate files necessary for **patgen** program
`stats_all_datasets`: compile statistics of all datasets
`cross_validate_all`: perform 10-fold cross-validation over all datassets with baseline profiles

### profiles/
Baseline parameter profiles.

### scripts/
Python scripts and packages used for data preprocessing, evaluation, and reporting.

### wikt_dump.zip
Compressed directory with JSON dump files of Wiktionary datasets.
