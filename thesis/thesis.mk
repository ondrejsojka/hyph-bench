SUK_TRIE_WEIGHT=0.00085
SUK_TRIE_NORMALIZER=15714

RESULTS_DIR=thesis/results
FINAL_PATTERNS=thesis/pattern_evaluation_dataset/patterns/optimized.pat

ANNOTATIONS=$(wildcard thesis/annotation_results/*)
PATTENRS=$(wildcard thesis/pattern_evaluation_dataset/patterns/*)

# Reproduce all results
thesis: suk531 kappa_table indistinct optimization_results evaluate_patterns

clean_thesis: 
	rm -rf $(RESULTS_DIR)

# Creates an table comparing the final patterns against Polyakov 
evaluate_patterns:
	cp $(FINAL_PATTERNS) thesis/pattern_evaluation_dataset/patterns/uk_final.pat
	python thesis/utils/evaluate_patterns.py \
		thesis/pattern_evaluation_dataset/evaluation.wl \
		--truth thesis/pattern_evaluation_dataset/human1.wl \
		--patterns $(PATTENRS) \
		--output $(RESULTS_DIR)/pattern_evaluation.tex

# Outputs the results with all experiments conducted on FUK
optimization_results: $(RESULTS_DIR)
	./thesis/utils/run_batches.sh thesis/batch_configs/results_table.json --collect-results
	mv results/results.csv $(RESULTS_DIR)
	mv results/results_table.tex $(RESULTS_DIR)

# Performs the indstinguishibility test between the models and the human annotators
indistinct: $(RESULTS_DIR)
	python -m thesis.utils.indistinguishability_test \
    	--model-table $(RESULTS_DIR)/model_gap.tex \
    	--word-table  $(RESULTS_DIR)/word_disagreement.tex

# Outputs the Cohen's kappa table describing similarity between the annotations
kappa_table: $(RESULTS_DIR)
	python -m scripts.compare_annotations $(ANNOTATIONS) > $(RESULTS_DIR)/annotation_comparision_table.tex

# Outputs the 531 words which were used for annotation
suk531: $(RESULTS_DIR)
	python -m scripts.optimize --lang uk \
		--trie-weight $(SUK_TRIE_WEIGHT) \
		--trie-normalizer $(SUK_TRIE_NORMALIZER) \
		--export-iteration-results
	mv results/uk_bad.txt $(RESULTS_DIR)/suk531.wl

$(RESULTS_DIR): 
	mkdir -p $(RESULTS_DIR)

# Recreate the universal patterns experiment
# REQUIRES THE WIKTIONARY DATASETS FROM THE wikt_dump.zip FILE
# AVAILABLE IN THE HYPH-BENCH REPOSITORY https://github.com/tondach01/hyph-bench
universal_patterns: $(RESULTS_DIR)
	python thesis/utils/hyphenate.py $(FINAL_PATTERNS) data/uk/dict_uk/uk_full_dictuk.wl $(RESULTS_DIR)/uk_full_dictuk.wlh
	python thesis/utils/merge_wlh.py \
			data/cssk/cshyphen/cssk-all-weighted_dis.wlhw \
			$(RESULTS_DIR)/uk_full_dictuk_dis.wlh \
			data/ru/wiktionary/ru_wiktionary_dis.wlh \
			data/pl/wiktionary/pl_enwiktionary_dis.wlh \
			-o $(RESULTS_DIR)/merged.wlh \
			-c $(RESULTS_DIR)/collisions
	python -m scripts.optimize --lang uk \
		--wordlist $(RESULTS_DIR)/merged.wlh \
		--translate thesis/pattern_evaluation_dataset/patterns/merged.tra \
		--objective f17_cv \
		--iterations 10 \
		--batch-size 15 \
		--export-iteration-results
	python thesis/utils/evaluate_patterns.py thesis/pattern_evaluation_dataset/evaluation.wlh \
		--truth thesis/pattern_evaluation_dataset/human1.wl \
		--patterns $(FINAL_PATTERNS) results/uk_final.pat

