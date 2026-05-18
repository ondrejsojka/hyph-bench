SUK_TRIE_WEIGHT=0.00085
SUK_TRIE_NORMALIZER=15714

RESULTS_DIR="thesis/results"

ANNOTATIONS=$(wildcard thesis/annotation_results/*)
PATTENRS=$(wildcard thesis/pattern_evaluation_dataset/patterns/*)

FINAL_PATTERNS=/var/tmp/xhulka/f17_cv_weight_2_larger_threshold/fuk/uk_final.pat

# Reproduce all results
thesis: suk531 kappa_table indistinct optimization_results evaluate_patterns

# Creates an table comparing the final patterns against Polyakov 
evaluate_patterns: $(FINAL_PATTERNS)
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

# Create the final optimized patterns
$(FINAL_PATTERNS):
	./thesis/utils/run_batches.sh thesis/batch_configs/only_best.json

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

