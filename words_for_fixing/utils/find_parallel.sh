#!/bin/bash

WORDLIST=$1
TRANSLATE=$2
TRIE_NORMALIZER=$(wc -l $WORDLIST | sed -e 's/ .*//')
RESULTS="/var/tmp/results"
BEST_COUNT=0
BEST_DIFF=500
BEST_WEIGHT=0

echo "weight,bad_count" > $RESULTS

for ITER in {1..10}
do
    WEIGHT=$(perl -e "print ($ITER / 100_000)")
    OUTPUT_DIR="/var/tmp/results_$WEIGHT"

    mkdir $OUTPUT_DIR
    echo "Starting iteration with weight $WEIGHT"
    nice -n +19 python -m scripts.optimize --lang uk --output-dir $OUTPUT_DIR --iterations 4 --batch-size 25 --wordlist $WORDLIST --translate $TRANSLATE --export-iteration-results --objective f17_trie --trie-normalizer $TRIE_NORMALIZER --trie-weight $WEIGHT

    BAD_COUNT=$(wc -l $OUTPUT_DIR/uk_bad.txt | sed -e 's/ .*//')
    DIFF=$(perl -e "print(abs($BAD_COUNT - 500))")
    
    if [[ DIFF -le BEST_DIFF ]]
    then
        BEST_COUNT=$BAD_COUNT
        BEST_DIFF=$DIFF
        BEST_WEIGHT=$WEIGHT
    fi

    echo "$WEIGHT,$BAD_COUNT" >> $RESULTS
done

echo "weight: $BEST_WEIGHT, count: $BEST_COUNT"