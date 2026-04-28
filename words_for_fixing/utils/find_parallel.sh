#!/bin/bash

WORDLIST=$1
TRANSLATE=$2
TRIE_NORMALIZER=$(wc -l $WORDLIST | sed -e 's/ .*//')
RESULTS="/var/tmp/results"

echo "weight,bad_count" > $RESULTS

for ITER in {5..100..5}
do
    WEIGHT=$(perl -e "print ($ITER / 100_000)")
    NAME="weight_$ITER"
    OUTPUT_DIR="/var/tmp/results_$NAME"
    
    mkdir $OUTPUT_DIR
    echo "Starting iteration with $NAME"
    nice -n +19 python -m scripts.optimize --lang uk --output-dir $OUTPUT_DIR --iterations 4 --batch-size 25 --wordlist $WORDLIST --translate $TRANSLATE --export-iteration-results --objective f17_trie --trie-normalizer $TRIE_NORMALIZER --trie-weight $WEIGHT

    BAD_COUNT=$(wc -l results/uk_bad.txt | sed -e 's/ .*//')
    echo "$WEIGHT,$BAD_COUNT" >> $RESULTS
done

cat $RESULTS
