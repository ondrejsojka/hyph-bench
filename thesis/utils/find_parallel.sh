#!/bin/bash

WORDLIST=$1
TRANSLATE=$2
TRIE_NORMALIZER=$(wc -l $WORDLIST | sed -e 's/ .*//')
RESULTS="/var/tmp/xhulka/results"
TOP=0.5
BOTTOM=0.1
WEIGHT=0
BAD_COUNT=1

mkdir "/var/tmp/xhulka"
echo "weight,bad_count" > $RESULTS

while [ $BAD_COUNT -ne 500 ]
do
    WEIGHT=$(perl -e "print (($BOTTOM + $TOP) / 2)")
    OUTPUT_DIR="$RESULTS$WEIGHT"

    mkdir $OUTPUT_DIR
    echo "Starting iteration with weight $WEIGHT"
    nice -n +19 python -m scripts.optimize --lang uk --output-dir $OUTPUT_DIR --iterations 4 --batch-size 25 --wordlist $WORDLIST --translate $TRANSLATE --export-iteration-results --objective f17_trie --trie-normalizer $TRIE_NORMALIZER --trie-weight $WEIGHT

    BAD_COUNT=$(wc -l $OUTPUT_DIR/uk_bad.txt | sed -e 's/ .*//')
    
    if [ $BAD_COUNT -lt 500 ] 
    then
        BOTTOM=$WEIGHT
    else
        TOP=$WEIGHT
    fi

    echo "$WEIGHT,$BAD_COUNT" >> $RESULTS
done

echo "best weight: $WEIGHT, bad count: $BAD_COUNT"
