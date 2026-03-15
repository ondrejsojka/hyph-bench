#!/bin/sh

TRIE_NORMALIZER=15714
TRIE_WEIGHT_BOTTOM=0.0008
TRIE_WEIGHT_TOP=0.001
BAD_COUNT=1
RESULTS="results/weight_setting.csv"

echo "trie_weight,bad_words" > $RESULTS

while [ $BAD_COUNT -ne 500 ]
do
    CENTER=$(perl -e "print (($TRIE_WEIGHT_BOTTOM + $TRIE_WEIGHT_TOP) / 2)")
    
    python -m scripts.optimize --lang uk --objective f17_trie -b --trie-normalizer $TRIE_NORMALIZER --trie-weight $CENTER

    BAD_COUNT=$(wc -l results/uk_bad.txt | sed -e 's/ .*//')

    echo "$CENTER,$BAD_COUNT" >> $RESULTS 

    if [ $BAD_COUNT -lt 500 ] 
    then
        TRIE_WEIGHT_BOTTOM=$CENTER
    else
        TRIE_WEIGHT_TOP=$CENTER
    fi
done

# baseline=42
# r=1/7=0.1428571428571429

# r
#     0.125 -> ~1300
#     0.13 -> 450

# trie_weight
#     0.001 -> 684
#     0.0005 -> 10
#     0.0008 -> 450


