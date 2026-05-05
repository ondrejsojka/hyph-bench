#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: full_batch.sh [OPTIONS]

Pipeline: fix wordlist -> optimize SUK -> hyphenate FUK -> optimize FUK -> cross-validate

Required:
  --name NAME               Output directory name under /var/tmp/xhulka/
  --weight WEIGHT           Weight for fixed words in replace_in_wordlist
  --annotation ANNOTATION   Annotation results directory name

SUK optimize objective (both args required together):
  --suk-objective OBJ       Objective for SUK phase (e.g. f17_trie, f17_target)
  --suk-args "ARGS"         Extra args for SUK objective (quoted string)

FUK optimize objective (both args required together):
  --fuk-objective OBJ       Objective for FUK phase
  --fuk-args "ARGS"         Extra args for FUK objective (quoted string)

Optimize tuning (applied to only to FUK optimization):
  --iterations N            Number of optimize iterations (default: 50)
  --batch-size N            Optimize batch size (default: 1)

Cross-validation:
  --nfold N                 Number of folds for cross-validation (default: 10)

Other:
  -h, --help                Show this help
EOF
}

# Defaults
ITERATIONS=50
BATCH_SIZE=1
NFOLD=10
SUK_OBJECTIVE=""
SUK_ARGS=""
FUK_OBJECTIVE=""
FUK_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)          OUTPUT_NAME="$2"; shift 2 ;;
        --weight)        WEIGHT="$2"; shift 2 ;;
        --annotation)    ANNOTATION="$2"; shift 2 ;;
        --suk-objective) SUK_OBJECTIVE="$2"; shift 2 ;;
        --suk-args)      SUK_ARGS="$2"; shift 2 ;;
        --fuk-objective) FUK_OBJECTIVE="$2"; shift 2 ;;
        --fuk-args)      FUK_ARGS="$2"; shift 2 ;;
        --iterations)    ITERATIONS="$2"; shift 2 ;;
        --batch-size)    BATCH_SIZE="$2"; shift 2 ;;
        --nfold)         NFOLD="$2"; shift 2 ;;
        -h|--help)       usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

for var in OUTPUT_NAME WEIGHT ANNOTATION; do
    if [[ -z "${!var:-}" ]]; then
        echo "Error: --${var,,} is required" >&2
        usage; exit 1
    fi
done
if [[ -z "$SUK_OBJECTIVE" || -z "$FUK_OBJECTIVE" ]]; then
    echo "Error: --suk-objective and --fuk-objective are required" >&2
    usage; exit 1
fi

OUTPUT=/var/tmp/xhulka/$OUTPUT_NAME
SUK_OUTPUT=$OUTPUT/suk
FUK_OUTPUT=$OUTPUT/fuk

TRANSLATE=data/uk/wiktionary/uk-full-wiktionary.wlh.tra
SUK_ORIGINAL=data/uk/wiktionary/uk-full-wiktionary.wlh
FUK_ORIGINAL=data/uk/dict_uk/uk_full_dictuk.wl
SUK=$SUK_OUTPUT/uk-full-wiktionary.wlh
FUK=$FUK_OUTPUT/uk_full_dictuk.wl

mkdir -p "$SUK_OUTPUT" "$FUK_OUTPUT"
cp "$SUK_ORIGINAL" "$SUK"
cp "$FUK_ORIGINAL" "$FUK"

S_SIZE=$(wc -l < "$SUK")
F_SIZE=$(wc -l < "$FUK")

echo "[$(date '+%H:%M:%S')] Starting batch: $OUTPUT_NAME"
echo "[$(date '+%H:%M:%S')] SUK size=$S_SIZE, FUK size=$F_SIZE"

# --- Step 1: fix wordlist ---
echo "[$(date '+%H:%M:%S')] Step 1: replace_in_wordlist"
python -m scripts.replace_in_wordlist \
    --wordlist "$SUK" \
    --fixed "thesis/annotation_results/$ANNOTATION" \
    --weight "$WEIGHT" \
    --output-dir "$OUTPUT"

# --- Step 2: optimize SUK ---
echo "[$(date '+%H:%M:%S')] Step 2: optimize SUK (objective=$SUK_OBJECTIVE)"
python -m scripts.optimize \
    --lang uk \
    --output-dir "$SUK_OUTPUT" \
    --export-iteration-results \
    --wordlist "$OUTPUT/fixed.wlh" \
    --translate "$TRANSLATE" \
    --objective "$SUK_OBJECTIVE" \
    $SUK_ARGS \
    2>&1 | tee "$SUK_OUTPUT/optimize.log"

# --- Step 3: hyphenate FUK with SUK patterns ---
echo "[$(date '+%H:%M:%S')] Step 3: hyphenate FUK"
python thesis/utils/hyphenate.py \
    "$SUK_OUTPUT/uk_final.pat" \
    "$FUK" \
    "$OUTPUT/uk_full_dictuk.wlh"

# --- Step 4: optimize FUK ---
echo "[$(date '+%H:%M:%S')] Step 4: optimize FUK (objective=$FUK_OBJECTIVE)"
python -m scripts.optimize \
    --lang uk \
    --output-dir "$FUK_OUTPUT" \
    --export-iteration-results \
    --wordlist "$OUTPUT/uk_full_dictuk.wlh" \
    --translate "$TRANSLATE" \
    --iterations "$ITERATIONS" \
    --batch-size "$BATCH_SIZE" \
    --objective "$FUK_OBJECTIVE" \
    $FUK_ARGS \
    2>&1 | tee "$FUK_OUTPUT/optimize.log"

# --- Step 5: cross-validate FUK patterns ---
echo "[$(date '+%H:%M:%S')] Step 5: cross-validate FUK ($NFOLD-fold)"

# Parse best params from FUK optimize log (last occurrence of the summary line)
FUK_PARAMS_LINE=$(grep "bad_weights=" "$FUK_OUTPUT/optimize.log" | tail -1)
FUK_WEIGHTS=$(echo "$FUK_PARAMS_LINE" | grep -oP '(?<=bad_weights=\()[\d, ]+(?=\))' | tr -d ' ' | tr ',' ' ')
FUK_THRESHOLD=$(echo "$FUK_PARAMS_LINE" | grep -oP '(?<=threshold=)\d+')

if [[ -z "$FUK_WEIGHTS" || -z "$FUK_THRESHOLD" ]]; then
    echo "[$(date '+%H:%M:%S')] WARNING: could not parse best params from FUK log, skipping cross-validation" >&2
else
    echo "[$(date '+%H:%M:%S')] Best FUK params: bad_weights=($FUK_WEIGHTS) threshold=$FUK_THRESHOLD"
    python -m scripts.cross_validate \
        --lang uk \
        --wordlist "$OUTPUT/uk_full_dictuk.wlh" \
        --translate "$TRANSLATE" \
        --params $FUK_WEIGHTS $FUK_THRESHOLD \
        --nfold "$NFOLD" \
        2>&1 | tee "$FUK_OUTPUT/crossval.log"
fi

echo "[$(date '+%H:%M:%S')] Done: $OUTPUT_NAME"
