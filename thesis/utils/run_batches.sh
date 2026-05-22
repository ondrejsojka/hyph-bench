#!/usr/bin/env bash
set -euo pipefail

# Launcher: runs multiple full_batch.sh instances in parallel from a JSON config.
# Usage: run_batches.sh <config.json> [--collect-results]
#
# JSON format: array of objects. Each object maps to full_batch.sh flags.
# Required keys per entry: name, weight, annotation,
#   suk_objective, suk_args, fuk_objective, fuk_args
# Optional: iterations, batch_size

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

usage() {
    cat <<EOF
Usage: run_batches.sh <config.json> [--collect-results]

  <config.json>       JSON file with array of batch configs
  --collect-results   After all runs finish, call collect_results.py
  -h, --help          Show this help

Example config.json:
[
  {
    "name": "run_f17_50",
    "weight": 3,
    "annotation": "claude_new_prompt",
    "suk_objective": "f17_trie",
    "suk_args": "--trie-weight 1.0",
    "fuk_objective": "f17_target",
    "fuk_args": "--bad-target 500 --bad-tolerance 50",
    "iterations": 50,
    "batch_size": 4
  }
]
EOF
}

CONFIG=""
COLLECT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --collect-results) COLLECT=1; shift ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "Unknown flag: $1" >&2; usage; exit 1 ;;
        *) CONFIG="$1"; shift ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Error: config.json required" >&2
    usage; exit 1
fi

if ! command -v jq &>/dev/null; then
    echo "Error: jq is required (sudo dnf install jq)" >&2
    exit 1
fi

N=$(jq 'length' "$CONFIG")
echo "Launching $N batch(es) in parallel..."

PIDS=()
LOG_DIR=/var/tmp/xhulka/launcher_logs
mkdir -p "$LOG_DIR"

for i in $(seq 0 $((N - 1))); do
    entry=$(jq -c ".[$i]" "$CONFIG")

    name=$(echo "$entry"       | jq -r '.name')
    weight=$(echo "$entry"     | jq -r '.weight')
    annotation=$(echo "$entry" | jq -r '.annotation')
    suk_obj=$(echo "$entry"    | jq -r '.suk_objective')
    suk_args=$(echo "$entry"   | jq -r '.suk_args')
    fuk_obj=$(echo "$entry"    | jq -r '.fuk_objective')
    fuk_args=$(echo "$entry"   | jq -r '.fuk_args')
    iters=$(echo "$entry"      | jq -r '.iterations // 50')
    batch=$(echo "$entry"      | jq -r '.batch_size // 1')

    launcher_log="$LOG_DIR/${name}.log"
    echo "  [$((i+1))/$N] $name -> $launcher_log"

    (
        cd "$REPO_ROOT"
        nice -n +19 bash "$SCRIPT_DIR/full_batch.sh" \
            --name "$name" \
            --weight "$weight" \
            --annotation "$annotation" \
            --suk-objective "$suk_obj" \
            --suk-args "$suk_args" \
            --fuk-objective "$fuk_obj" \
            --fuk-args "$fuk_args" \
            --iterations "$iters" \
            --batch-size "$batch"
    ) >"$launcher_log" 2>&1 &

    PIDS+=($!)
done

echo "Waiting for all batches to finish..."
FAILED=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    name=$(jq -r ".[$i].name" "$CONFIG")
    if wait "$pid"; then
        echo "  [OK] $name (pid $pid)"
    else
        echo "  [FAILED] $name (pid $pid) — see $LOG_DIR/${name}.log"
        FAILED=$((FAILED + 1))
    fi
done

if [[ $FAILED -gt 0 ]]; then
    echo "$FAILED batch(es) failed." >&2
fi

if [[ $COLLECT -eq 1 ]]; then
    echo "Collecting results..."
    python "$SCRIPT_DIR/collect_results.py" "$CONFIG" --output-dir "$REPO_ROOT/results"
fi

exit $FAILED
