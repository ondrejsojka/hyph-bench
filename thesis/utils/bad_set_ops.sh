#!/usr/bin/env bash
# Computes union and intersection of word lists (one word per line).
# Usage: bad_set_ops.sh <file1> [file2 ...]
# Output: union to stdout with header, intersection below.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <file1> <file2> [file3 ...]" >&2
    exit 1
fi

echo "=== UNION ==="
sort -u "$@"

echo ""
echo "=== INTERSECTION ==="
# Start with sorted contents of the first file, then keep only lines present in each subsequent file
tmp=$(sort -u "$1")
for f in "${@:2}"; do
    tmp=$(comm -12 <(echo "$tmp") <(sort -u "$f"))
done
echo "$tmp"
