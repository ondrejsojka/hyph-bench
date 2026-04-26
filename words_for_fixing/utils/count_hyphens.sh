#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <folder>"
    exit 1
fi

folder="$1"

if [ ! -d "$folder" ]; then
    echo "Error: '$folder' is not a directory"
    exit 1
fi

for file in "$folder"/*; do
    if [ -f "$file" ]; then
        count=$(tr -cd '-' < "$file" | wc -c)
        echo "$(basename "$file"): $count"
    fi
done