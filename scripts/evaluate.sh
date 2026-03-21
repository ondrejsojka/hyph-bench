#!/bin/bash

# evaluate.sh - Evaluate hyph-bench datasets against different patgen binaries and profiles in parallel.
# Measures execution time and memory usage using /usr/bin/time -v.
#
# Usage: Run from the project root INSIDE WSL.
# ./scripts/evaluate.sh [output.csv]

# Configuration
BINARIES=("patgen" "../utf-patgen/build/utfpatgen")
PROFILES_DIR=${PROFILES_DIR:-profiles}
DATA_DIR=${DATA_DIR:-data}
OUTPUT_FILE=${1:-evaluation_results.csv}
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 1)}
LOGS_DIR=${LOGS_DIR:-logs}

# Ensure logs directory exists
mkdir -p "$LOGS_DIR"

# Find all .wlh files in the data directory
WLH_FILES=$(find "$DATA_DIR" -name "*.wlh" | sort)

# Function to run a single evaluation task
evaluate_task() {
    local binary="$1"
    local profile="$2"
    local wlh="$3"
    local output_file="$4"
    
    local profile_name=$(basename "$profile")
    local dataset_name=$(basename "$wlh" .wlh)
    local binary_name=$(basename "$binary")
    local tr="${wlh%.wlh}.tr"
    
    if [ ! -f "$tr" ]; then
        return
    fi

    # Create a temporary directory for this run
    local tmp_dir=$(mktemp -d)
    local input_file="$tmp_dir/input.in"
    
    local task_log="$LOGS_DIR/${dataset_name}_${binary_name}_${profile_name}.log"
    
    echo "[STARTED]  $dataset_name | Binary: $binary | Profile: $profile_name"
    
    # Construct input from profile
    local num_levels=$(grep -v '^#' "$profile" | grep -v '^[[:space:]]*$' | wc -l)
    
    # Range of levels
    echo "1 $num_levels" > "$input_file"
    
    while read -r line; do
        # Skip comments and empty lines
        [[ "$line" =~ ^# ]] && continue
        [[ -z "$line" ]] && continue
        
        # Each line: pat_start pat_finish good_weight bad_weight threshold
        if [[ "$line" =~ ([0-9]+[[:space:]]+){4}[0-9]+ ]]; then
            read -r p_s p_f g b t <<< "$line"
            echo -e "${p_s} ${p_f}\n${g} ${b} ${t}" >> "$input_file"
        fi
    done < "$profile"
    # End input with 'y' to hyphenate word list (to get stats)
    echo "y" >> "$input_file"
    
    # Run measurement
    /usr/bin/time -v "$binary" "$wlh" "/dev/null" "$tmp_dir/final.pat" "$tr" < "$input_file" > "$task_log" 2> "$tmp_dir/time.log"
    
    # Parse metrics
    local user_time=$(grep "User time (seconds):" "$tmp_dir/time.log" | awk '{print $4}')
    local sys_time=$(grep "System time (seconds):" "$tmp_dir/time.log" | awk '{print $4}')
    local max_rss=$(grep "Maximum resident set size (kbytes):" "$tmp_dir/time.log" | awk '{print $6}')
    
    local last_stats=$(grep "good," "$task_log" | tail -n 1)
    local tp=$(echo "$last_stats" | awk '{print $1}')
    local fp=$(echo "$last_stats" | awk '{print $3}')
    local fn=$(echo "$last_stats" | awk '{print $5}')
    
    local num_patterns=$(wc -l < "$tmp_dir/final.pat")
    local num_nodes=$(grep "pattern trie has .* nodes" "$task_log" | tail -n 1 | awk '{print $4}')
    
    # Default to 0 if not found
    tp=${tp:-0}; fp=${fp:-0}; fn=${fn:-0}
    num_patterns=${num_patterns:-0}; num_nodes=${num_nodes:-0}

    echo "[FINISHED] $dataset_name | Binary: $binary | Profile: $profile_name | Time: ${user_time}s | RAM: ${max_rss}KB"
    
    # Output result (append to CSV)
    echo "$binary,$profile_name,$dataset_name,$user_time,$sys_time,$max_rss,$tp,$fp,$fn,$num_patterns,$num_nodes" >> "$output_file"
    
    # Cleanup
    rm -rf "$tmp_dir"
}

# Print CSV header
echo "Binary,Profile,Dataset,UserTime(s),SystemTime(s),MaxRSS(KB),TP,FP,FN,Patterns,TrieNodes" > "$OUTPUT_FILE"
echo "Results will be saved to $OUTPUT_FILE"
echo "Running up to $JOBS parallel jobs..."

# Generate list of tasks and run them in parallel
for binary in "${BINARIES[@]}"; do
    # Check if binary is available
    if ! command -v "$binary" &> /dev/null && [ ! -f "$binary" ]; then
        echo "Binary $binary not found, skipping." >&2
        continue
    fi

    for profile in "$PROFILES_DIR"/*.in; do
        [ -e "$profile" ] || continue
        for wlh in $WLH_FILES; do
            # Check if .tr file exists before spawning background job
            tr="${wlh%.wlh}.tr"
            if [ ! -f "$tr" ]; then
                continue
            fi

            # Run in background
            evaluate_task "$binary" "$profile" "$wlh" "$OUTPUT_FILE" &
            
            # Limit number of parallel jobs
            if [[ $(jobs -r | wc -l) -ge $JOBS ]]; then
                wait -n
            fi
        done
    done
done

# Wait for all remaining jobs to finish
wait

echo "Done. Results written to $OUTPUT_FILE"
