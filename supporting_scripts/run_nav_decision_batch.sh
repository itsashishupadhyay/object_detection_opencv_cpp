#!/usr/bin/env bash
# Batch-run the nav-decision C++ binary over all test-bracket images.
# Usage: run_nav_decision_batch.sh
set -u

WS=/Users/upadhyay/dev/ICES/object_detection_opencv_cpp
cd "$WS"

BIN="./build/opencv_cpp_release"
MODEL="weight/cassini_issna_planets.onnx"
NAMES="weight/cassini_issna_planets.names"
TEST_SPLIT="artifacts/cassini_issna/test_split.csv"
MANIFEST="artifacts/cassini_issna/image_manifest.csv"
OUT_DIR="artifacts/cassini_issna/decisions"
LOG="artifacts/cassini_issna/nav_decision_batch.log"
STDERR_LOG="artifacts/cassini_issna/nav_decision_batch.stderr.log"

mkdir -p "$OUT_DIR"
: > "$LOG"
: > "$STDERR_LOG"

# Build image_id -> local_path map from manifest
declare -A LOCAL_PATH
while IFS=, read -r image_id mission instrument body filter exposure timestamp_utc target_body source_url sha256 local_path rest; do
    [[ "$image_id" == "image_id" ]] && continue
    LOCAL_PATH["$image_id"]="$local_path"
done < "$MANIFEST"

total=0
processed=0
nominal=0
degraded=0
refused=0
missing=0
errors=0

while IFS=, read -r image_id body timestamp_utc split; do
    [[ "$image_id" == "image_id" ]] && continue
    total=$((total+1))
    path="${LOCAL_PATH[$image_id]:-}"
    if [[ -z "$path" ]]; then
        echo "MISSING_MANIFEST $image_id" >> "$LOG"
        missing=$((missing+1))
        continue
    fi
    # Convert absolute path to relative from workspace root if needed
    rel_path="${path#$WS/}"
    if [[ ! -f "$rel_path" ]]; then
        if [[ -f "$path" ]]; then
            rel_path="$path"
        else
            echo "MISSING_FILE $image_id $path" >> "$LOG"
            missing=$((missing+1))
            continue
        fi
    fi
    # Run binary; capture final status line only
    out=$("$BIN" --nav-decision --mission cassini --instrument issna \
        -p "$rel_path" -m "$MODEL" -l "$NAMES" 2>>"$STDERR_LOG") || {
        echo "BIN_ERROR $image_id exit=$?" >> "$LOG"
        errors=$((errors+1))
        continue
    }
    # Summary line format: "NOMINAL: N | DEGRADED: M | REFUSED: K"
    summary=$(echo "$out" | tail -5 | grep -oE 'NOMINAL: [0-9]+ \| DEGRADED: [0-9]+ \| REFUSED: [0-9]+' | tail -1)
    if [[ -z "$summary" ]]; then
        # Also try to find the per-image status from JSON
        status=$(echo "$out" | grep -oE '"status": "(NOMINAL|DEGRADED|REFUSED)"' | head -1 | sed 's/.*"\(NOMINAL\|DEGRADED\|REFUSED\)".*/\1/')
    else
        # Parse counts from summary (should be 1/0/0, 0/1/0, or 0/0/1 for single image)
        n=$(echo "$summary" | awk -F'[:|]' '{gsub(/ /,""); print $2}')
        d=$(echo "$summary" | awk -F'[:|]' '{gsub(/ /,""); print $4}')
        r=$(echo "$summary" | awk -F'[:|]' '{gsub(/ /,""); print $6}')
        if [[ "$n" == "1" ]]; then status="NOMINAL"
        elif [[ "$d" == "1" ]]; then status="DEGRADED"
        elif [[ "$r" == "1" ]]; then status="REFUSED"
        else status="UNKNOWN"
        fi
    fi
    case "$status" in
        NOMINAL)  nominal=$((nominal+1)) ;;
        DEGRADED) degraded=$((degraded+1)) ;;
        REFUSED)  refused=$((refused+1)) ;;
        *)        echo "UNKNOWN_STATUS $image_id" >> "$LOG" ; errors=$((errors+1)) ;;
    esac
    processed=$((processed+1))
    if (( processed % 10 == 0 )); then
        echo "PROGRESS $processed/$total  N=$nominal D=$degraded R=$refused errors=$errors missing=$missing"
    fi
done < "$TEST_SPLIT"

echo "DONE total=$total processed=$processed NOMINAL=$nominal DEGRADED=$degraded REFUSED=$refused missing=$missing errors=$errors"
