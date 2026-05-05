# Multi-format Smoke Test with the `retriever` CLI

This page is the `retriever`-CLI counterpart to
`nv-ingest/api/api_tests/smoke_test.sh`.

The original script loops over formats (pdf, jpeg, png, tiff, bmp, wav, pptx,
docx) and submits each file to the running ingest service via
`nv-ingest-cli`, counting PASS/FAIL and printing a table.

With `retriever`, each format is driven by a single `--input-type` dispatch.
The same `exit_code` logic works — `retriever` returns non-zero on failure.

## Replacement script

```bash
#!/bin/bash
# smoke_test_retriever.sh — multi-format parity smoke test for `retriever`.

set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$( dirname "$SCRIPT_DIR" )"

OUTPUT_DIR="./processed/smoke_test_retriever"
DATA_DIR="${BASE_DIR}/../data"
mkdir -p "$OUTPUT_DIR"

LOG_DIR="./extraction_logs_retriever"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/extraction_test_$(date +%Y%m%d_%H%M%S).log"
TEMP_LOG_DIR="$LOG_DIR/temp_logs"
mkdir -p "$TEMP_LOG_DIR"

echo "====== Multimodal Extraction Test (retriever) - $(date) ======" | tee -a "$LOG_FILE"

run_extraction() {
    local file_ext=$1
    local input_type=$2
    local test_file="$DATA_DIR/multimodal_test.$file_ext"
    local temp_log="$TEMP_LOG_DIR/${file_ext}_extraction.log"
    local result_file="$TEMP_LOG_DIR/${file_ext}_result"

    {
        echo "====== Testing $file_ext (input-type=$input_type) ======"
        echo "File: $test_file"
        echo "Started at: $(date)"

        if [ ! -f "$test_file" ]; then
            echo "ERROR: File $test_file not found. Skipping test."
            echo "1" > "$result_file"
            return 1
        fi

        start_time=$(date +%s)

        retriever pipeline run "$test_file" \
            --input-type "$input_type" \
            --run-mode inprocess \
            --extract-text --extract-tables --extract-charts --extract-infographics \
            --save-intermediate "$OUTPUT_DIR/${file_ext}" \
            --lancedb-uri "$OUTPUT_DIR/lancedb_${file_ext}" \
            2>&1

        exit_code=$?
        end_time=$(date +%s)
        duration=$((end_time - start_time))

        echo "Exit code: $exit_code"
        echo "Duration: $duration seconds"
        echo "--------------------------------------------------"

        echo "$exit_code" > "$result_file"
    } > "$temp_log" 2>&1

    exit_code=$(cat "$result_file")
    return $exit_code
}

declare -A formats=(
    ["pdf"]="pdf"
    ["jpeg"]="image"
    ["png"]="image"
    ["tiff"]="image"
    ["bmp"]="image"
    ["wav"]="audio"
    ["pptx"]="doc"
    ["docx"]="doc"
)

total_formats=0
successful_formats=0
declare -A test_results=()
declare -A test_durations=()

for ext in "${!formats[@]}"; do
    input_type="${formats[$ext]}"
    ((total_formats++))

    run_extraction "$ext" "$input_type" &
    pid=$!
    pids+=($pid)
    format_pids["$pid"]="$ext"
done

for pid in "${pids[@]}"; do
    wait $pid
    exit_code=$?
    format="${format_pids[$pid]}"

    temp_log="$TEMP_LOG_DIR/${format}_extraction.log"
    duration="N/A"
    if [ -f "$temp_log" ]; then
        duration=$(grep "Duration:" "$temp_log" | awk '{print $2}')
        test_durations["$format"]="$duration"
    fi

    if [ $exit_code -eq 0 ]; then
        ((successful_formats++))
        test_results["$format"]="PASS"
    else
        test_results["$format"]="FAIL"
        echo "====== FAILED: $format extraction ======" | tee -a "$LOG_FILE"
        cat "$temp_log" | tee -a "$LOG_FILE"
    fi
done

echo "====== SUMMARY ======" | tee -a "$LOG_FILE"
echo "Total formats tested: $total_formats" | tee -a "$LOG_FILE"
echo "Successful extractions: $successful_formats" | tee -a "$LOG_FILE"
echo "Failed extractions: $((total_formats - successful_formats))" | tee -a "$LOG_FILE"

if [ $successful_formats -eq $total_formats ]; then
    echo "All tests passed successfully!"
    exit 0
else
    exit 1
fi
```

## Format-to-`--input-type` mapping

| File extension | `--input-type` |
|----------------|----------------|
| `.pdf` | `pdf` |
| `.jpg`, `.jpeg`, `.png`, `.tiff`, `.bmp` | `image` |
| `.mp3`, `.wav`, `.m4a` | `audio` |
| `.docx`, `.pptx` | `doc` |
| `.txt` | `txt` |
| `.html` | `html` |

(See `nemo_retriever/src/nemo_retriever/pipeline/__main__.py::_resolve_file_patterns`.)

## Parity notes

- The old script required a running ingest service at `localhost:7670`. The
  `retriever` run is self-contained — no service needed. This is usually an
  improvement for CI smoke tests.
- PASS / FAIL semantics and the per-format log layout are preserved.
- `--run-mode inprocess` is recommended for smoke tests: it skips Ray startup
  and gives a cleaner failure mode.
- Each format gets its own `--lancedb-uri` so parallel runs do not contend on
  the same LanceDB directory.
