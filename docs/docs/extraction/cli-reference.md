# CLI Reference

After you install the Python dependencies, you can run the graph pipeline example script at `src/nemo_retriever/examples/graph_pipeline.py`.

The recommended invocation is:

```bash
python3 -m nemo_retriever.examples.graph_pipeline <input-path> [OPTIONS]
```

!!! note "What this page covers"
    This page documents the `graph_pipeline.py` example script. It does not use the older `retriever --task ...` interface, so options such as `--dataset`, `--task`, `--output_directory`, and `--pdf_split_page_count` are not described here.

To list the available options in an environment with the project dependencies installed, run:

```bash
python3 -m nemo_retriever.examples.graph_pipeline --help
```


## Parameter Reference

The script requires a positional `input_path`, which can be either a single file or a directory containing files of the selected `--input-type`.
The table below focuses on the flags used most often with `graph_pipeline.py`.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `<input_path>` | path | required | File or directory to ingest. |
| `--run-mode` | string | `batch` | Execution mode: `batch` (Ray Data) or `inprocess` (single-process pandas). |
| `--input-type` | string | `pdf` | Input type: `pdf`, `doc`, `txt`, `html`, `image`, or `audio`. |
| `--method` | string | `pdfium` | Extraction method used for PDF and document ingestion. |
| `--dpi` | int | `300` | Render DPI for PDF page images. |
| `--extract-text / --no-extract-text` | bool | `True` | Enable or disable text extraction. |
| `--extract-tables / --no-extract-tables` | bool | `True` | Enable or disable table extraction. |
| `--extract-charts / --no-extract-charts` | bool | `True` | Enable or disable chart extraction. |
| `--extract-infographics / --no-extract-infographics` | bool | `False` | Enable or disable infographic extraction. |
| `--extract-page-as-image / --no-extract-page-as-image` | bool | `True` | Preserve rendered page images during extraction. |
| `--use-graphic-elements` | flag | `False` | Enable graphic-element processing when the configured pipeline supports it. |
| `--use-table-structure` | flag | `False` | Enable table-structure processing when the configured pipeline supports it. |
| `--table-output-format` | string | none | Optional output format for table-structure extraction. |
| `--api-key` | string | none | API key for remote NIM endpoints. If omitted, the script also checks `NVIDIA_API_KEY` and `NGC_API_KEY`. |
| `--page-elements-invoke-url` | URL | none | Remote page-elements endpoint. |
| `--ocr-invoke-url` | URL | none | Remote OCR endpoint. |
| `--graphic-elements-invoke-url` | URL | none | Remote graphic-elements endpoint. |
| `--table-structure-invoke-url` | URL | none | Remote table-structure endpoint. |
| `--embed-invoke-url` | URL | none | Remote embedding endpoint. |
| `--embed-model-name` | string | `nvidia/llama-nemotron-embed-1b-v2` | Embedding model name. |
| `--embed-modality` | string | `text` | Embedding modality, for example `text` or `text_image`. |
| `--embed-granularity` | string | `element` | Embedding granularity. |
| `--text-elements-modality` | string | none | Override modality for text elements. |
| `--structured-elements-modality` | string | none | Override modality for structured elements. |
| `--text-chunk` | flag | `False` | Add a text-splitting stage after extraction. |
| `--text-chunk-max-tokens` | int | none | Maximum tokens per chunk. |
| `--text-chunk-overlap-tokens` | int | none | Token overlap between adjacent chunks. |
| `--dedup / --no-dedup` | bool | auto | Explicitly enable or disable deduplication. If captioning is enabled, dedup is enabled automatically unless `--no-dedup` is passed. |
| `--dedup-iou-threshold` | float | `0.45` | IOU threshold used by deduplication. |
| `--caption / --no-caption` | bool | `False` | Add an image-captioning stage. |
| `--caption-invoke-url` | URL | none | Remote caption endpoint. |
| `--caption-model-name` | string | `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` | Caption model name. |
| `--caption-device` | string | none | Device override for local captioning. |
| `--caption-context-text-max-chars` | int | `0` | Max surrounding text characters supplied to the captioner. |
| `--caption-gpu-memory-utilization` | float | `0.5` | Fraction of GPU memory reserved for captioning. |
| `--store-images-uri` | URI | none | Persist extracted images to external storage. |
| `--store-text / --no-store-text` | bool | `False` | Store extracted text alongside images when `--store-images-uri` is set. |
| `--strip-base64 / --no-strip-base64` | bool | `True` | Remove inline base64 payloads after storing artifacts. |
| `--ray-address` | string | none | Ray cluster address for batch execution. |
| `--ray-log-to-driver / --no-ray-log-to-driver` | bool | `True` | Control Ray worker log forwarding in batch mode. |
| `--lancedb-uri` | path | `lancedb` | LanceDB directory used for the output table. |
| `--hybrid / --no-hybrid` | bool | `False` | Enable hybrid retrieval for evaluation. |
| `--query-csv` | path | `./data/bo767_query_gt.csv` | Query CSV used for recall evaluation. If the file does not exist, evaluation is skipped. |
| `--evaluation-mode` | string | `recall` | Evaluation mode: `recall` or `beir`. |
| `--recall-match-mode` | string | `pdf_page` | Recall matching mode: `pdf_page`, `pdf_only`, or `audio_segment`. |
| `--audio-match-tolerance-secs` | float | `2.0` | Audio matching tolerance used with `audio_segment` recall. |
| `--segment-audio / --no-segment-audio` | bool | `False` | Enable audio segmentation before transcription. |
| `--audio-split-type` | string | `size` | Audio split mode: `size`, `time`, or `frame`. |
| `--audio-split-interval` | int | `500000` | Split interval used by the selected audio split mode. |
| `--runtime-metrics-dir` | path | none | Directory for runtime summary JSON output. |
| `--runtime-metrics-prefix` | string | none | Prefix for the runtime summary file name. |
| `--detection-summary-file` | path | none | Write a detection summary JSON file after ingestion. |
| `--log-file` | path | none | Tee stdout, stderr, and logs into a single file. |


## Output and Artifacts

The graph pipeline script does not write per-document JSON files to an `output_directory`.

Instead, the main outputs are:

- **LanceDB table**: The script writes the ingested rows to the `nv-ingest` table under `--lancedb-uri`. Each run overwrites that table.
- **Runtime summary JSON**: If `--runtime-metrics-dir` or `--runtime-metrics-prefix` is provided, the script writes `<prefix>.runtime.summary.json`.
- **Detection summary JSON**: If `--detection-summary-file` is set, the script writes a detection summary for the processed results.
- **Stored media artifacts**: If `--store-images-uri` is set, extracted images are written to the configured storage URI. `--store-text` and `--strip-base64/--no-strip-base64` control the stored payload.


## Errors and Exit Codes

The script does not define a custom exit-code table, but the following behaviors are enforced directly by `graph_pipeline.py`:

| Condition | Behavior | Exit |
|-----------|----------|------|
| Input path does not exist | Raises `BadParameter` for the positional input path. | Non-zero |
| No matching files for `--input-type` | Raises `BadParameter` after scanning the input directory. | Non-zero |
| Unsupported `--input-type` | Raises `BadParameter`. | Non-zero |
| Unsupported `--run-mode` | Raises `ValueError`. | Non-zero |
| Unsupported `--recall-match-mode` | Raises `ValueError`. | Non-zero |
| Unsupported `--audio-split-type` | Raises `ValueError`. | Non-zero |
| Unsupported `--evaluation-mode` | Raises `ValueError`. | Non-zero |
| `--evaluation-mode beir` without BEIR inputs | Raises `ValueError` unless both `--beir-loader` and `--beir-dataset-name` are provided. | Non-zero |
| Missing `--query-csv` for recall mode | Logs a warning and skips recall evaluation. | Zero |


## Examples

Run the following commands from the source tree after activating the environment that has the NeMo Retriever dependencies installed.


### Example: PDF ingestion in batch mode

This example ingests a PDF, uses a remote embedding endpoint, and writes the resulting vectors to LanceDB.

```bash
python3 -m nemo_retriever.examples.graph_pipeline ./data/test.pdf \
  --run-mode batch \
  --input-type pdf \
  --method pdfium \
  --embed-invoke-url http://localhost:8000/v1 \
  --lancedb-uri ./artifacts/lancedb
```


### Example: In-process PDF run for local testing

Use `inprocess` mode when you want a simpler local run without Ray.

```bash
python3 -m nemo_retriever.examples.graph_pipeline ./data/test.pdf \
  --run-mode inprocess \
  --input-type pdf \
  --method pdfium \
  --ocr-invoke-url http://localhost:9000/v1 \
  --embed-invoke-url http://localhost:8000/v1 \
  --lancedb-uri ./artifacts/lancedb
```


### Example: PDF extraction with chunking

The graph pipeline does not expose the old `split` task or `--pdf_split_page_count`. To add a post-extraction text-splitting stage, use the text chunking flags instead.

```bash
python3 -m nemo_retriever.examples.graph_pipeline ./data/test.pdf \
  --input-type pdf \
  --method pdfium \
  --text-chunk \
  --text-chunk-max-tokens 512 \
  --text-chunk-overlap-tokens 100 \
  --embed-invoke-url http://localhost:8000/v1 \
  --lancedb-uri ./artifacts/lancedb
```


### Example: Caption extracted page imagery

This script supports captioning, but it does not expose the old caption `reasoning` option. Use the caption flags that `graph_pipeline.py` actually provides.

```bash
python3 -m nemo_retriever.examples.graph_pipeline ./data/test.pdf \
  --input-type pdf \
  --extract-page-as-image \
  --caption \
  --caption-invoke-url http://localhost:8010/v1 \
  --caption-context-text-max-chars 500 \
  --embed-invoke-url http://localhost:8000/v1 \
  --lancedb-uri ./artifacts/lancedb
```


### Example: Store extracted images to object storage

Use `--store-images-uri` to persist extracted image artifacts outside the LanceDB table.

```bash
python3 -m nemo_retriever.examples.graph_pipeline ./data/test.pdf \
  --input-type pdf \
  --store-images-uri s3://my-bucket/nemo-retriever/images \
  --store-text \
  --no-strip-base64 \
  --embed-invoke-url http://localhost:8000/v1 \
  --lancedb-uri ./artifacts/lancedb
```


### Example: Process plain-text files

For text inputs, switch the input type to `txt` and point the script at either a single file or a directory of `.txt` files.

```bash
python3 -m nemo_retriever.examples.graph_pipeline ./data/text_docs \
  --input-type txt \
  --embed-invoke-url http://localhost:8000/v1 \
  --lancedb-uri ./artifacts/lancedb
```


### Example: Process HTML files

```bash
python3 -m nemo_retriever.examples.graph_pipeline ./data/html_docs \
  --input-type html \
  --embed-invoke-url http://localhost:8000/v1 \
  --lancedb-uri ./artifacts/lancedb
```


### Example: Process image files with multimodal embeddings

```bash
python3 -m nemo_retriever.examples.graph_pipeline ./data/images \
  --input-type image \
  --embed-modality text_image \
  --embed-granularity page \
  --caption \
  --caption-invoke-url http://localhost:8010/v1 \
  --embed-invoke-url http://localhost:8000/v1 \
  --lancedb-uri ./artifacts/lancedb
```


### Example: Process audio files

Use the audio-specific flags when ingesting `.mp3`, `.wav`, or `.m4a` inputs.

```bash
python3 -m nemo_retriever.examples.graph_pipeline ./data/audio \
  --input-type audio \
  --segment-audio \
  --audio-split-type time \
  --audio-split-interval 45 \
  --recall-match-mode audio_segment \
  --embed-invoke-url http://localhost:8000/v1 \
  --lancedb-uri ./artifacts/lancedb
```


### Example: Write runtime and detection summaries

```bash
python3 -m nemo_retriever.examples.graph_pipeline ./data/test.pdf \
  --input-type pdf \
  --embed-invoke-url http://localhost:8000/v1 \
  --lancedb-uri ./artifacts/lancedb \
  --runtime-metrics-dir ./artifacts/runtime \
  --runtime-metrics-prefix sample-run \
  --detection-summary-file ./artifacts/detection_summary.json \
  --log-file ./artifacts/graph_pipeline.log
```
