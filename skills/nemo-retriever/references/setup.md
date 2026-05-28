# Setup turn (when `./lancedb/nv-ingest.lance` doesn't exist)

`retriever ingest ./pdfs/` runs the full pipeline (text extraction + page-element detection + OCR + embedding + LanceDB insert). On corpora >~800 pages this often won't fit a typical setup turn budget (10 min) — the OCR + page-element stages dominate and scale roughly linearly with page count. Always build an index — pick the recipe by corpus size:

```bash
TOTAL_PAGES=$(<RETRIEVER_VENV>/bin/python -c "import pypdfium2, glob; print(sum(len(pypdfium2.PdfDocument(p)) for p in glob.glob('./pdfs/*.pdf')))" 2>/dev/null || echo 0)
echo "total_pages=$TOTAL_PAGES"
if [ "$TOTAL_PAGES" -le 800 ]; then
  <RETRIEVER_VENV>/bin/retriever ingest ./pdfs/ --embed-model-name nvidia/llama-nemotron-embed-1b-v2 --quiet
else
  <RETRIEVER_VENV>/bin/retriever pipeline run ./pdfs/ --run-mode inprocess --method pdfium --no-extract-tables --no-extract-charts --no-extract-page-as-image --evaluation-mode none --embed-model-name nvidia/llama-nemotron-embed-1b-v2 --quiet
fi
```

Always pass `--quiet` on whichever branch fires. It suppresses progress bars, HuggingFace download logs, vLLM init noise, Ray worker stdout, and INFO-level pipeline status lines on success, while still flushing captured output to stderr if ingest errors. Without it the setup turn burns thousands of tokens on irrelevant progress output. On success you only see one line: `Ingested N document(s) into LanceDB lancedb/nv-ingest.` (for `retriever ingest`) or `Pipeline complete: N page(s) → lancedb lancedb/nv-ingest (T.Ts).` (for `retriever pipeline run`).

The `else` branch skips page-element detection, OCR, table extraction, and chart extraction — only pdfium text extraction + embedding. Embedding runs locally via the bundled HuggingFace model by default (no remote NIM needed). It's strictly better to have a text-only index than no index at all: the per-query pdfium text-extract fallback re-extracts a full PDF *per query*, which is both slow and expensive. Page-element detection may emit warning logs when its remote endpoint isn't reachable; the warnings are non-fatal as long as the embedding step itself succeeds (and are silenced by `--quiet` on a successful run).

Don't pre-OCR, don't pre-chunk, don't write Python wrappers — the CLI handles extraction + (optionally) page-element detection + OCR + embedding + LanceDB insert in one shot.

After the setup command returns successfully, STOP. Don't run smoke queries to "warm up" — the first query turn does that naturally.
