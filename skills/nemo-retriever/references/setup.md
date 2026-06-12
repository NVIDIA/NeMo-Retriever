# Setup Turn

Use this when `./lancedb/nemo-retriever.lance` does not exist yet.

`retriever ingest ./pdfs/` runs the full local ingest workflow: text extraction,
page-element detection, OCR where needed, embedding, and LanceDB insert. On very
large corpora the OCR and page-element stages dominate runtime, so always build
an index but choose the recipe by corpus size.

```bash
TOTAL_PAGES=$(<RETRIEVER_VENV>/bin/python -c "import pypdfium2, glob; print(sum(len(pypdfium2.PdfDocument(p)) for p in glob.glob('./pdfs/*.pdf')))" 2>/dev/null || echo 0)
echo "total_pages=$TOTAL_PAGES"
if [ "$TOTAL_PAGES" -le 50000 ]; then
  <RETRIEVER_VENV>/bin/retriever ingest ./pdfs/ \
    --embed-model-name nvidia/llama-nemotron-embed-1b-v2
else
  <RETRIEVER_VENV>/bin/retriever ingest ./pdfs/ \
    --profile fast-text \
    --embed-model-name nvidia/llama-nemotron-embed-1b-v2
fi
```

Both branches write the same default LanceDB table:
`lancedb/nemo-retriever`. That is the table `retriever query` reads by default.
Keep `--lancedb-uri` and `--table-name` aligned if you override either one.

`retriever ingest` is quiet by default. Quiet mode suppresses progress bars,
HuggingFace download logs, vLLM init noise, Ray worker stdout, and INFO-level
pipeline status lines on success, while still flushing captured output to stderr
on error. On success you should see one summary line similar to:

```text
Ingested N file(s) -> M row(s) in LanceDB lancedb/nemo-retriever.
```

The `fast-text` branch skips expensive PDF recall stages and focuses on pdfium
text extraction plus embedding. It is strictly better to have a text-only index
than no index at all: the per-query pdfium text-extract fallback re-extracts a
full PDF per query, which is both slow and expensive.

Use `retriever ingest batch ./pdfs/ --profile fast-text` only when Ray/batch
throughput is explicitly desired and the environment can support it.

Do not pre-OCR, do not pre-chunk, and do not write Python wrappers. The CLI
handles extraction, optional page-element detection/OCR, embedding, and LanceDB
insert in one shot.

After the setup command returns successfully, stop. Do not run smoke queries to
warm up the index; the first query turn does that naturally.

## Other Input Shapes

Use the same `retriever ingest` command. Root ingest auto-detects supported file
families from extensions; do not pass `--input-type`.

Install extras for non-PDF media live in `references/install.md` under
"Optional extras".

**Images / scanned forms / charts** (`.jpg` `.png` `.tiff` `.bmp` `.svg`):

```bash
<RETRIEVER_VENV>/bin/retriever ingest ./images/ \
  --ocr-version v2 \
  --ocr-lang english
```

For mixed-script docs such as bilingual contracts or multilingual forms, use
`--ocr-lang multi`. Chart understanding runs inline; no separate call is needed.

**HTML / TXT** - ingest even though `Read` could work; chunking and citation
metadata matter:

```bash
<RETRIEVER_VENV>/bin/retriever ingest ./docs/
```

**Office** (`.docx` `.pptx`) - requires LibreOffice on the host:

```bash
<RETRIEVER_VENV>/bin/retriever ingest ./office/
```

**Audio / video** - requires the `[multimedia]` extra and ffmpeg on the host:

```bash
<RETRIEVER_VENV>/bin/retriever ingest ./media/
```

Audio extensions are `.mp3`, `.wav`, and `.m4a`. Video extensions are `.mp4`,
`.mov`, and `.mkv`. Inventory first if the directory might contain unsupported
media such as `.flac`.
