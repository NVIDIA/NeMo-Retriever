# Setup Turn

Use this when `./lancedb/nemo-retriever.lance` does not exist yet.

`retriever ingest ./documents/` runs extraction, embedding, and LanceDB insert as
one workflow. Always use root ingest unless the user explicitly asks for a
different mode or profile. File formats are inputs, not commands.

For extraction-only tasks, use the defaults:

```bash
<RETRIEVER_VENV>/bin/retriever ingest ./documents/
```

For search or question-answering tasks that will use this skill's hybrid and
sparse query passes, build the matching hybrid index explicitly:

```bash
<RETRIEVER_VENV>/bin/retriever ingest ./documents/ --index-mode hybrid
```

The command writes the default LanceDB table:
`lancedb/nemo-retriever`. That is the table `retriever query` reads by default.
Keep `--lancedb-uri` and `--table-name` aligned if you override either one.
`--index-mode hybrid` builds a full-text BM25 index alongside vectors so
`retriever query --retrieval-mode hybrid` can fuse exact-term and vector
retrieval. For table-heavy documents, explicitly add
`--extract-tables --table-output-format markdown`; do not treat structured tables
or hybrid indexing as engine defaults.

## Local versus hosted inference

- On a supported GPU host with the `[local]` extra, the default command may run
  embedding locally.
- On a CPU-only host with `NVIDIA_API_KEY` or `NGC_API_KEY`, run the same command.
  Retriever automatically uses NVIDIA's hosted default embedding endpoint.
- Use `--embed-invoke-url` only for a different endpoint supplied by the user or
  deployment. Do not pass NVIDIA's default URL manually.
- Do not install `torch` or `transformers` merely to ingest HTML/TXT on CPU.

`retriever ingest` is quiet by default. Quiet mode suppresses progress bars,
HuggingFace download logs, vLLM init noise, Ray worker stdout, and INFO-level
pipeline status lines on success, while still flushing captured output to stderr
on error. On success you should see one summary line similar to:

```text
Ingested N file(s) -> M row(s) in LanceDB lancedb/nemo-retriever.
```

Do not pre-OCR, pre-chunk, or write Python extraction wrappers. The CLI handles
extraction, embedding, and LanceDB insert in one shot.

After the setup command returns successfully, stop. Do not run smoke queries to
warm up the index; the first query turn does that naturally.

## Other Input Shapes

Use the same `retriever ingest` command. Root ingest auto-detects supported file
families from extensions; do not pass `--input-type`. Add `--index-mode hybrid`
when the target workflow uses `retriever query --retrieval-mode hybrid`.

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
