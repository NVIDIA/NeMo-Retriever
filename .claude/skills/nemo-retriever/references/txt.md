# retriever txt

Plain-text extraction: scan a directory for `*.txt`, tokenizer-split each
file into chunks, and write `<stem>.txt_extraction.json` sidecars in the
same primitives shape as the rest of the pipeline.

If flags below look stale, re-check `retriever txt run --help`.

## When to use this

- You have plain-text corpora (logs, scraped articles, transcripts) and want
  to feed them into embed → VDB downstream stages.
- Quick way to seed a LanceDB table for retrieval experiments without going
  through PDF rendering.

**Use a different command when:**

- Input is HTML → [[html]].
- Input is PDF/audio/etc → [[pdf]], [[audio]], or the unified [[pipeline]]
  with `--input-type txt`.

## Canonical invocations

Default chunking (512 tokens, no overlap):

```bash
retriever txt run --input-dir data/text/
```

Smaller chunks with overlap:

```bash
retriever txt run --input-dir data/text/ --max-tokens 256 --overlap 32
```

## Inputs

- **`--input-dir DIR`** — required, scanned for `*.txt`.

## Outputs

- `<stem>.txt_extraction.json` per file (next to source by default, or in
  `--output-dir` if set).
- Same primitives-like shape as stage5 input: `text`, `path`, `page_number`
  (always 0 for txt), `metadata`.

## Downstream

After this, run (as the `--help` text instructs):

```bash
retriever local stage5 run --input-dir <dir> --pattern "*.txt_extraction.json"
retriever local stage6 run --input-dir <dir>
```

Or pipe straight through [[pipeline]] with `--input-type txt`.

## Key flags

| Flag | Default | Notes |
|---|---|---|
| `--max-tokens` | `512` | Hard cap per chunk. |
| `--overlap` | `0` | Token overlap between consecutive chunks. |
| `--encoding` | `utf-8` | File read encoding. |
| `--limit` | — | Cap number of files processed. |

## Common failure modes

- **Empty output files** — input `.txt` is empty or all-whitespace; the
  tokenizer produced 0 chunks.
- **Mojibake in extracted text** — wrong `--encoding`; try `latin-1` or
  `utf-16` for legacy files.

## Related

- [[html]] — sibling command for HTML inputs.
- [[pipeline]] — wraps txt extraction + embed + VDB in one command.
- [[vector-store]] — upload the resulting embeddings.
