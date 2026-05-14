# retriever html

HTML extraction: `markitdown` converts HTML → Markdown, then tokenizer-split
into chunks. Writes `<stem>.html_extraction.json` sidecars in the standard
primitives shape.

If flags below look stale, re-check `retriever html run --help`.

## When to use this

- You scraped a set of HTML pages and want them in the retriever pipeline.
- You want the same downstream contract as [[txt]] but for HTML inputs.

**Use a different command when:**

- Input is plain text → [[txt]].
- You want to run full ingest end-to-end on HTML → [[pipeline]] with
  `--input-type html`.

## Canonical invocations

Default chunking:

```bash
retriever html run --input-dir data/html/
```

Smaller chunks with overlap:

```bash
retriever html run --input-dir data/html/ --max-tokens 256 --overlap 32
```

## Inputs

- **`--input-dir DIR`** — required, scanned for `*.html`.

## Outputs

- `<stem>.html_extraction.json` per file (next to source by default, or in
  `--output-dir`).
- Same primitives-like shape as stage5 input.

## Downstream

```bash
retriever local stage5 run --input-dir <dir> --pattern "*.html_extraction.json"
retriever local stage6 run --input-dir <dir>
```

Or [[pipeline]] with `--input-type html`.

## Key flags

| Flag | Default | Notes |
|---|---|---|
| `--max-tokens` | `512` | Per-chunk cap. |
| `--overlap` | `0` | Tokens of overlap. |
| `--encoding` | `utf-8` | HTML file encoding. |
| `--limit` | — | Cap number of files processed. |

## Common failure modes

- **Heavy boilerplate in chunks (nav menus, footers)** — `markitdown` is
  intentionally low-magic. Strip nav/footer in a pre-step if it pollutes
  retrieval.
- **JS-rendered pages produce near-empty output** — `markitdown` doesn't run
  JS. Pre-render with a headless browser before feeding here.

## Related

- [[txt]] — sibling for plain-text inputs.
- [[pipeline]] — full extract → embed → VDB for HTML.
