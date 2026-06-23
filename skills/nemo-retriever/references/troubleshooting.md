# Troubleshooting and recovery

Read this only after you hit one of the named errors below. Don't read it pre-emptively.

## If the index is missing or `retriever query` returns empty `evidence`

Means ingest didn't complete (e.g. the text-only pipeline still hit the turn wall, or the table is empty). Tight fallback using the retriever's own pdfium-based extractor (always available — same binary the agent just used for `retriever query`):

1. `ls ./pdfs/` (one call) to see filenames.
2. Pick the SINGLE PDF whose name best matches the question.
3. ONE call: `<RETRIEVER_VENV>/bin/retriever pdf stage page-elements ./pdfs --method pdfium --json-output-dir /tmp/pdf_text --compact-json`. This emits a JSON sidecar per PDF at `/tmp/pdf_text/<basename>.pdf.pdf_extraction.json` containing per-page text primitives — pdfium only, no OCR, no NIM, fast.
4. `Read` `/tmp/pdf_text/<name>.pdf.pdf_extraction.json` for the chosen PDF and synthesize from the per-page text. If the answer isn't there, still write your best guess based on the filename + extracted pages plus a one-sentence acknowledgement of uncertainty in `final_answer`. Then stop.

Do NOT keep doing text-extract calls across many PDFs to hunt — that exhausts the turn budget. Better to answer partially than to time out. Never re-run `retriever ingest`.

For an unlisted subcommand: `<RETRIEVER_VENV>/bin/retriever <subcommand> --help`.

## Failure modes (expected, not errors)

- **First `ingest` takes ~60s+** — vLLM warmup. Expected.
- **First `query` is slow** — embedder cold-start. ~10–15s on an idle GPU, but **1–3 minutes under concurrent load**. Expected — wait for it; do not kill or relaunch. It is wrapped in `timeout 2000`, so let it run to that ceiling before treating it as failed.
- **Empty `evidence`** — ingest didn't run (use the fallback above), or the question is genuinely out-of-corpus — read `coverage.thin_spots` to tell which.
- **`Clamping num_partitions ...`** — informational on tiny corpora, not an error.
- **Low-relevance top hit on tiny corpus** — even an unrelated query returns *something*; trust the ranking order (the `score` field is informational, not calibrated confidence).
- **Page-element-detection warnings during ingest** — non-fatal as long as the embedding step itself succeeds (and they're silenced on a successful run, since `ingest` is quiet by default).

## Unsupported file types

`retriever ingest` auto-detects supported input types from file extensions. It
supports `.pdf`, `.docx`, `.pptx`, `.txt`, `.html`, `.jpg`, `.jpeg`, `.png`,
`.tiff`, `.tif`, `.bmp`, `.svg`, `.mp3`, `.wav`, `.m4a`, `.mp4`, `.mov`, and
`.mkv`. Treat other extensions such as `.flac`, `.rtf`, `.eml`, `.py`, `.jsonl`,
and `.zip` as setup issues. Before ingest, inventory:

```bash
find <dir> -type f -name '*.*' | sed 's/.*\.//' | sort -u
```

If unsupported extensions appear, name them in your reply and ask the user
whether to skip or convert them.

## You ran more than 2 Bash calls on a query turn

Budget violation. Stop, write `final_answer` from what you have, end the turn. Long turns cost ~5× a disciplined turn and usually still produce the wrong answer.

## Query-turn cost discipline (recap)

- ONE `retriever query` per turn. ONE optional targeted text-extract on the rank-1 PDF if the chunks miss the asked-for fact. That's the budget — it is a hard cap, not a soft preference.
- After your 2nd tool call, write `final_answer` with what you have and STOP. If both calls left the asked-for fact unresolved, write `final_answer` that **explicitly states the retrieved pages don't contain the requested fact** (naming the closest related content if any) — **do not run more tool calls hunting for it, and do not extrapolate a plausible value.**
- Don't read whole PDFs.
- Don't make speculative Read/Glob/Grep calls "to confirm". The retriever already found the relevant pages — trust the ranking.
- Don't spawn agents, write plans, or make todo lists. The workflow is the workflow.
