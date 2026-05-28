# Pitfalls and recovery

Read this only after you hit one of the named errors below. Don't read it pre-emptively.

## If the index is missing or `retriever query` returns `[]`

Means ingest didn't complete (e.g. the text-only pipeline still hit the turn wall, or the table is empty). Tight fallback using the retriever's own pdfium-based extractor (always available — same binary the agent just used for `retriever query`):

1. `ls ./pdfs/` (one call) to see filenames.
2. Pick the SINGLE PDF whose name best matches the question.
3. ONE call: `<RETRIEVER_VENV>/bin/retriever pdf stage page-elements ./pdfs --method pdfium --json-output-dir /tmp/pdf_text --compact-json`. This emits a JSON sidecar per PDF at `/tmp/pdf_text/<basename>.pdf.pdf_extraction.json` containing per-page text primitives — pdfium only, no OCR, no NIM, fast.
4. `Read` `/tmp/pdf_text/<name>.pdf.pdf_extraction.json` for the chosen PDF and synthesize from the per-page text. If the answer isn't there, still write your best guess based on the filename + extracted pages plus a one-sentence acknowledgement of uncertainty in `final_answer`. Then stop.

Do NOT keep doing text-extract calls across many PDFs to hunt — that exhausts the turn budget. Better to answer partially than to time out. Never re-run `retriever ingest`.

For an unlisted subcommand: `<RETRIEVER_VENV>/bin/retriever <subcommand> --help`.

## Failure modes (expected, not errors)

- **First `ingest` takes ~60s+** — vLLM warmup. Expected.
- **First `query` takes ~10–15s** — embedder cold-start. Expected.
- **Empty result** — ingest didn't run. Use the fallback above.
- **`Clamping num_partitions ...`** — informational on tiny corpora, not an error.
- **Low-relevance top hit on tiny corpus** — look at `_distance` *gaps* between hits, not absolute values.
- **Page-element-detection warnings during ingest** — non-fatal as long as the embedding step itself succeeds (and they're silenced by `--quiet` on a successful run).

## You ran more than 2 Bash calls on a query turn

Budget violation. Stop, write `final_answer` from what you have, write `./output.json`, end the turn. Long turns cost ~5× a disciplined turn and usually still produce the wrong answer.

## Query-turn cost discipline (recap)

- ONE `retriever query` per turn. ONE optional targeted text-extract on the rank-1 PDF if the chunks miss the asked-for fact. That's the budget — it is a hard cap, not a soft preference.
- After your 2nd tool call, write `final_answer` with what you have and STOP. If both calls left the asked-for fact unresolved, write `final_answer` that **explicitly states the retrieved pages don't contain the requested fact** (naming the closest related content if any) — **do not run more tool calls hunting for it, and do not extrapolate a plausible value.**
- Don't read whole PDFs.
- Don't make speculative Read/Glob/Grep calls "to confirm". The retriever already found the relevant pages — trust the ranking.
- Don't spawn agents, write plans, or make todo lists. The workflow is the workflow.
