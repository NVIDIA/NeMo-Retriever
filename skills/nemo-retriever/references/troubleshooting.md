# Troubleshooting and recovery

Read this only after you hit one of the named errors below. Don't read it pre-emptively.

## If ingest fails or the index is missing

Stay on `retriever ingest`; do not switch to format-specific, stage, or pipeline
commands.

1. Read the surfaced error. If more context is needed, rerun the same ingest once
   with `--no-quiet`.
2. On a CPU-only host, verify that `NVIDIA_API_KEY` or `NGC_API_KEY` is non-empty
   without printing its value. The default hosted embedding endpoint is automatic.
3. Use `--embed-invoke-url` only when the user supplied a different endpoint.
4. Do not install `torch` or `transformers` merely to process HTML/TXT.

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
`.mkv`. Other extensions such as `.md`, `.flac`, `.rtf`, `.eml`, `.py`, `.jsonl`,
and `.zip` are skipped. Before ingest, inventory:

```bash
find <dir> -type f -name '*.*' | sed 's/.*\.//' | sort -u
```

Ingest all supported files in the same directory command and report which
unsupported extensions were skipped. Convert them only if the user asks.

## You ran more than 2 Bash calls on a query turn

Budget violation. Stop, write `final_answer` from what you have, end the turn. Long turns cost ~5× a disciplined turn and usually still produce the wrong answer.

## Query-turn cost discipline (recap)

- ONE `retriever query` per turn. ONE optional targeted text-extract on the rank-1 PDF if the chunks miss the asked-for fact. That's the budget — it is a hard cap, not a soft preference.
- After your 2nd tool call, write `final_answer` with what you have and STOP. If both calls left the asked-for fact unresolved, write `final_answer` that **explicitly states the retrieved pages don't contain the requested fact** (naming the closest related content if any) — **do not run more tool calls hunting for it, and do not extrapolate a plausible value.**
- Don't read whole PDFs.
- Don't make speculative Read/Glob/Grep calls "to confirm". The retriever already found the relevant pages — trust the ranking.
- Don't spawn agents, write plans, or make todo lists. The workflow is the workflow.
