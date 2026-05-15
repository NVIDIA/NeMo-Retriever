---
name: nemo-retriever
description: Use when the user wants to search, index, or answer questions over a folder of PDFs (or other documents) — including building a RAG / search index over PDFs, looking up information across many PDFs, or running the `retriever` CLI (ingest, query, pipeline, recall, eval, etc.).
---

# nemo-retriever

The `retriever` CLI indexes a folder of PDFs into LanceDB (`retriever ingest`) and serves vector search over it (`retriever query`). For any task about searching/answering questions across a folder of PDFs, use this CLI — do not write a custom RAG.

## Setup turn (when `./lancedb/nv-ingest.lance` doesn't exist)

`retriever ingest ./pdfs/` runs the full pipeline (text extraction + page-element detection + OCR + embedding + LanceDB insert). On corpora >~800 pages this often won't fit a typical setup turn budget (10 min) — the OCR + page-element stages dominate and scale roughly linearly with page count. Always build an index — pick the recipe by corpus size:

```bash
TOTAL_PAGES=$(for p in ./pdfs/*.pdf; do pdfinfo "$p" 2>/dev/null | awk '/^Pages:/{print $2}'; done | awk '{s+=$1} END{print s+0}')
echo "total_pages=$TOTAL_PAGES"
if [ "$TOTAL_PAGES" -le 800 ]; then
  retriever ingest ./pdfs/
else
  retriever pipeline run ./pdfs/ --run-mode inprocess --method pdfium --no-extract-tables --no-extract-charts --no-extract-page-as-image --evaluation-mode none
fi
```

The `else` branch skips page-element detection, OCR, table extraction, and chart extraction — only pdfium text extraction + embedding. Embedding runs locally via the bundled HuggingFace model by default (no remote NIM needed). It's strictly better to have a text-only index than no index at all: the per-query pdfium text-extract fallback re-extracts a full PDF *per query*, which is both slow and expensive. Page-element detection may emit warning logs when its remote endpoint isn't reachable; the warnings are non-fatal as long as the embedding step itself succeeds.

Don't pre-OCR, don't pre-chunk, don't write Python wrappers — the CLI handles extraction + (optionally) page-element detection + OCR + embedding + LanceDB insert in one shot.

## Query turn — the WHOLE workflow

```bash
HITS=$(retriever query "<the user's question>" --top-k 10)
echo "$HITS" | jq -r '.[] | "rank=\(.rank // 0) page=\(.page_number) pdf=\(.pdf_basename) type=\(.metadata.type // "?") text=\(.text[:200])"'
```

That's your FIRST tool call on every query turn. Do not Read, Glob, Grep, or list PDFs before this — those duplicate what `retriever query` already did.

**No narration between tool calls.** Do not write "Let me search…", "I'll now analyze…", "The retriever returned…", or any other commentary. Every assistant token you emit between the `retriever query` Bash call and the `Write` of `./output.json` becomes input tokens (and cached input tokens) for every subsequent turn in this session — quadratic cost. Go straight from reading the jq summary to writing the JSON file. The only assistant text in a query turn should be the tool calls themselves.

Each hit has: `text`, `pdf_basename`, `page_number` (int, **1-indexed**: the first page of a PDF is page `1`), `pdf_page` (string composite key `"<basename>_<page_number>"` — not a number, don't use it as one), `_distance`, and `metadata` (JSON with `type` ∈ `text|table|chart|image`).

**Then write `./output.json` directly from $HITS:**

- `final_answer`: synthesize from the top hits' `text`. Include the exact number / name / date / row / column the question asks for, plus the source PDF and 0-indexed page. One paragraph. No restating the question, no hedging caveats. If the chunks talk *around* the fact but don't state it, run ONE `retriever pdf stage page-elements --input-dir ./pdfs --method pdfium --json-output-dir /tmp/pdf_text` and read `/tmp/pdf_text/<top_pdf>.json` for the rank-1 page (or rank-2 if rank-1 is metadata) — that almost always surfaces the exact figure. Then synthesize. **If after both calls the asked-for fact still isn't in the evidence, write `final_answer` that says so explicitly** — e.g. "The retrieved pages do not state [X] for [entity]; the closest content is [Y]." Do NOT invent, extrapolate, or generate plausible-sounding content from adjacent material. A confidently-wrong answer scores worse than an honest "not in the retrieved pages".
- `ranked_retrieved`: one entry per hit in the order `retriever query` returned: `{"doc_id": "<pdf_basename without .pdf>", "page_number": <int>, "rank": <i+1>}`. Up to 10. Duplicate `(doc, page)` is fine. **Indexing:** the retriever's `page_number` is 1-indexed. If the task's output schema says 0-indexed (e.g. "first page is page 0"), emit `hit.page_number - 1`; if the task says 1-indexed or doesn't specify, emit `hit.page_number` as-is.

**Before writing `final_answer`, re-read the question.** If it lists multiple entities, years, or categories, your answer must address each one explicitly — even if for some of them the chunks say "not provided" or contain no data. Missing entities lose more judge points than imprecise numbers.

**Charts and images need extra caution — this is the single biggest source of judge=2/3 trials.** When `metadata.type` of a hit is `chart` or `image`, its `text` field is a model-generated transcription that frequently:

- reverses direction words (`increase`↔`decrease`, `rose`↔`fell`, `surge`↔`drop`), and
- rounds or misreads exact percentages (e.g. transcribing 12% as 20%).

If a question asks for an exact percentage or a directional claim **and the evidence is only a chart/image hit** (no `text`-type hit corroborates the same number or direction):

1. Run the targeted `retriever pdf stage page-elements --method pdfium` text-extract on the rank-1 PDF (this counts as your second tool call) and look for the number in prose.
2. If prose confirms the chart number, assert it confidently.
3. If prose doesn't mention it, **quote the chart transcription verbatim with an explicit hedge in `final_answer`**: "The chart on page N indicates [verbatim phrase] (chart-derived, not verified against prose)." Do NOT restate the chart's number as a confident fact.

When both a chart hit and a text hit cover the same fact, always prefer the text hit's number.

After writing the file, STOP. No print, no summary, no further tool calls.

### Hard limits (cost discipline)

- ONE `retriever query` per turn. ONE optional targeted text-extract on the rank-1 PDF if the chunks miss the asked-for fact. That's the budget — it is a hard cap, not a soft preference.
- After your 2nd tool call, write `final_answer` with what you have and STOP. If both calls left the asked-for fact unresolved, write `final_answer` that **explicitly states the retrieved pages don't contain the requested fact** (naming the closest related content if any) — **do not run more tool calls hunting for it, and do not extrapolate a plausible value.** Long-running query turns (5+ tool calls, 1M+ cache-read tokens) cost ~5× a disciplined turn and usually still produce the wrong answer.
- Don't read whole PDFs.
- Don't make speculative Read/Glob/Grep calls "to confirm". The retriever already found the relevant pages — trust the ranking.
- Don't spawn agents, write plans, or make todo lists. The workflow above is the workflow.

### If the index is missing or `retriever query` returns `[]`

Means ingest didn't complete (e.g. the text-only pipeline still hit the turn wall, or the table is empty). Tight fallback using the retriever's own pdfium-based extractor (always available — same binary the agent just used for `retriever query`):
1. `ls ./pdfs/` (one call) to see filenames.
2. Pick the SINGLE PDF whose name best matches the question.
3. ONE call: `retriever pdf stage page-elements --input-dir ./pdfs --method pdfium --json-output-dir /tmp/pdf_text`. This emits a JSON sidecar per PDF at `/tmp/pdf_text/<basename>.json` containing per-page text primitives — pdfium only, no OCR, no NIM, fast.
4. `jq` (or read directly) `/tmp/pdf_text/<name>.json` for the chosen PDF and synthesize from the per-page text. If the answer isn't there, still write your best guess based on the filename + extracted pages plus a one-sentence acknowledgement of uncertainty in `final_answer`. Then stop.

Do NOT keep doing text-extract calls across many PDFs to hunt — that exhausts the turn budget. Better to answer partially than to time out. Never re-run `retriever ingest`.

For an unlisted subcommand: `retriever <subcommand> --help`.

## Failure modes

- **First `ingest` takes ~60s+** — vLLM warmup. Expected.
- **First `query` takes ~10–15s** — embedder cold-start. Expected.
- **Empty result** — ingest didn't run. Use the fallback above.
- **`Clamping num_partitions ...`** — informational on tiny corpora, not an error.
- **Low-relevance top hit on tiny corpus** — look at `_distance` *gaps* between hits, not absolute values.
