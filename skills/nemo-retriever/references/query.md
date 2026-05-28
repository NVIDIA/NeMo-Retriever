# Query turn — the WHOLE workflow

## Filename fast path — try BEFORE `retriever query`

If the user's question literally contains a PDF basename from `./pdfs/` (stem ≥6 chars, with or without `.pdf`, case-insensitive), skip semantic search. Direct pdfium extraction on the named file is faster and avoids semantic-search misses — the right doc is given, and pages rank by query-token overlap.

```bash
<RETRIEVER_VENV>/bin/python <skill_dir>/scripts/filename_fast_path.py "<the user's question>"
```

`<skill_dir>` is the "Base directory for this skill" announced at load time. Stdout is one of:

- `NO_MATCH` — no literal basename in the query. Fall through to the standard `retriever query` workflow below.
- `NO_TEXT` — matched file is image-only / pdfium got no text. Also fall through.
- A JSON object with `"ranking"` followed by `---TOP_PAGE_TEXT---` and the top page's raw text — that's the fast-path hit. Write `./output.json` directly: copy the `"ranking"` entries verbatim into `ranked_retrieved`, synthesize `final_answer` from the printed `TOP_PAGE_TEXT` (exact number/name/date; one paragraph; honest "not in the retrieved pages" if the fact genuinely isn't there; no chart hedging needed — pdfium extracts text only). Then STOP. Fast-path total: 2 tool calls (this Bash + Write). Do NOT also call `retriever query` — it's mutually exclusive.

## Standard path: `retriever query`

```bash
<RETRIEVER_VENV>/bin/retriever query "<the user's question>" --top-k 10 --embed-model-name nvidia/llama-nemotron-embed-1b-v2 --rerank \
  | tee /tmp/hits.json \
  | <RETRIEVER_VENV>/bin/python -c "import json,sys; [print(f'rank={h.get(\"rank\",0)} page={h[\"page_number\"]} pdf={h[\"pdf_basename\"]} type={h.get(\"metadata\",{}).get(\"type\",\"?\")}') for h in json.load(sys.stdin)]"
```

Run that **exactly** as a single pipeline — do not split it into `HITS=$(...)` + `echo "$HITS" | <RETRIEVER_VENV>/bin/python -c ...` (the assignment swallows stdout, the pipe sees nothing, you waste 3 bash calls recovering). Stdout is clean JSON (model-init logs are silenced at the CLI layer); leave stderr unredirected so real errors surface on the first call. The summary above lists only rank/page/pdf/type — to read hit text for synthesizing `final_answer`, parse `/tmp/hits.json` directly. The top hit's text is one one-liner away: `<RETRIEVER_VENV>/bin/python -c "import json; print(json.load(open('/tmp/hits.json'))[0]['text'])"` (or `[i]` for the rank-(i+1) hit). Fetch only what you need — pulling all 10 hits' text into context inflates cached prompt size on every subsequent turn.

That's your FIRST tool call on every query turn. Do not Read, Glob, Grep, or list PDFs before this — those duplicate what `retriever query` already did.

**No narration between tool calls.** Do not write "Let me search…", "I'll now analyze…", "The retriever returned…", or any other commentary. Every assistant token you emit between the `retriever query` Bash call and the `Write` of `./output.json` becomes input tokens (and cached input tokens) for every subsequent turn in this session — quadratic cost. Go straight from reading the summary to writing the JSON file. The only assistant text in a query turn should be the tool calls themselves.

Each hit has: `text`, `pdf_basename`, `page_number` (int, **1-indexed**: the first page of a PDF is page `1`), `pdf_page` (string composite key `"<basename>_<page_number>"` — not a number, don't use it as one), `_distance`, and `metadata` (JSON with `type` ∈ `text|table|chart|image`).

## Keyword/regex search across the corpus

If you need exact text matches that semantic `retriever query` may have skipped — e.g. "find every mention of 'mRNA-1273' across all PDFs" — use:

```bash
<RETRIEVER_VENV>/bin/python <skill_dir>/scripts/grep_corpus.py "<regex>" [--max-hits 50]
```

It scans the LanceDB table the retriever already built — no PDF re-extraction. Output is `<pdf>:p<page>:<type>:  ...<snippet>...` per hit; `NO_MATCH` if nothing. Counts against the same "one optional follow-up call" budget as the targeted text-extract (mutually exclusive — pick one).

Don't reach for `pdftotext`, `pdftohtml`, or `pdfgrep` — they're system tools that aren't guaranteed installed on the user's machine. The retriever venv bundles pdfium and `lancedb`; `grep_corpus.py` and `retriever pdf stage page-elements --method pdfium` cover the same use cases without that dependency.

## Write `./output.json` directly from the hits

- `final_answer`: synthesize from the top hits' `text`. Include the exact number / name / date / row / column the question asks for, plus the source PDF and 0-indexed page. One paragraph. No restating the question, no hedging caveats. If the chunks talk *around* the fact but don't state it, run ONE `<RETRIEVER_VENV>/bin/retriever pdf stage page-elements ./pdfs --method pdfium --json-output-dir /tmp/pdf_text --compact-json` and `Read` `/tmp/pdf_text/<top_pdf>.pdf.pdf_extraction.json` for the rank-1 page (or rank-2 if rank-1 is metadata) — that almost always surfaces the exact figure. Then synthesize. **If after both calls the asked-for fact still isn't in the evidence, write `final_answer` that says so explicitly** — e.g. "The retrieved pages do not state [X] for [entity]; the closest content is [Y]." Do NOT invent, extrapolate, or generate plausible-sounding content from adjacent material. A confidently-wrong answer scores worse than an honest "not in the retrieved pages".
- `ranked_retrieved`: one entry per hit in the order `retriever query` returned: `{"doc_id": "<pdf_basename without .pdf>", "page_number": <int>, "rank": <i+1>}`. Up to 10. Duplicate `(doc, page)` is fine. **Indexing:** the retriever's `page_number` is 1-indexed. If the task's output schema says 0-indexed (e.g. "first page is page 0"), emit `hit.page_number - 1`; if the task says 1-indexed or doesn't specify, emit `hit.page_number` as-is.

**Before writing `final_answer`, re-read the question.** If it lists multiple entities, years, or categories, your answer must address each one explicitly — even if for some of them the chunks say "not provided" or contain no data. Missing entities lose more judge points than imprecise numbers.

## Charts and images — the single biggest source of judge=2/3 trials

When `metadata.type` of a hit is `chart` or `image`, its `text` field is a model-generated transcription that frequently:

- reverses direction words (`increase`↔`decrease`, `rose`↔`fell`, `surge`↔`drop`), and
- rounds or misreads exact percentages (e.g. transcribing 12% as 20%).

If a question asks for an exact percentage or a directional claim **and the evidence is only a chart/image hit** (no `text`-type hit corroborates the same number or direction):

1. Run the targeted `<RETRIEVER_VENV>/bin/retriever pdf stage page-elements --method pdfium` text-extract on the rank-1 PDF (this counts as your second tool call) and look for the number in prose.
2. If prose confirms the chart number, assert it confidently.
3. If prose doesn't mention it, **quote the chart transcription verbatim with an explicit hedge in `final_answer`**: "The chart on page N indicates [verbatim phrase] (chart-derived, not verified against prose)." Do NOT restate the chart's number as a confident fact.

When both a chart hit and a text hit cover the same fact, always prefer the text hit's number.

After writing `./output.json`, STOP. No print, no summary, no further tool calls.
