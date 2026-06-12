# Query turn — the WHOLE workflow

```bash
timeout 2000 <RETRIEVER_VENV>/bin/retriever query "<the user's question>" --format evidence --hybrid --top-k 10 \
  --embed-model-name nvidia/llama-nemotron-embed-1b-v2 --query-embed-backend hf \
  | tee ./evidence.json
```

That's your FIRST tool call on every query turn, run **exactly** as one pipeline (cold runs take ~20–30s; wait for it — don't background it or fire parallel queries). Do not Read, Glob, Grep, or list PDFs first — those duplicate what `retriever query` already did. `--format evidence` returns answer-ready JSON:

```
{ "evidence": [ { text, source, locator, modality, fidelity, score, citation } ], "coverage": {...} }
```

`tee ./evidence.json` keeps the full result in the cwd (not `/tmp` — clobbered under parallel queries). Read it back only as needed (`<RETRIEVER_VENV>/bin/python -c "import json; print(json.load(open('./evidence.json'))['evidence'][0]['text'])"`); pulling all chunks' text into context inflates cached prompt size on every later turn.

**No narration between tool calls.** Do not write "Let me search…", "The retriever returned…", or any commentary — every token between the `query` call and the `Write` of `./output.json` becomes input (and cached input) for every later turn (quadratic cost). Go straight from reading the result to writing the file.

Each evidence item carries: `text`, `source` (doc basename), `locator` (`{kind: page, value: <int, 1-indexed>}`), `modality` (`text|table|chart|image|audio|video_frame`), **`fidelity`** (`verbatim > ocr > transcribed > vlm_caption`), `score`, and `citation` (ready-to-quote source + locator). Hit ORDER is authoritative; `score` is informational.

## Trust by fidelity — the core of a correct answer

A number or directional claim resting ONLY on a `vlm_caption` (chart/image transcription) is **unconfirmed** — chart transcriptions often flip direction words (`increase`↔`decrease`) or misread exact figures. Prefer `verbatim`/`ocr`/`table` evidence for exact values. If the figure you need appears only in a `vlm_caption`, quote it verbatim and tag "(chart-derived, unconfirmed)" unless a higher-fidelity item states the same fact. Never upgrade a low-fidelity reading to a confident fact.

## When the answer isn't in the first result

Re-`query` only when the top evidence doesn't yet answer — for a genuinely *distinct* sub-question (per entity when comparing/listing), or **with the exact term/phrase** when `coverage.thin_spots` flags a miss or a specific ID/code/figure isn't in the returned text (the fused BM25 leg matches exact strings semantic search skips — e.g. re-query `"mRNA-1273"` to surface every chunk that names it). Read `coverage.thin_spots` to tell "broaden the search" from "out of corpus". Do NOT re-issue reworded variants of the same question, reach for `pdftotext`/`pdfgrep`, or open the LanceDB table yourself — `query` already searched the whole corpus.

## Compose your reply from the evidence

- `final_answer`: **lead with the direct answer** — the exact figure (in the evidence's own units) or a bare Yes/No, for the exact entity asked — then support it. Synthesize from the evidence `text`. One paragraph, no restating the question, no hedging caveats. **Re-read the question**: address every entity / year / category it names, even those the evidence marks "not provided" (missing entities lose more judge points than imprecise numbers). If the asked-for fact isn't in the evidence, say so explicitly — never invent or extrapolate from adjacent material.
- `ranked_retrieved`: one entry per evidence item in returned order: `{"doc_id": "<source>", "page_number": <locator.value>, "rank": <i+1>}`. Up to 10. **Indexing:** `locator.value` is 1-indexed; if the task's schema says 0-indexed, emit `value - 1`, else emit as-is.

After your reply, STOP. No print, no summary, no further tool calls.
