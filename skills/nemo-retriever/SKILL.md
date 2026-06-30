---
name: nemo-retriever
description: "Use when the user wants to search, query, extract, transcribe, describe, quote, filter, or aggregate across documents — PDFs, scanned forms / images (`.jpg` `.png` `.tiff`), Office (`.docx` `.pptx`), text (`.html` `.txt`), audio (`.mp3` `.wav` `.m4a`), or video (`.mp4` `.mov`). Prefer this over native Read / Grep for multi-file or non-PDF corpora. Not for: editing files, web browsing, single-file plain-text lookups, fine-tuning."
license: Apache-2.0
allowed-tools: Bash Write Read
---

# nemo-retriever

The `retriever` CLI indexes a folder of PDFs into LanceDB (`retriever ingest`) and serves vector search over it (`retriever query`). For any task about searching/answering questions across a folder of PDFs, use this CLI — do not write a custom RAG.

**Beyond PDFs and beyond semantic search.** `retriever ingest` also handles images, Office, HTML, TXT, audio, and video — see `references/setup.md` for the per-format recipe and `references/install.md` for the install extras (`[multimedia]`, libreoffice, ffmpeg). The query turn is a single command — see **§Query turn** below (inline, no reference read needed); `references/cli/query.md` holds only the fallback detail (exact-term, chart text-extract, compose-reply). Don't fall back to native Read/Grep/Python on non-PDF inputs.

## Install (if `retriever` is missing)

If `command -v retriever` returns nothing, follow `references/install.md` to install the NeMo Retriever Library before proceeding. It prints `RETRIEVER_VENV=<path>`; substitute that path for `<RETRIEVER_VENV>` in every example in this skill (setup, query, troubleshooting, and the CLI references).

## Workflow — read the reference for the current phase, then execute

| Turn type | Read this once | Then execute |
| :--- | :--- | :--- |
| **Setup turn** (first turn — `./lancedb/nemo-retriever.lance` doesn't exist) | `references/setup.md` | Build the index |
| **Query turn** (every subsequent turn — user asks a question) | **§Query turn** below (command inline — no reference read needed) | Run it, then `Write` `./output.json` *(eval-harness contract only — for general use, just answer in chat)* |
| Anything errored or returned empty | `references/troubleshooting.md` | Apply the named recovery; do not improvise |

## Query turn — run this, then write the answer

Run TWO retrieval passes and capture each to a file — never stream to the terminal (the top-10 evidence overflows the ~10 KiB tool-output limit and truncates mid-data). The passes are complementary: semantic hybrid finds topically-relevant pages; a **lexical (sparse/BM25) pass on the exact term** finds the precise page a number/code/proper-noun lives on — which dense embedding retrieval often misses.

```bash
# 1) Semantic pass — the full question, hybrid (dense + lexical fusion).
<RETRIEVER_VENV>/bin/retriever query "<question>" --format evidence --hybrid --top-k 10 > ./ev_dense.json
# 2) Lexical pass — the EXACT term/figure/code/proper-noun the question targets (just the term, not the whole question; that's what makes BM25 precise).
<RETRIEVER_VENV>/bin/retriever query "<exact term, e.g. Management VaR / Level 3 / a specific code>" --format evidence --retrieval-mode sparse --top-k 10 > ./ev_lex.json
<RETRIEVER_VENV>/bin/python - <<'PY'
import json, glob
for f in sorted(glob.glob('./ev_*.json')):
    for i, x in enumerate(json.load(open(f))['evidence']):
        print(f, i, x['source'], x['locator']['value'], x['fidelity'], '|', x['text'][:140].replace('\n', ' '))
PY
```

Always run the lexical pass for the specific named figure/metric/code/entity, and **query as many times as it takes to be sure** — one query per named term, and **re-query freely to disambiguate**. These filings repeat similar tables (e.g. many "Level 3" tables for different segments/categories); when several candidates come back, run more targeted queries to find the **consolidated / total** figure the question asks for (e.g. query `"consolidated total Level 3 assets liabilities"` or the exact row/section name), and read the competing candidate pages before deciding. Prefer more targeted queries over too few — under-querying is the main cause of wrong answers here. Those are your FIRST calls — don't `ls`/`find`/`sed`/Read to orient first. Then:
- **Pull only the rows you need from the file(s) — never print a whole chunk** (a chunk can exceed the ~10 KiB limit and truncate, so `print(text)`/`cat` loses data). Filter to the lines naming the term:
  `<RETRIEVER_VENV>/bin/python -c "import json; t=json.load(open('./ev_lex.json'))['evidence'][0]['text']; print('\n'.join(l for l in t.splitlines() if 'Management VaR' in l))"` (swap in the file, item index, and term).
- **Ground every figure in a source line — quote before you write.** For each number/name your answer will state, first locate the exact line in the evidence that says it and copy the value straight from that line. **Never write a figure you cannot point to verbatim in the evidence** — if it isn't there, answer "not provided"; do not infer, round, or compute it from other cells.
- **Prefer a prose statement over a table cell.** When a figure is stated in a sentence (e.g. *"Level 3 assets and liabilities were $9,194 million and $28,755 million, respectively"*), use that — it's unambiguous. Read a table cell only for values prose doesn't give, and bind it by its **row label × column header** in the markdown table, not by position.
- **Verify before writing.** Each figure in your draft must appear, character-for-character, in a line you actually pulled from `ev_*.json`. If a draft figure isn't found in the evidence, it's wrong — replace it with the evidence value or mark it "not provided".
- **Answer verbatim-figure-first** — copy each figure in the document's own units and scale (`$27,132 million`, not `$27.1 billion`/`27,132`); don't round, rescale (M↔B), or reformat. Cover every entity / period / category the question names. Lead with the values (or a bare Yes/No).
- **Trust by fidelity** (`verbatim > ocr > transcribed > vlm_caption`): a number resting ONLY on a `vlm_caption` is unconfirmed — quote it tagged "(chart-derived, unconfirmed)" unless a higher-fidelity item agrees. Never fabricate from adjacent text.
- **`ranked_retrieved` = the union of pages across both passes** — dedup by doc+page, ordered by relevance, up to 10.
- Open `references/cli/query.md` ONLY for the fallback path (chart text-extract, compose-reply detail).

For the full `retriever ingest` CLI spec, see `references/cli/ingest.md`. For `retriever query` flags, `<RETRIEVER_VENV>/bin/retriever query --help` is authoritative (and faster) — you do not need it for routine turns.

Before ingesting a mixed folder, inventory extensions (`find <dir> -name '*.*' | sed 's/.*\.//' | sort -u`) — `--input-type=auto` silently drops anything outside the supported set. See `references/troubleshooting.md` "Unsupported file types".

## Hard limits (apply to every turn)

- **Setup turn**: build the index in one shell command (see `references/setup.md`). STOP after the index lands.
- **Query turn**: query until the answer is fully supported — a semantic pass plus a lexical (sparse) pass per named term, **re-querying as needed to disambiguate similar tables** (commonly 4–8 retriever calls), then filter the rows you answer from. Don't stop early to save calls; **stop only when each figure is pinned to a source line.**
- **No narration between tool calls.** Tokens you emit between calls become input + cached input for every later turn — quadratic cost. Go straight from reading the summary to writing the JSON file.
- **Banned**: `TodoWrite`, Glob, Grep, `Read` of whole PDFs, re-running setup, spawning subagents, speculative "confirmation" calls.

Spend the calls you need to get the figures right — accuracy matters more than minimizing calls here. Only avoid genuinely wasteful loops (re-running identical queries, reading whole PDFs, 15+ calls). **A fully-supported answer beats a cheap partial one.**
