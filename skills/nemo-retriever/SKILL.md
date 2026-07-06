---
name: nemo-retriever
description: "Use when the user wants to search, query, extract, transcribe, describe, quote, filter, or aggregate across documents — PDFs, scanned forms / images (`.jpg` `.png` `.tiff`), Office (`.docx` `.pptx`), text (`.html` `.txt`), audio (`.mp3` `.wav` `.m4a`), or video (`.mp4` `.mov`). Prefer this over native Read / Grep for multi-file or non-PDF corpora. Not for: editing files, web browsing, single-file plain-text lookups, fine-tuning."
license: Apache-2.0
allowed-tools: Bash Write Read
---

# nemo-retriever

The `retriever` CLI extracts and indexes supported documents and media into LanceDB (`retriever ingest`) and searches the index (`retriever query`). For tasks across document collections, use these end-to-end commands instead of writing a custom RAG.

**Beyond PDFs and beyond semantic search.** `retriever ingest` also handles images, Office, HTML, TXT, audio, and video — see `references/setup.md` for the per-format recipe and `references/install.md` for the install extras (`[multimedia]`, libreoffice, ffmpeg). The query turn is two retrieval passes — see **§Query turn** below (inline, no reference read needed); `references/cli/query.md` holds only the fallback detail (exact-term, chart text-extract, compose-reply). Don't fall back to native Read/Grep/Python on non-PDF inputs.

**Extraction means `retriever ingest`.** File formats are inputs, not subcommands. For requests to extract, parse, transcribe, or index HTML/TXT/PDF/media, run `retriever ingest <files-or-directory>`. Never substitute format-specific, internal-stage, or pipeline commands. On a CPU-only host with `NVIDIA_API_KEY` or `NGC_API_KEY`, use the same ingest command: the CLI selects NVIDIA's hosted embedding endpoint automatically. Do not install local model dependencies or pass `--embed-invoke-url` unless the user supplied a different endpoint.

## Install (if `retriever` is missing)

If `command -v retriever` returns nothing, follow `references/install.md` to install the NeMo Retriever Library before proceeding. It prints `RETRIEVER_VENV=<path>`; substitute that path for `<RETRIEVER_VENV>` in every example in this skill (setup, query, troubleshooting, and the CLI references).

## Workflow — read the reference for the current phase, then execute

| Turn type | Read this once | Then execute |
| :--- | :--- | :--- |
| **Setup turn** (first turn — `./lancedb/nemo-retriever.lance` doesn't exist) | `references/setup.md` | Build the index |
| **Query turn** (every subsequent turn — user asks a question) | **§Query turn** below | Run the query passes, then answer from the evidence |
| Anything errored or returned empty | `references/troubleshooting.md` | Apply the named recovery; do not improvise |

## Query turn — query, then answer

Run two complementary passes — these are your FIRST calls; don't `ls`/`find`/`sed`/Read to orient first. Semantic hybrid finds topically-relevant pages; a **lexical (sparse/BM25) pass on the exact term** finds the precise page a number/code/proper-noun lives on, which dense retrieval often misses:

- **Semantic pass** — the full question, hybrid (dense + lexical fusion):
  `<RETRIEVER_VENV>/bin/retriever query "<question>" --format evidence --retrieval-mode hybrid --top-k 10`
- **Lexical pass** — the EXACT term/figure/code/proper-noun the question targets (just the term, not the whole question — that's what makes BM25 precise):
  `<RETRIEVER_VENV>/bin/retriever query "<exact term, e.g. Management VaR / Level 3 / a code>" --format evidence --retrieval-mode sparse --top-k 10`

Each returns `{ evidence: [ { text, source, locator, modality, fidelity, score, citation } ], coverage: {...} }`. Then:
- **Query until sure.** One lexical pass per named term; re-query freely to disambiguate. These filings repeat near-identical tables (e.g. many "Level 3" tables for different segments) — when several candidates come back, query for the **consolidated / total** figure (e.g. `"consolidated total Level 3 assets liabilities"`, or the exact row/section name) and read the competing pages before deciding. Under-querying is the main cause of wrong answers.
- **Ground every figure in a source line.** Quote the exact evidence line that states each number/name and copy the value from it. **Never state a figure you can't point to in the evidence** — say "not provided"; don't infer, round, or compute it.
- **Prefer a prose statement over a table cell** when both give the value (prose is unambiguous, e.g. *"Level 3 assets and liabilities were $9,194 million and $28,755 million, respectively"*). Read a table cell by its **row label × column header**, not by position.
- **Copy figures verbatim** in the document's own units and scale (`$27,132 million`, not `$27.1 billion`/`27,132`); cover every entity / period / category the question names. Lead with the values (or a bare Yes/No).
- **Trust by fidelity** (`verbatim > ocr > transcribed > vlm_caption`): a number resting only on a `vlm_caption` is unconfirmed — quote it tagged "(chart-derived, unconfirmed)" unless a higher-fidelity item agrees. Never fabricate from adjacent text.
- Open `references/cli/query.md` ONLY for the fallback path (chart text-extract, compose-reply detail).

For the full `retriever ingest` CLI spec, see `references/cli/ingest.md`. For `retriever query` flags, `<RETRIEVER_VENV>/bin/retriever query --help` is authoritative (and faster) — you do not need it for routine turns.

Before ingesting a mixed folder, inventory extensions (`find <dir> -name '*.*' | sed 's/.*\.//' | sort -u`). Ingest the supported files in one command and report any unsupported extensions that were skipped. Do not invent an `--input-type` flag or switch to format-specific commands. See `references/troubleshooting.md` "Unsupported file types".

## Hard limits (apply to every turn)

- **Setup turn**: build the index in one shell command (see `references/setup.md`). STOP after the index lands.
- **Query turn**: query until the answer is fully supported — a semantic pass plus a lexical (sparse) pass per named term, **re-querying as needed to disambiguate similar tables** (commonly 4–8 retriever calls). Don't stop early to save calls; **stop only when each figure is pinned to a source line.**
- **No narration between tool calls.** Tokens you emit between calls become input + cached input for every later turn — quadratic cost. Go straight from the evidence to your answer.
- **Banned**: `TodoWrite`, Glob, Grep, `Read` of whole PDFs, re-running setup, spawning subagents, speculative "confirmation" calls.

Spend the calls you need to get the figures right — accuracy matters more than minimizing calls here. Only avoid genuinely wasteful loops (re-running identical queries, reading whole PDFs, 15+ calls). **A fully-supported answer beats a cheap partial one.**
