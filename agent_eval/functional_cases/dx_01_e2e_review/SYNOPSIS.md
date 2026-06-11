# Synopsis — DX #1: end-to-end pipeline review

**What user task this covers.** A developer on a **clean machine** wants to confirm the
**whole platform pipeline works end-to-end**: install it, set it up, load some documents,
ask a question, and get a correct answer **with a citation** — every stage wired, **no
broken steps**, **no stale model names**, no dead (404) endpoints. Success means the full
chain **install → setup → ingest → query → cite** runs through with no manual repair and
returns a correct, timely answer pointing at a real `(source, page)`.

**How we test it.** Five agent prompts that paraphrase the seed queries ("walk me through
getting set up… ask a question… surface a citation"). Each hands the agent a small set of
PDFs (or none, for the install rung) and checks that the agent drives the `retriever` CLI
through the happy-path chain — `retriever --version`, `retriever ingest <corpus>/`,
`retriever query "<q>" --top-k 5` — with **current** model names (embedder
`nvidia/llama-nemotron-embed-1b-v2`, reranker `llama-nemotron-rerank-1b-v2`, extraction
`nemotron-page-elements-v3 → nemotron-ocr-v2 → nemotron-table-structure-v1`) and a
**resolvable** `(source, page_number)` citation on the final answer.

**The five tests, simplest to hardest:**

1. **Version / install sanity** — install on a clean machine and confirm the CLI is there
   and **current** (no stale version). The pipeline's entry point.
2. **Ingest stage** — load a small PDF corpus and confirm ingest ran end-to-end with a
   non-zero row count and no broken extraction step (current extraction names).
3. **Query stage** — ask a question and get a **correct, grounded answer** back, proving the
   query/embedding stage is wired (current embedder).
4. **Citation surfaces and resolves** — the answer's `(source, page)` pointer is present,
   non-null, and **resolves** to the right page of the right file (table cell ⇒ 978 on p1).
5. **Acceptance gate** — the **whole pipeline** in one pass: install → ingest → query →
   cite, with no broken step, **no stale name**, no 404, a correct cited answer, and a
   ≤ 10-minute deadline. This is the test the others build up to.

**Why this order.** Each rung lights up exactly one more stage of the chain — install, then
ingest, then query, then the citation — and the final rung composes them into the real
pass/fail gate. A reviewer can see precisely which stage breaks first.

**Relationship to the SETUP suites.** The SETUP suites test *where* a single stage runs
(CPU/hosted vs. local GPU). This suite is different on purpose: it is a **whole-pipeline
integration review** — every stage wired correctly, **current** model names, no 404s — not
one capability and not host-specific.

**Status.** Tests are authored and grounded in the real CLI, `references/install.md`, and an
offline `retriever ingest --dry-run`; **not yet run live** (a live run may pull GPU/CUDA-13
wheels and/or hit billable hosted endpoints). See `README.md` for the full spec and
`cases.json` for the machine-gradable definitions.
