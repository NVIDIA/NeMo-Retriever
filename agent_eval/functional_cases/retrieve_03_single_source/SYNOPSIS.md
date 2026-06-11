# Synopsis — RETRIEVE: single-source Q&A (answer from ONE named document)

**What user task this covers.** A user points at **one** document — "What does this memo
recommend, and what's the total?", "Pull up the revenue numbers from *this* PDF", "What's
the company saying about R&D in *their* 2022 10-K?" — and wants a direct, correct answer
*from that document*, through the platform's RAG runtime, in under a minute. The whole point
is **scoping**: the answer has to come from the named document and nothing else, even when
other files are sitting right next to it.

**How we test it.** Five agent prompts, each handing the agent a small real document (or a
small multi-doc folder) and a question with a known answer. We check the agent drives the
`retriever` CLI correctly: `retriever ingest` to load the document(s), then a **scoped**
`retriever query` to answer. This is an **operational-pass** check (correct grounded answer +
right subcommand + right scope), **not** a RAGAS judge score — that lives in the separate
performance-eval suites.

**The honest scoping detail.** The CLI has **no** "filter to this one document" flag at query
time (no `--where`, no `--source`). So single-source scoping is done one of two real ways,
and the tests use both: **(a)** ingest only the named document into its **own table** and
query that table (nothing else can come back); or **(b)** load everything into one table and
**filter by the document's name** with a short LanceDB python snippet from the skill's
reference. We deliberately do **not** invent a `--where` flag the product doesn't have.

**The five tests, simplest to hardest:**

1. **One document, one question** — the index holds a single PDF, so "single-source" is free.
   Proves the basic load→ask→grounded-answer loop. (Ground truth: the poem says the woods'
   owner's house is "in the village".)
2. **Several files, one named** — multiple PDFs on disk, but the user names one. The agent
   ingests **only** that document into its own table and queries it. (Ground truth: James,
   2019 → **978**.)
3. **Prove the scope** — all documents in one shared index; the agent must **show** it
   restricted the lookup to the named document (list the distinct sources, count the scoped
   rows) before answering. (Ground truth: the Giraffe is "Driving a car".)
4. **Exact number, with a decoy** — two documents that *both* contain number tables; scope to
   the named one and return the exact cell, so the decoy table can't leak a wrong figure.
   (Ground truth: Susan, 2023 → **970**.)
5. **Acceptance gate** — a named **Word memo** sitting in a real multi-document folder; the
   agent must answer **both** what it recommends and the dollar total, using only the memo —
   no bleed from the neighboring files. (Ground truth: procure **512 H200 GPUs**, total
   **$14.8M**.)

**Why this order.** Each rung adds exactly one new thing: first "can it answer from a single
doc at all," then "scope to one named doc among several," then "*prove* the scope held," then
"get the exact number even with a look-alike decoy present," then everything composed — a
non-PDF named source in a genuinely mixed folder, returning a span *and* a number with no
cross-document bleed.

**Status.** Tests are authored and grounded in the real CLI (`--help` / `--dry-run`) and the
skill's `references/query.md`; **not yet run live**. Live runs need a reachable embedding/
reranker backend (hosted with an API key, or a local GPU) and, for test 5, the libreoffice
host package. See `README.md` for the full spec and `cases.json` for the machine-gradable
definitions.
