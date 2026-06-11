# Functional test suite — RETRIEVE: NL question → grounding context with structured citations

Agent-driven functional tests for the **NeMo Retriever Library skill**, built against the
real CLI in `nemo_retriever/src` (`retriever ingest` / `retriever query`). Each test is a
self-contained triple — a prompt, a per-case `cases/<id>/data/` folder, and an expected
output naming the correct `retriever` subcommand(s) and flags.

This suite covers the **RETRIEVE** job: accept a **natural-language question** and return
**retrieved grounding context with structured citations** — a source identifier, a page
number, and a chunk/element id — so every answer is traceable back to a real passage.

---

## The user task under test

> **JTBD: RETRIEVE — P0.** "Accept a natural-language question and return retrieved
> grounding context with structured citations (source identifier, page, chunk ID)."

**Success criteria for the row (operational pass — NOT RAGAS):**

| Dimension | Target for this row |
|---|---|
| Quality (operational) | binary pass: a NL question returns grounding hits where **every** hit carries a structured citation (**non-null** source identifier + **non-null** 1-indexed page number), the citation **resolves** to a real ingested row, and the answer is **grounded in** (quotes/uses) the cited row's text. **No null citations.** This is operational pass, not a RAGAS judge score (RAGAS lives in the separate performance-eval suites). |
| Time | **RETRIEVE ≤ 1 min** per case (small corpus already ingested + one query) |
| Trigger rate | ≥ 95% — an "answer this and cite where it came from" prompt must fire the skill |
| Subcommand accuracy | ≥ 90% — `retriever ingest <paths>` then `retriever query "<NL question>" --top-k 5`; the structured citation in each hit is the deliverable. **No `--input-type` flag** (it does not exist). |
| Token usage | tracked, not gated |

Seed queries this suite is derived from (paraphrased into the prompts):
- *"What was `<metric>` in `<year>`?"* (→ T4: James, 2019 → 978, with the citation text proving it)
- *"What's the main heading of this document image?"* (→ the "name the source passage" framing in T1/T2)
- *"What's on page 14 of `<doc>.pdf`?"* — **deliberately NOT in this suite.** That query is
  really a **filtering** test (retrieve a specific page); it is left to **`retrieve_11_filter`**.
  This suite stays focused on **citations** (is the source+page emitted, correct, complete,
  and grounded?), not on page/metadata filtering.

---

## How the CLI behaves for this task (grounded in source + `--dry-run`)

Verified with `retriever ingest <file> --dry-run` (offline, no network) and the CLI hit
shape in `src/nemo_retriever/cli/main.py`:

- **Format auto-detection.** `retriever ingest data/woods_frost.pdf` resolves `.pdf` from
  the extension (`--dry-run` → `branch_summary pdf:1`). A folder of N PDFs → `pdf:N`. There
  is **no `--input-type` flag**.
- **Retrieval is a single `query`.** `retriever query "<NL question>" --top-k 5` runs the
  NL question against the ingested table and prints a **JSON array of hits**.
- **The structured citation is the hit itself.** At the CLI layer **each hit has exactly
  three keys** — `{source, page_number, text}`:
  - `source` — the **source identifier**, resolvable to the ingested file's **filename and
    full path** (under the case's `data/` folder).
  - `page_number` — an **int**, **1-indexed**.
  - `text` — the retrieved passage (the grounding context; the answer must be supported by it).
  There is **no** `metadata` / `_distance` key at the CLI layer.
- **The internal citation carries the full structured set.** The CLI projects each hit from
  an internal citation object that also exposes `source_id`, `filename`, full **path**, the
  chunk/element id, and a human-readable **`pdf_page`** string of the form
  `<DOCSTEM>_<PAGE>` (the spec's example is `WALMART_2017_10K_42` = page 42 of
  `WALMART_2017_10K.pdf`). For these fixtures the analogous `pdf_page` values are
  `woods_frost_1`, `woods_frost_2`, `table_test_1`, etc. The CLI surfaces `source` +
  `page_number`; the assertions below treat the **1-indexed `page_number`** as the page
  field and require it to be **non-null and correct**.
- **Ingest success line:** `Ingested N file(s) -> M row(s) in LanceDB lancedb/<table>.`

Convention in every command: `RETRIEVER=$RETRIEVER_VENV/bin/retriever`.

**Ground truth (verified by reading the fixtures):**
- `woods_frost.pdf` **p1** = the poem "Stopping by Woods on a Snowy Evening" by Robert
  Frost — owner's *"house is in the village"*, *"miles to go before I sleep"* (repeated).
  **p2** = "Frost's Collections" table: *A Boy's Will 1913, North of Boston 1914, Mountain
  Interval 1916, **New Hampshire 1923**, West Running Brook 1928, A Further Range 1937*.
- `table_test.pdf` **p1** = a Year × {Bill, Amy, James, Ted, Susan} grid; **James 2019 = 978**.
- `multimodal_test.pdf` = 3-page "TestingDocument" (distractor in the acceptance corpus).

---

## The five tests (increasing complexity)

| # | id | What it adds over the previous rung | Subcommands |
|---|---|---|---|
| 1 | `retrieve-cit-001` | **Baseline.** NL question → grounded answer + **one** structured citation (non-null source + page), resolvable to the file. The citation floor. | `ingest`, `query --top-k 5` |
| 2 | `retrieve-cit-002` | **Correct page.** A **page-2-only** fact (collections table) must cite **page 2, not 1** — page-number accuracy, not just presence. | `ingest`, `query --top-k 5` |
| 3 | `retrieve-cit-003` | **No nulls.** A multi-hit result over **2 docs** where **every** hit has non-null source AND page — citation completeness across all hits/sources. | `ingest data/`, `query --top-k 5` |
| 4 | `retrieve-cit-004` | **Grounding.** The cited row's `text` must **literally contain** the answer value (James 2019 = **978**) — citation supports the answer. | `ingest`, `query --top-k 5` |
| 5 | `retrieve-cit-005` | **Acceptance gate.** NL question over a **3-doc** corpus: right doc + right page among distractors, every hit cited, citations resolve to real grounded rows. | `ingest data/`, `query --top-k 5` |

The ladder: T1 proves a citation comes back at all and is non-null; T2 adds page-number
**correctness** (a p2-only fact); T3 adds **completeness** (no null citation anywhere across
multiple sources); T4 adds **grounding** (the cited text actually contains the answer); T5
**composes** all three over a multi-doc corpus where the wrong document/page is an available
distractor.

---

### T1 — `retrieve-cit-001` · baseline NL question → answer + a source+page citation  *(complexity 1)*
- **Satisfies:** RETRIEVE-with-citations core, simplest form (one NL question → grounded
  answer + ≥1 non-null structured citation).
- **Data:** `cases/retrieve-cit-001/data/woods_frost.pdf`.
- **Expected:** `RETRIEVER ingest data/woods_frost.pdf` → non-zero rows;
  `RETRIEVER query "Where does the owner of the woods live?" --top-k 5` → answer **"the
  village"** (owner's *house is in the village*), with the top hit carrying a **non-null
  `source`** (resolves to `woods_frost.pdf`) and a **non-null `page_number` (1)**.

### T2 — `retrieve-cit-002` · the cited page_number is correct  *(complexity 2)*
- **Satisfies:** the **page** field of the citation, proven **accurate**.
- **Data:** `cases/retrieve-cit-002/data/woods_frost.pdf`.
- **Adds:** the answer lives **only on page 2** (the "Frost's Collections" table; page 1 is
  the poem). **Ground truth: New Hampshire (1923)**, and the cited `page_number` must be
  **2, not 1** — catching a citation that is present but points at the wrong page.

### T3 — `retrieve-cit-003` · every hit carries a citation (no nulls)  *(complexity 3)*
- **Satisfies:** the "every claim is cited" **completeness** clause across a multi-hit result.
- **Data:** `cases/retrieve-cit-003/data/{woods_frost.pdf, table_test.pdf}` (2-doc folder).
- **Adds:** a `--top-k 5` result spanning **two source files**; the assertion is that
  **every** hit in the array has a **non-null `source`** (resolving to one of the two files)
  **and** a **non-null `page_number`** — **zero** null citations.

### T4 — `retrieve-cit-004` · the citation's text supports the answer  *(complexity 4)*
- **Satisfies:** the **grounding** clause (citation resolves AND its text supports the answer).
- **Data:** `cases/retrieve-cit-004/data/table_test.pdf`.
- **Adds:** it is no longer enough for the citation to be present — the cited hit's `text`
  must **literally contain** the value used in the answer. **Ground truth: James 2019 = 978**;
  the cited row's `text` must contain **`978`** in the James/2019 context.

### T5 — `retrieve-cit-005` · acceptance gate, multi-doc, cite-every-claim  *(complexity 5)*
- **Satisfies:** the composed operational-pass for the RETRIEVE-citations row.
- **Data:** `cases/retrieve-cit-005/data/{woods_frost.pdf, table_test.pdf, multimodal_test.pdf}`.
- **Expected:** one `ingest data/` (3 PDFs), then `query "...miles to go before I sleep..."
  --top-k 5`. **Ground truth: "Stopping by Woods on a Snowy Evening" by Robert Frost,
  woods_frost.pdf page 1** — the **right doc and right page** among 2 distractor docs.
- **Adds (the gate):** composes correct **page** (T2) + **no-null completeness** across
  sources (T3) + **grounding** (T4): every hit non-null `source`+`page_number`, each citation
  **resolves to a real row** whose text supports the claim (the cited row's text contains
  *"miles to go before I sleep"*), answer in **≤ 1 min**.

---

## Running / grading

Mount each case's `cases/<id>/data/` into the agent workdir, give it the `prompt`, and grade
against `pass_when` in `cases.json`. The defining checks for this suite:
**(a)** the agent runs `ingest` then `query "<NL question>" --top-k 5`;
**(b)** **every** returned hit is shaped `{source, page_number, text}` with **non-null**
`source` and **non-null, 1-indexed** `page_number` (no null citations);
**(c)** the cited `page_number` is **correct** for the fact (T2: a p2-only fact cites p2);
**(d)** the cited row's `text` **supports** the answer (T4/T5: contains the value/phrase);
**(e)** each `source` **resolves** to a real ingested file (filename + full path), and over
a multi-doc corpus the answer cites the **right document** (T5);
**(f)** answers match ground truth (the village; New Hampshire 1923; James 2019 = 978;
"Stopping by Woods on a Snowy Evening" / Robert Frost) — and **no `--input-type` flag** is
used (it does not exist).

**Note on live runs.** Expected outputs are grounded in the CLI **source + `--dry-run`**
(the `{source, page_number, text}` hit shape and the `branch_summary pdf:N` ingest plan) and
in reading the real fixtures; the suite has **not been run live yet**. A live run requires a
reachable **embedding backend** to encode the query and corpus (the default bundled HF model,
or a hosted `integrate.api.nvidia.com` endpoint with `NVIDIA_API_KEY` on a no-GPU box, or a
local GPU per the SETUP suites) — text extraction for these PDFs is local pdfium (no network).
Running live would capture the real per-file **row counts**, the **actual `page_number`
values** emitted per hit, citation **resolvability** rates, retrieval **latencies** (against
the ≤ 1 min RETRIEVE SLA), and token baselines.
