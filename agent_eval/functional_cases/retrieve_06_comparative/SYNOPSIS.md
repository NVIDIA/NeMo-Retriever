# Synopsis — RETRIEVE: comparative (compare entities or sections across chunks)

**What user task this covers.** A user wants a **side-by-side comparison** built from a
document corpus — "compare A's and B's metric in this year," "side-by-side these two
attributes," or "compare this section in doc A vs doc B." Success is not a single retrieved
fact: it is a **structured two-sided answer** where each side carries **one citation** and
the facts on **both** sides are correct. When the two things being compared live in different
documents, the two citations must point at **different sources**; when they are two entities
inside the **same** document (e.g. two people in a table), the two citations must resolve to
**different rows**.

**How we test it.** Five agent prompts, each handing the agent a small real PDF corpus and a
"compare X vs Y" question. We check the agent drives the `retriever` CLI correctly:
`retriever ingest <files>` to build the corpus (format auto-detected — there is **no**
`--input-type` flag; the table cases add `--use-table-structure` so individual cells come
back as their own rows), then one or two `retriever query` calls to pull the comparable
chunks, and finally that the agent **composes a structured side-by-side from the cited hits**.
This is graded **operationally** (right structure, one citation per side, correct facts) —
**not** with a RAGAS judge; that lives in the separate performance suites.

**The five tests, simplest to hardest:**

1. **Two entities, one doc** — compare James and Susan in 2019 from one table. Ground truth:
   **James 978, Susan 922**. Side-by-side, one citation per side, distinct table rows.
2. **A different year** — same two people for **2023**, where the lead flips: **James 539,
   Susan 970** (Susan higher). Same shape, new metric, guards against memorized numbers.
3. **A section across two docs** — compare the "New Hampshire" collection year in
   `woods_frost.pdf` (**1923**) against what the Giraffe is doing in `multimodal_test.pdf`'s
   Table 1 (**"Driving a car," at the beach**). Now the two citations are **different sources**.
4. **All five people at once** — the full 2019 row (**Bill 665, Amy 600, James 978, Ted 707,
   Susan 922**; James highest). Stresses per-side accuracy: five distinct cells, no bleed.
5. **Acceptance gate** — a mixed 3-doc corpus where the agent must produce, in one structured
   answer, both a within-doc comparison (James 978 vs Susan 922, distinct rows) and an
   across-doc comparison (Frost 1923 vs Giraffe "Driving a car," distinct sources) — without
   mis-citing a fact to the wrong document.

**Why this order.** Each rung adds exactly one new thing: first the structured two-sided
answer with distinct rows, then a new metric (with a flipped verdict), then the cross-document
axis (distinct sources), then per-side accuracy at scale (five sides), then everything
composed — both citation forms in a single answer with no cross-contamination.

**Status.** Tests are authored and grounded in the real CLI (`--dry-run`) and the skill's
ingest/query references; **not yet run live**. Live runs need a reachable embedding backend
(hosted `integrate.api.nvidia.com` with an API key, or a local GPU) and the table-structure
backend for the table cases. See `README.md` for the full spec and `cases.json` for the
machine-gradable definitions.
