# Synopsis — RETRIEVE: multi-source synthesis

**What user task this covers.** A user has loaded several documents and asks a question
whose answer lives across **more than one** of them — "what do A's, B's, and C's docs say
about X?", "across all the docs I loaded, what's the most commonly mentioned thing?", "across
the memos in this folder, list every item and a fact from each project." Success means NeMo
Retriever pulls the relevant chunks **from multiple different documents**, and the final
answer **stitches them together while crediting each source** — covering every document the
question expects, not just the easiest one.

**How we test it.** Five agent prompts, each handing the agent a small 2–3 PDF corpus loaded
into **one** index and a question that **cannot be answered from any single document**. We
check the agent drives the `retriever` CLI correctly: `retriever ingest a.pdf b.pdf [c.pdf]`
into one table (the dry-run shows all the PDFs collapse into one `pdf` branch — there is
**no** `--input-type` flag), then `retriever query "<cross-doc question>" --top-k 8` (top-k
large enough that several documents are represented; `--rerank` at the top rung so the best
hit from each doc surfaces). This is RETRIEVE graded **operationally** (correct grounded
answer + right subcommand + the multi-source gates) — **not** RAGAS scoring, which lives in
the separate performance-eval suites.

**The multi-source gates** every rung is built around:
1. retrieved hits span **≥ 2 distinct source documents** (not all from one doc);
2. the answer **cites each contributing source** by name;
3. the answer **covers all expected sources** — a relevant loaded source missing from the
   answer is a **partial fail**.

**The corpus (deterministic, facts spread across docs).** `woods_frost.pdf` (Frost poem +
a "Frost's Collections" table: New Hampshire 1923, West Running Brook 1928), `table_test.pdf`
(a numeric grid: James 2019 = 978, Susan 2023 = 970), and `multimodal_test.pdf` (Table 1:
Giraffe → Driving a car → At the beach). A verified shared theme: **all three contain a
table**, which the common-theme rung leans on.

**The five tests, simplest to hardest:**

1. **Multi-source floor** — a 2-PDF question needing one fact from each doc (New Hampshire =
   1923 from Frost; James 2019 = 978 from the grid), so the hits must come from both
   documents. The thing that makes this task different from single-source retrieval.
2. **Citation gate** — same two docs, but now the answer must **name which file** each fact
   came from; correct-but-unattributed fails.
3. **Common theme across three docs** — add a third document and ask what element they all
   share (a table), answerable only by retrieving from all three.
4. **Coverage gate** — one distinct fact from each of three named docs; **missing any one is
   a partial fail**, even if the others are right and span two sources.
5. **Acceptance gate** — one synthesis answer that satisfies all three gates at once
   (≥2 distinct sources, each cited, full coverage) as a per-document roll-up.

**Why this order.** Each rung adds exactly one new thing: first "does the answer pull from
≥2 docs at all," then "is each source credited," then a third document plus corpus-wide
common-theme synthesis, then the strict full-coverage requirement, then all three gates
composed into one acceptance answer.

**Status.** Tests are authored and grounded in the real CLI (`--dry-run` shows the 3-PDF
corpus as one `pdf:3` branch into one table) and in the verbatim text of the three fixtures;
**not yet run live**. Live runs need a reachable embedding backend (hosted
`integrate.api.nvidia.com` with an API key, or a local GPU) and the table-extraction stack
for the table-typed rungs. See `README.md` for the full spec and `cases.json` for the
machine-gradable definitions.
