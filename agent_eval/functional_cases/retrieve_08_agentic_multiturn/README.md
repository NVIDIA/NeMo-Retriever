# Functional test suite — RETRIEVE: agentic retrieval, multi-turn (query rewrite + multi-turn)

Agent-driven functional tests for the **NeMo Retriever Library skill**, built against the
real CLI in `nemo_retriever/src` (`retriever ingest` / `retriever query`). Each test is a
self-contained triple — a prompt, a per-case `cases/<id>/data/` folder, and an expected
output naming the correct `retriever` subcommand(s), flags, and the **rewritten standalone
query the agent must issue for each conversational turn**.

This suite covers the **RETRIEVE** job: **agentic multi-turn retrieval** — the user holds a
conversation over an ingested table, and the agent must (a) **rewrite** vague/elliptical
turns into well-formed retriever queries and (b) carry **multi-turn context** across turns,
issuing a correct **standalone** `retriever query` each time.

---

## The user task under test

> **JTBD: RETRIEVE — P0.** "Agentic retrieval — multi-turn (query rewrite and multi-turn)."

**Success criteria for the row (operational pass — NOT RAGAS):**

| Dimension | Target for this row |
|---|---|
| Quality (operational) | binary pass per turn: each conversational turn is resolved by the **agent** into a well-formed **standalone** `retriever query` (pronouns / ellipsis / comparatives rewritten using prior-turn context), the right subcommand + flags are used, and the returned grounded hit yields the correct answer. The retriever is **stateless per call**; the agent owns conversation state. |
| Time | **agentic RETRIEVE — ≤ 5 min** per case (a longer budget than the other RETRIEVE rows, which are ≤ 1 min, because each case is a **multi-call conversation**: one ingest + a **sequence** of standalone queries, one per turn). |
| Trigger rate | ≥ 95% — a "have a multi-turn conversation / answer this follow-up over this table" prompt must fire the skill, and must fire `retriever query` **again for each follow-up turn** |
| Subcommand accuracy | ≥ 90% — each turn runs `retriever query "<rewritten standalone q>"` (with `--content-types text,table` after `ingest --use-table-structure`). **No `--input-type` flag** (does not exist). **No** attempt to pass conversation history / session id to the CLI (there is no such flag). |
| Token usage | tracked, not gated |

Seed conversation this suite is derived from (paraphrased into the prompts):
- Turn 1 *"What does the table show for James?"* → Turn 2 *"And Susan?"* (follow-up relies
  on carried context) → Turn 3 *"Which of them scored higher in 2019?"*
- Plus a **query-rewrite** seed: a vague utterance the agent rewrites into a precise
  retriever query.

---

## CRITICAL caveat — NRL is stateless; multi-turn is an agent-layer behavior

**NRL 26.05 has NO conversational / session state.** `retriever query` takes exactly one
query string and returns hits for **that string only**. There is **no** `--history`,
`--session`, `--context`, or `--turn` flag. This is the spec's negative-test category
**(d)**: a bare elliptical follow-up such as *"What about the Q3 numbers?"* (or here, *"And
Susan?"*) carries **no** multi-turn context inside NRL — sent verbatim, it retrieves on
those words alone.

Therefore **multi-turn + query rewrite must be implemented AT THE AGENT LAYER.** The agent:
1. tracks conversation state itself (entities, year, prior results / derived winners);
2. **rewrites** each user turn into a self-contained query that names the entity / year /
   comparison explicitly (resolving pronouns and ellipsis from prior turns);
3. issues a **fresh, stateless** `retriever query` for that rewritten string;
4. computes any comparison / aggregation (e.g. "which is higher") itself, since NRL returns
   **cells, not verdicts**.

These tests verify the **agent** rewrites/resolves references across turns and issues
correct standalone queries — **NOT** that NRL maintains a session.

---

## How the CLI behaves for this task (grounded in `--dry-run` + skill references)

Verified with `retriever ingest <file> --dry-run` (offline, no network) and the skill's CLI
references:

- **Format auto-detection.** `retriever ingest <file>` resolves `.pdf` from the extension.
  There is **no `--input-type` flag**.
- **Table cells must be materialized.** The conversation targets values in a **table**, so
  every case ingests with **`--use-table-structure`** (flips `use_table_structure` true and
  `table_output_format` to `markdown`, materializing **nemotron-table-structure-v1**) so
  James/Susan year cells are queryable as discrete `content_type=table` rows. Queries use
  `--content-types text,table`.
- **`retriever query` is single-shot and stateless.** It accepts one positional query
  string and the flags above (`--top-k`, `--candidate-k`, `--content-types`, table/uri,
  embed/rerank). **No history/session flag exists.** Multi-turn is therefore a *sequence* of
  independent queries the agent composes.
- **Ingest success line:** `Ingested N file(s) -> M row(s) in LanceDB lancedb/<table>.`
- **Query hit shape:** a JSON array; each hit has exactly `{source, page_number, text}`;
  `page_number` is a 1-indexed int.

Convention in every command: `RETRIEVER=$RETRIEVER_VENV/bin/retriever`.

### Fixture ground truth (read from the real files)

`table_test.pdf` (p1) — `Year × {Bill, Amy, James, Ted, Susan}`:

| Year | James | Susan |
|---|---|---|
| 2023 | 539 | 970 |
| 2020 | 922 | 994 |
| 2019 | **978** | **922** |
| 2018 | 664 | 976 |

So in **2019, James (978) > Susan (922)**; in **2020, Susan (994) > James (922)**.
`woods_frost.pdf` (used in rung 4 only) — p1 Frost poem, p2 a *different* table (Frost
collections × years, e.g. *New Hampshire — 1923*); it is loaded so that *"the numbers
table"* becomes a reference the agent must disambiguate to `table_test.pdf`.

---

## The five tests (increasing complexity)

| # | id | What it adds over the previous rung | Subcommands |
|---|---|---|---|
| 1 | `retrieve-mt-001` | **Query rewrite (single turn).** One vague utterance → agent rewrites it into a precise standalone query naming entity + year. No prior context to carry yet — isolates rewrite. | `ingest --use-table-structure`, `query --content-types text,table` |
| 2 | `retrieve-mt-002` | **Multi-turn carry.** A second, elliptical turn ("And Susan?") resolvable only from turn-1 context → agent carries "2019" and rewrites it. | `ingest …`, 2× `query` |
| 3 | `retrieve-mt-003` | **Comparative/derived turn.** A third turn ("which scored higher that year?") references both prior entities; agent computes the comparison itself over standalone-query results. | `ingest …`, 3× `query` |
| 4 | `retrieve-mt-004` | **Reference resolution.** A pronoun/anaphor turn whose referent is a *prior result* ("whoever scored higher"), plus "that year"; a 2nd doc makes "the table" itself an ambiguous reference. | `ingest (2 files) …`, 3× `query` |
| 5 | `retrieve-mt-005` | **Acceptance gate.** One end-to-end conversation composing all four: vague→rewrite, ellipsis carry, agent-computed comparison, and a derived anaphor — each a correct standalone query; explicitly notes NRL is stateless. | `ingest …`, 4× `query` |

The ladder: T1 isolates **query rewrite** (one turn, nothing to carry); T2 adds a **second
turn** whose ellipsis must be filled from turn 1; T3 adds a **comparative/derived** third
turn computed at the agent layer; T4 adds **reference resolution** where the referent is a
*derived result* (and "the table" must be disambiguated across two docs); T5 **composes**
all of it into one acceptance conversation. Each rung adds exactly one new dimension.

---

### T1 — `retrieve-mt-001` · single vague turn → rewrite  *(complexity 1)*
- **Satisfies:** the query-rewrite half, simplest single-turn form (no context to carry).
- **Data:** `cases/retrieve-mt-001/data/table_test.pdf`.
- **Expected:** `ingest … --use-table-structure`; the vague *"what's his number for the 2019
  line?"* is **rewritten** to a standalone query naming **James** and **2019**, e.g.
  `query "James value in 2019 from the table" --content-types text,table` → **James 2019 =
  978** (table_test.pdf p1). Fails if the raw vague text is passed verbatim.

### T2 — `retrieve-mt-002` · two-turn ellipsis carry  *(complexity 2)*
- **Satisfies:** the multi-turn half — a follow-up resolvable only from turn-1 context.
- **Data:** `cases/retrieve-mt-002/data/table_test.pdf`.
- **Adds:** a **second** turn. Turn 1 → James 2019 = 978; the bare **"And Susan?"** is
  rewritten using the carried **2019** constraint into a standalone `query "Susan value in
  2019 from the table"` → **Susan 2019 = 922**. Two distinct query calls. Fails if "And
  Susan?" is sent to NRL verbatim expecting NRL to remember "2019" (it is stateless).

### T3 — `retrieve-mt-003` · three-turn comparative  *(complexity 3)*
- **Satisfies:** the full seed conversation (James → Susan → comparison).
- **Data:** `cases/retrieve-mt-003/data/table_test.pdf`.
- **Adds:** a **third, comparative/derived** turn ("Which of them scored higher that
  year?"). The agent carries both entities and "that year" = 2019, retrieves both 2019 cells
  via standalone queries, and **computes** the comparison itself: **James higher in 2019
  (978 > 922)**. NRL returns cells, not a "higher" verdict.

### T4 — `retrieve-mt-004` · reference resolution (derived referent)  *(complexity 4)*
- **Satisfies:** the explicit "query rewrite resolves pronouns/ellipsis" clause, stressed.
- **Data:** `cases/retrieve-mt-004/data/table_test.pdf` **and** `…/woods_frost.pdf`.
- **Adds:** the follow-up's referent is the **result** of a prior turn, not a literal.
  Turn 1 "the numbers table" must be disambiguated to **table_test.pdf** (not woods_frost's
  Frost-collections table) → James 2020 = 922. Turn 2 "that year" = 2020 → Susan 2020 = 994,
  agent computes **Susan higher**. Turn 3 "**whoever scored higher that year**" resolves to
  the **derived** entity **Susan**, and the new ask is **2018** (the carried year is *not*
  reused) → standalone `query "Susan value in 2018 …"` → **Susan 2018 = 976**.

### T5 — `retrieve-mt-005` · acceptance gate, full conversation  *(complexity 5)*
- **Satisfies:** the composed operational-pass for the agentic multi-turn RETRIEVE row.
- **Data:** `cases/retrieve-mt-005/data/table_test.pdf`.
- **Expected:** one `ingest … --use-table-structure`, then a **sequence** of standalone
  queries, one per turn:
  - **T1 (vague→rewrite):** "his 2019 line" → James 2019 = **978** (p1);
  - **T2 (ellipsis carry):** "And Susan?" → Susan 2019 = **922** (p1);
  - **T3 (comparative, agent-computed):** James higher in 2019 (**978 > 922**);
  - **T4 (derived anaphor):** "whoever won" = **James**, "the year before" = **2018** →
    James 2018 = **664** (p1).
- **Adds (the gate):** every turn must be a correct **standalone** `retriever query` with
  references resolved in the rewrite, none relying on CLI state, **and** the agent must
  explicitly note that NRL is stateless and that it carried James/Susan/2019/the-winner at
  the agent layer. All answers carry `source` + `page_number` citations.

---

## Running / grading

Mount each case's `cases/<id>/data/` into the agent workdir, give it the `prompt`, and grade
against `pass_when` in `cases.json`. The defining checks for this suite:
**(a)** the agent ingests with `--use-table-structure` so cells are `content_type=table`
rows; **(b)** each conversational turn fires a **fresh** `retriever query` — the follow-ups
re-trigger the skill; **(c)** each query string is a **standalone rewrite** with pronouns /
ellipsis / derived referents resolved (NOT the raw user turn, NOT reliant on any CLI
session/history flag — none exists); **(d)** answers match the grounded values (James 2019 =
978; Susan 2019 = 922; James higher in 2019; Susan 2020 = 994; Susan 2018 = 976; James 2018
= 664) with `table_test.pdf` p1 citations; **(e)** comparative/derived turns are computed at
the **agent layer**; **(f)** no `--input-type` flag is used (it does not exist).

**Note on live runs.** Expected outputs are grounded in the CLI **source + `--dry-run`** and
the skill's ingest/query references; the suite has **not** been run live yet. A live run
requires a reachable retrieval backend (embedding/rerank — hosted `integrate.api.nvidia.com`
with an `NVIDIA_API_KEY`, which makes small billable calls, or a local GPU per the SETUP-GPU
suite) and the table-structure extraction backend (hosted `ai.api.nvidia.com` or local GPU).
Running live would capture the real per-turn hit counts, table-cell fidelity, the latency of
the full multi-call conversation (against the ≤ 5 min budget), and token baselines.

**Statelessness caveat (restated).** **NRL 26.05 is stateless per call — there is no session
state and no `--history`/`--session`/`--context` flag.** All multi-turn behavior in this
suite (rewriting the vague turn, carrying "2019", computing "which is higher", and resolving
"whoever won" / "the year before") is implemented at the **agent layer**: a correct run is a
**sequence of standalone `retriever query` calls** whose query strings the agent composed
from conversation state it tracked itself. The retriever does not remember anything between
calls.
