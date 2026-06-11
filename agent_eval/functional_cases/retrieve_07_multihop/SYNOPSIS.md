# Synopsis — RETRIEVE: multi-hop retrieval (single source, multiple chunks)

**What user task this covers.** A user has **one** document and asks a question that can't be
answered by a single lookup — it needs **two chained steps**. First find an intermediate fact
(a year, a title, the entity with the largest value), then **use that fact** to look up the
real answer. Example: *"In which year did James score his maximum, and what did Susan score
that same year?"* — you can only ask about Susan's year once you've found James's max year.
Success means the agent does it as a genuine **two-hop chain over multiple chunks of the same
source**, not one combined query, and lands the exact right number. Time budget: ≤ 1 minute.

**How we test it.** Five agent prompts, each over a single small real document, where the
answer requires hop-1 → hop-2. We check the agent drives the `retriever` CLI correctly:
`retriever ingest <file>` once (with `--use-table-structure` for the table cases), then **two
sequential `retriever query` calls** — the second query built from what the first one
returned. This is **operational pass, not RAGAS**: we grade the chained-query signature plus
the exact grounded final value, both verified against the real fixtures.

**The five tests, simplest to hardest:**

1. **Explicit 2-hop** — the user spells out both hops. The agent just runs two queries:
   "year James scored 978?" → **2019** → "Susan in 2019?" → **922**. Establishes the
   chained-query pattern.
2. **Intermediate must be retrieved** — the linking value isn't in the prompt. Hop-1 must run
   to learn that "North of Boston" was published in **1914**; hop-2 then finds the next
   collection by year, **Mountain Interval (1916)**. (woods_frost.pdf)
3. **Superlative hop-1** — hop-1 is an argmax over a column: James's **maximum is 987 in
   2015** (not his 978-in-2019 value), then Susan in 2015 = **854**. Tests computing the
   intermediate, with the 978/2019-vs-987/2015 trap built in.
4. **Two distinct query calls, cross-page** — the hops land on different pages of one
   document (p2 table title-ish lookup → p1 poem), so a one-shot query is impossible: hop-1
   gets the poem title **"Stopping by Woods on a Snowy Evening"**, hop-2 uses it to fetch the
   repeated closing line **"And miles to go before I sleep"**. We assert two separate query
   calls in the trace.
5. **Acceptance gate** — composes a superlative hop-1 with a cross-cell hop-2: Susan's
   **maximum 994 in 2020**, then James in 2020 = **922**. Graded purely on the operational
   signature (one ingest, ≥ 2 chained queries, hop-2 built from hop-1) plus the exact value.

**Why this order.** Each rung adds exactly one new thing: T1 the chained-query pattern itself
(both hops handed over); T2 makes the linking value something you must *retrieve*; T3 makes
hop-1 a *computed superlative*; T4 forces the chain across pages so two distinct calls are
unavoidable and explicitly checked; T5 composes the superlative-then-cross-cell chain as the
acceptance gate for the row.

**Status.** Tests are authored and grounded in the real CLI (`--dry-run`) and in the fixture
content read directly off disk (all hop chains and final values verified); **not yet run
live**. Live runs need a reachable retrieval backend (hosted `ai.api.nvidia.com` /
`integrate.api.nvidia.com` with an API key, or a local GPU). See `README.md` for the full
spec and `cases.json` for the machine-gradable definitions.
