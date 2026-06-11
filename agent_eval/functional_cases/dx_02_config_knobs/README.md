# Functional test suite — DX: tinker with any exposed configuration knob across the pipeline

Agent-driven functional tests for the **NeMo Retriever Library skill**, built against the
real CLI in `nemo_retriever/src` (`retriever ingest` / `retriever query`). Each test is a
self-contained triple — a prompt, a per-case `cases/<id>/data/` folder, and an expected
output naming the correct `retriever` subcommand(s) and flags.

This suite covers the **DEVELOPER EXPERIENCE (DX)** job: a developer wants to **tinker with
an exposed configuration knob** — chunk size/overlap, the embedding model, the number of
reranked results — for a single run, and have the override **take effect immediately, with
no rebuild**, be **visible in the trace**, and **not stick** to later runs.

---

## The user task under test

> **JTBD: DEVELOPER EXPERIENCE — P0.** "Tinker with any exposed configuration knob across
> the pipeline."

**Success criteria for the row (operational pass):**

| Dimension | Target for this row |
|---|---|
| Quality (operational) | binary pass: an exposed knob is overridden on the CLI and **(a)** the override **applies on the next call with no rebuild/reinstall/restart**, **(b)** the override is **visible in the trace** (the new value reached the resolved operator plan, not the default), and **(c)** **default behavior is restored** on a subsequent call (no sticky override unless explicitly persisted) |
| Time | **fast — ≤ 30s** per case. Ingest-time knobs are proven via `--dry-run` (offline, no network/GPU); query-time knobs via the observed effect on one tiny pre-ingested doc |
| Trigger rate | ≥ 95% — a "change/override/use a different `<knob>` for this run, don't rebuild" prompt must fire the skill |
| Subcommand accuracy | ≥ 90% — each knob maps to its **real** flag (below); ingest-time knobs proven with `--dry-run`. No invented flags. |
| Token usage | tracked, not gated |

Seed queries this suite is derived from (paraphrased into the prompts):
- *"Load `<doc>.pdf` but use a 1000-token chunk size with 100-token overlap."*
- *"Use a different embedding model for this run — don't rebuild anything."*
- *"Bump the number of reranked results from 5 to 20 for this one query."*

---

## How the CLI behaves for this task (grounded in `--help` + `--dry-run`)

Verified on `retriever 2026.06.10.devXXXX` with `retriever ingest <file> [flags] --dry-run`
(offline, no network/GPU — the resolved plan is JSON) and `retriever ingest/query --help`.

**The `--dry-run` plan IS the trace for ingest-time knobs.** `retriever ingest <doc>
--dry-run` prints the fully resolved plan; the overridden value lands in a specific field,
so we can read it back and compare to the default. `retriever query` has **no `--dry-run`**,
so query-time knobs (top-k / rerank) are proven by the **observed effect** on the next call
(the size/shape of the returned hit array), not a plan dump.

**Knob → real flag → where it shows in the trace → verified default vs override:**

| Knob | Real flag(s) | Trace location | Default (verified) | Override (verified) |
|---|---|---|---|---|
| Chunk size | `retriever ingest --text-chunk --text-chunk-max-tokens N` | `split_config.pdf.max_tokens` | **1024** | **1000** |
| Chunk overlap | `--text-chunk-overlap-tokens N` | `split_config.pdf.overlap_tokens` | **150** | **100** |
| Embedding model | `retriever ingest --embed-model-name <name>` | `embed.embed_model_name` / `embed.model_name` | embed block **null/unset** unless an `--embed-*` flag is set (spec default embedder `nvidia/llama-nemotron-embed-1b-v2`) | `nvidia/nv-embedqa-e5-v5` |
| Reranked result count | `retriever query --top-k N --rerank` | size of returned JSON hit array (no `--dry-run` for query) | `--top-k` **10**; `--rerank` **off** (`--no-rerank`) | `--top-k 20 --rerank` (on) |

Notes grounded from the live dry-run plan:
- `split_config` is **only materialized when `--text-chunk` is enabled**; without it the
  field is `null`. With `--text-chunk` and no size flags, the plan shows the defaults
  **1024 / 150**; adding `--text-chunk-max-tokens 1000 --text-chunk-overlap-tokens 100`
  flips the same fields to **1000 / 100**.
- The `embed` block is **only materialized in the dry-run plan when an `--embed-*` flag is
  passed**. With `--embed-model-name nvidia/nv-embedqa-e5-v5` the plan's `embed.embed_model_name`
  **and** `embed.model_name` both read `nvidia/nv-embedqa-e5-v5`.
- Every `--dry-run` is **offline** — the override is resolved purely from CLI args, so it
  cannot involve a rebuild/reinstall/restart. That is exactly the DX guarantee under test.

Convention in every command: `RETRIEVER=$RETRIEVER_VENV/bin/retriever`.

---

## The five tests (increasing complexity)

| # | id | What it adds over the previous rung | Subcommands |
|---|---|---|---|
| 1 | `dx-ck-001` | **Baseline.** One ingest-time knob (chunk 1000/100) visible in the `--dry-run` plan vs the 1024/150 default. The DX floor: override one knob, see it in the trace, no rebuild. | `ingest … --dry-run` |
| 2 | `dx-ck-002` | **Different stage / model swap.** Embedding model overridden via `--embed-model-name`; lands in the plan with no reinstall (the swap most likely thought to need a rebuild). | `ingest … --dry-run` |
| 3 | `dx-ck-003` | **Query-time knob.** `--top-k` 5→20 + `--rerank` honored on the next query, **no re-ingest** — proven by observed effect, not a plan. | `ingest`, `query --top-k --rerank` |
| 4 | `dx-ck-004` | **Non-stickiness.** Overridden call then a plain call → defaults (1024/150) restored; the override did not leak. | 2× `ingest … --dry-run` |
| 5 | `dx-ck-005` | **Acceptance gate.** Multiple knobs across stages in one workflow (chunk at ingest + top-k/rerank at query), each visible in its own trace, then defaults restored. | `ingest … --dry-run`, `ingest`, `query --top-k --rerank`, `ingest … --dry-run` |

The ladder: T1 proves one ingest-time knob reaches the trace; T2 changes only the knob (a
model swap on a different pipeline stage); T3 changes the stage (query-time, observed effect,
no re-ingest); T4 adds non-stickiness (default restored after an override); T5 composes
everything — knobs on two stages in one workflow, each traced, defaults restored after.

---

### T1 — `dx-ck-001` · chunk-size override visible in the dry-run plan  *(complexity 1)*
- **Satisfies:** DX config-knob operational-pass core, simplest form.
- **Data:** `cases/dx-ck-001/data/woods_frost.pdf` (text-bearing, best for the chunk knob).
- **Expected:** `RETRIEVER ingest data/woods_frost.pdf --text-chunk --text-chunk-max-tokens
  1000 --text-chunk-overlap-tokens 100 --dry-run` → plan shows
  `split_config.pdf.max_tokens = 1000`, `overlap_tokens = 100` (vs the verified default
  **1024 / 150**). Offline; no rebuild.

### T2 — `dx-ck-002` · embedding-model override reaches the plan  *(complexity 2)*
- **Satisfies:** "use a different embedding model for this run — don't rebuild."
- **Data:** `cases/dx-ck-002/data/woods_frost.pdf`.
- **Adds:** a different knob on a different stage (embedding). `RETRIEVER ingest … 
  --embed-model-name nvidia/nv-embedqa-e5-v5 --dry-run` → plan's `embed.embed_model_name`
  and `embed.model_name` both read `nvidia/nv-embedqa-e5-v5`. No reinstall/rebuild.

### T3 — `dx-ck-003` · query-time top-k 5→20 on the next query  *(complexity 3)*
- **Satisfies:** "bump the number of reranked results from 5 to 20 for this one query."
- **Data:** `cases/dx-ck-003/data/woods_frost.pdf` (ingested once, then queried).
- **Adds:** a query-time knob, proven by observed effect (`query` has no `--dry-run`), with
  **no re-ingest**. `RETRIEVER query "…" --top-k 20 --rerank` honors the raised top-k and
  turns reranking on; answer grounded in `woods_frost.pdf` p1 ("miles to go before I sleep").
  **Caveat:** the 2-page doc holds fewer than 20 chunks, so `--top-k 20` is a ceiling — the
  pass condition is that the knob is honored (raised from the default, rerank on), not a
  literal 20-hit array.

### T4 — `dx-ck-004` · default restored, override not sticky  *(complexity 4)*
- **Satisfies:** "default behavior is restored on subsequent calls (no sticky override)."
- **Data:** `cases/dx-ck-004/data/woods_frost.pdf`.
- **Adds:** non-stickiness. Call 1 dry-run with `--text-chunk-max-tokens 1000
  --text-chunk-overlap-tokens 100` → 1000/100; call 2 dry-run with `--text-chunk` only →
  **1024/150** (default restored). Each CLI invocation resolves independently; nothing
  persisted, no rebuild between calls.

### T5 — `dx-ck-005` · acceptance gate, multiple knobs across the pipeline  *(complexity 5)*
- **Satisfies:** the composed operational-pass for the DX config-knob row.
- **Data:** `cases/dx-ck-005/data/woods_frost.pdf` + `cases/dx-ck-005/data/table_test.pdf`.
- **Expected (one workflow, no rebuild between steps):**
  1. `ingest … --text-chunk-max-tokens 1000 --text-chunk-overlap-tokens 100 --dry-run` →
     plan shows **1000/100** (ingest-time knob traced);
  2. the **real** `ingest … --text-chunk-max-tokens 1000 --text-chunk-overlap-tokens 100` →
     non-zero rows for both files (`branch_summary pdf:2`);
  3. `query "What value did James have in 2019?" --top-k 20 --rerank` → honors the raised
     top-k + rerank with **no re-ingest**; answer **James 2019 = 978** citing `table_test.pdf` p1;
  4. plain `ingest … --text-chunk --dry-run` → back to **1024/150** (defaults restored).
- **Adds (the gate):** knobs on two pipeline stages in one workflow, each visible in its own
  trace (dry-run plan for chunking; honored flags/hit array for query), defaults restored
  after, no invented flags, no rebuild/reinstall between steps.

---

## Running / grading

Mount each case's `cases/<id>/data/` into the agent workdir, give it the `prompt`, and grade
against `pass_when` in `cases.json`. The defining checks for this suite:
**(a)** each knob maps to its **real** flag (no invented flags);
**(b)** the override is **visible in the trace** — for ingest-time knobs, the `--dry-run`
plan field (`split_config.pdf.{max_tokens,overlap_tokens}`, `embed.embed_model_name`)
holds the overridden value; for the query-time knob, the raised `--top-k`/`--rerank` is
honored on the next call;
**(c)** the override **applies with no rebuild/reinstall/restart** (the `--dry-run` calls
are offline; the query reuses the existing index — no re-ingest);
**(d)** the override is **not sticky** — a following plain call reverts to the verified
defaults (1024/150);
**(e)** grounded values are correct (default 1024/150 vs override 1000/100; embed
`nvidia/nv-embedqa-e5-v5`; top-k default 10 → 20; answer 978 for the T5 query).

**Note on live runs.** Expected outputs are grounded in the CLI **source + `--dry-run`** and
`--help`; the suite has **not** been run live yet. The four `--dry-run` steps are fully
offline. The two steps that execute for real — the T3/T5 `ingest` and `query` — need a
reachable embedding/reranker backend (the **hosted** `integrate.api.nvidia.com` /
`ai.api.nvidia.com` endpoints with `NVIDIA_API_KEY`, making small billable calls, or a
**local GPU** per the SETUP-GPU suite). Running live would capture the real per-file row
counts at the overridden chunking, the actual hit-array size as top-k grows from ~5 toward
20 on a larger corpus, latencies, and token baselines.
