# agent_eval results — corpus-level batch 4

**Set:** `agent_corpus_level_batch_4` — 1001 queries across 3 domains
(vidore_v3_finance_en / hr / pharmaceuticals).
**Runs:** baseline (no retriever) and skill (nemo-retriever) for both **claude**
(`claude-opus-4-7`) and **codex** (`gpt-5.5`).
**Scoring:** recall@k vs manifest gold (page-index normalized); LLM judge
(`llama-3.3-nemotron-super-49b`); agent-aware retriever-execution detection; codex
cost derived from tokens × gpt-5.5 rates; token totals normalized to
non-cached-input + output + cache_read (apples-to-apples).

## Baseline vs Skill — claude & codex : PR 2201 (sha: 6c6b9f40406d519bbf6f689bbb3d5131fb815d11)

| run | succ | retr_att | retr_clean | retr_eng | r@1 | r@5 | r@10 | judge | tokens | cost | $/query |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline-claude | 0.999 | 0.00 | 0.00 | 0.00 | 0.314 | 0.456 | 0.474 | 4.26 | 558M | $564 | $0.56 |
| baseline-codex  | 1.00  | 0.00 | 0.00 | 0.00 | 0.323 | 0.477 | 0.503 | 4.19 | 436M | $686 | $0.69 |
| **skill-claude**| 1.00  | 0.99 | **0.99** | 0.99 | 0.312 | 0.544 | **0.574** | 4.18 | 319M | **$372** | **$0.37** |
| skill-codex     | 1.00  | 0.70 | 0.27 | 0.70 | 0.333 | 0.499 | 0.526 | 4.18 | 416M | $642 | $0.64 |

_recall over all 1001 gold-bearing queries; `retr_clean` = retriever command exited
cleanly, `retr_eng` = retriever engine returned hits (even if the exec didn't show a
clean exit)._

## Tokens by phase (setup vs query)

Token usage split into **setup** (agent-driven index-build turns — skill only; baseline has
no ingest) and **query** (the 1001 per-question turns), with input / cache / output. "Cache"
is shown as its two components: `cache_read` (context reused from cache) and `cache_creation`
(newly written to cache). Raw values as recorded — matches each run's `report.md`.

| run | phase | turns | input | cache_read | cache_creation | output | total |
|---|---|---|---|---|---|---|---|
| baseline-claude | setup | 0 | 0 | 0 | 0 | 0 | 0 |
| baseline-claude | query | 1001 | 18.2K | 525.6M | 27.3M | 5.1M | 558.0M |
| baseline-codex  | setup | 0 | 0 | 0 | 0 | 0 | 0 |
| baseline-codex  | query | 1001 | 429.0M | 372.1M | 0 | 7.2M | 808.2M† |
| **skill-claude**| setup | 3 | 42 | 502.5K | 38.7K | 3.2K | 544.5K |
| **skill-claude**| query | 1001 | 16.8K | 292.8M | 22.9M | 3.2M | 318.9M |
| skill-codex     | setup | 3 | 85.0K | 531.1K | 0 | 9.8K | 625.9K |
| skill-codex     | query | 1001 | 50.1M | 358.0M | 0 | 7.1M | 415.2M |

- **Setup is negligible vs query.** Skill index-build is 3 ingest turns totaling <1M tokens
  (claude 544K, codex 626K); baseline has none. ~99.8%+ of every run's tokens are query-side —
  so the headline `tokens`/`cost` columns are effectively all query.
- **Cache dominates the volume.** For claude, `cache_read` is ~92-94% of tokens (the skill +
  corpus context is re-fed from cache each turn); `output` is the small slice that drives cost
  (gpt-5.5/opus output rates). `cache_creation` is the one-time write of that reused context.
- **† Codex `input` is reported *inclusive* of `cache_read`** (OpenAI convention), so codex's
  raw `total` here (baseline-codex **808.2M**) is larger than the headline `tokens` column
  (**436M**), which normalizes `input → input − cache_read` to avoid double-counting cached
  tokens. Non-cached input ≈ `input − cache_read` (baseline-codex ≈ 57M; skill-codex ≈ 0 — its
  query input is almost entirely cache hits). Claude `input` is already cache-exclusive, so its
  raw `total` matches the headline.

## Skill − Baseline (per agent)

| agent | recall@10 | cost | retr_engine | judge |
|---|---|---|---|---|
| **claude** | 0.474 → **0.574** (+21%) | $564 → **$372** (−34%) | 0.00 → 0.99 | 4.26 → 4.18 |
| **codex**  | 0.503 → 0.526 (+4%)      | $686 → $642 (−6%)       | 0.00 → 0.70 | 4.19 → 4.18 |

## Read

- **The skill helps both agents and is cheaper than baseline for both** — biggest for
  claude (**+21% recall, −34% cost**); codex gains modestly (+4% recall, −6% cost).
- **skill-claude is the standout overall**: best recall@10 (0.574), lowest cost
  ($0.37/query), fewest tokens — the retriever hands it focused chunks, so it reads far
  less than baseline (which ingests whole PDFs).
- **Execution-model contrast (now measured for both agents):** claude's
  `retr_clean ≈ retr_engine ≈ 0.99` (Bash runs to completion); codex ran `retriever
  query` in **0.70** of turns and got hits back nearly every time (`engine ≈ attempted
  ≈ 0.70`), but only **0.27 clean** — its ~1s exec-yield backgrounds the call, so clean
  exits rarely register. Codex also reached for the `retriever` **CLI** less than claude
  (0.70 vs 0.99) — but the other 30% mostly used the **same index via the Python API**, so
  effective index usage is ~87% (see *Where the skill-codex 30% went* below).
- **Baselines are clean/blind** (retr ≈ 0 for both — the PATH shim / deny-list held), so
  the recall gains are genuine retriever lift, not leakage.
- **Codex costs more than claude in both profiles** despite comparable/fewer total
  tokens — gpt-5.5's $30/1M output rate plus codex's reasoning-token volume.

## Where the skill-codex 30% went (non-CLI retriever)

`retr_attempted = 0.70` counts only the `retriever` **CLI**. The other **30% (302 of 1001
queries)** issued no `retriever` command — but most still used the **same prebuilt index**,
just through the Python API. Classifying the 302 by what actually read the data:

| method (of the 302) | queries | % of 302 | % of all 1001 |
|---|---|---|---|
| Direct LanceDB query in Python (`open_table('nemo-retriever').search()/to_pandas()`) | 125 | 41% | 12% |
| LanceDB **and** PDF extract (index → then read the page) | 41 | 14% | 4% |
| Direct PDF text extraction (`pdftotext -layout pdfs/x.pdf \| rg …`) | 57 | 19% | 6% |
| Neither — filename/metadata listing (`rg --files ./pdfs`, `ls`) | 79 | 26% | 8% |

**So the CLI number understates index usage.** Folding in direct-Python LanceDB access:

- **~166 of the 302 (55%) still hit the index** via `lancedb.connect().open_table('nemo-retriever')`
  instead of `retriever query` — the path `query.md` documents for metadata/non-semantic lookups.
  → **effective index usage ≈ 70% (CLI) + ~17% (direct Python) ≈ 87% of all queries.**
- **~6% read *only* the source PDFs** (`pdftotext`), a true retrieval bypass; a further ~4%
  read a page with `pdftotext` *after* first locating it via the index.
- **~8% used neither** — filename/directory listings, which already answer count/list/metadata
  questions (e.g. "how many 2022 10-Ks", "list the companies") with no content retrieval needed.

Only **~14% of all 1001 queries went truly outside the index** (PDF-only or filename-only).
Why codex routes around the CLI at all: its ~1 s `exec_command` yield backgrounds the
long-running `retriever query` (≈76% of its CLI calls get detached; see
`codex_yield_backgrounding_recall.md`), so it often prefers a Python `to_pandas()`/`search()`
call that returns inline. The index still does the work — which is why skill-codex keeps its
recall lift (0.503 → 0.526) despite the lower *CLI* attempt rate.

## Provenance

- baseline: `/tmp/agent_eval_corpus_base/` (`comparison.md` + per-run `report.{md,json}`)
- skill:    `/tmp/agent_eval_corpus_skill/`
- manifest: `/raid/retriever-sdg-v3/runs/agent_corpus_level_batch_4/agent_corpus_level_manifest.json`
- queries:  `/raid/retriever-sdg-v3/runs/agent_corpus_level_batch_4/queries.json`

Regenerate: `python3 build_report.py <run_dir> <run_dir> --manifest <manifest> --out <dir>`
(see `README.md` → *Cost & token accounting* for the pricing table and token convention).
