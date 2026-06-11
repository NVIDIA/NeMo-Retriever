# Functional eval — findings (claude vs codex, batch4 `functional_corpus_variants`)

Analysis of the full pass/fail run over the **203 single-turn** functional prompts (the 6
stateful tests are deferred). Companion to the raw merged report
`/raid/agent_eval_func/functional_report_full.md`. **Headline: the raw pass-rates are
deflated by harness/grading artifacts; the artifact-adjusted rates are ~82% (claude) and
~77% (codex), and the two agents are much closer than the raw numbers suggest.**

## Raw results

| agent | model | PASS | gradable | pass-rate | infra timeouts (excluded) |
|---|---|---|---|---|---|
| claude | claude-opus-4-7 | 134 | 177 | **76%** | 26 |
| codex | gpt-5.5 | 128 | 195 | **66%** | 8 |

| functional_type | claude | codex |
|---|---|---|
| ingest | 46/54 (85%) | 44/53 (83%) |
| ingest_plus_answer | 6/8 (75%) † | 9/23 (39%) |
| retrieval_answer | 82/115 (71%) | 75/119 (63%) |

† claude's `ingest_plus_answer` sample is only **8 gradable** — 16 of 24 timed out on the
in-turn financebench ingest — so that cell is low-confidence.

## Update — e04 + i12 fixes applied

**e04 rubric — FIXED (re-graded, no re-run needed).** The ingest rubric now (a) drops the
unobservable `validation_signal` for ingest-only tests and (b) treats a count-bearing ingest
confirmation as satisfying any `expected_output_shape` clause that asks for streaming/per-doc
progress markers (the final message structurally cannot replay mid-run streaming; ingestion is
proven by the rows>0 gate). After re-grading, **all gradable e04 now pass** for both agents
(claude: 2 pass + 4 infra timeouts; codex: 5 pass + 1 timeout). New totals:

| agent | before | after e04 fix |
|---|---|---|
| claude | 134/177 (76%) | **136/177 (76.8%)** |
| codex | 128/195 (66%) | **132/195 (67.7%)** (+4) |

**i12 mount bug — FIXED in code, but needs a RE-RUN (not yet done).** `extract_functional`
now maps a bare `test-data` mention → mount the whole tree, and `functional_queries.json` is
regenerated (i12 `corpus_refs=['test-data']`). But the existing i12 *runs* executed with no
corpus mounted, so re-grading still fails them (claude 6 / codex 5). Fixing i12 for real means
**re-running the 12 i12 queries**, each of which ingests **all 1.7 GB of `test-data`** (every
modality incl. the large vidorev3 PDF corpora) within one turn — the heaviest query type, and
likely to hit the same in-turn-ingest timeouts. **Decision pending:** re-run i12 (heavy, may
mostly time out) vs. mark i12 as a known harness-limited case.

## Failure taxonomy

Every logic-FAIL (claude 43, codex 67) bucketed into **artifact** (grading/harness, not a
real agent failure) vs **real** signal:

| bucket | kind | claude | codex | notes |
|---|---|---|---|---|
| `i12` corpus never mounted | **artifact (harness bug)** | 5 | 4 | `corpus_refs=[]`; `test-data/` never mounted → agent correctly says "missing" |
| `e04` streaming progress markers | **artifact (unobservable)** | 2 | 4 | rubric wants mid-run "10/368 done" in the *final* message |
| output-shape pedantry (`q13`/`d04`/…) | **artifact (mostly)** | 5 | 16 | correct but *richer* answer (table+citations) vs "bare list/integer" |
| codex KB-not-persisted (`m*`) | **real (codex) — answer OK, KB not built** | 1 | 7 | judge=PASS but LanceDB table absent in workdir (`rows=None`) |
| `p02`/`p03` capability gap | **real** | 10 | 6 | query decomposition / adaptive dense→BM25 fallback not implemented |
| `q12` count mismatch | **real** | 5 | 6 | answered 366/270/275 instead of 368 unique docs |
| `d06`/`d07` semantic-recall miss | **real** | 9 | 5 | expected companies (MSFT/Walmart) omitted from semantic-scan list |
| `q07` filename-pattern | **real** | 0 | 2 | included `*_annualreport.pdf` / wrong-year against a `*_2022_10K.pdf` filter |
| `s01`/`s06` slash routing | **real-ish** | 2 | 4 | clarified/refused instead of firing the ingest on slash args |
| `q14` count method/shape | mixed | 1 | 3 | filename-count vs content-count + shape |
| other (`e08`,`i09`,`i11`,`f02`,`f05`,`m*`) | mixed | 3 | 10 | one-offs (e.g. `e08` missing score-deltas in the comparison table) |

### Artifacts (do not count against the skill)

1. **`i12` — harness mount bug (my bug).** The `i12` prompts say "walk `test-data/`", but the
   extractor (`extract_functional._corpus_refs`) only captures `test-data/<subpath>`, not the
   bare directory, so `corpus_refs` came back **empty** and nothing was mounted. Both agents
   correctly reported the directory missing → **9 false fails**. Fix: when a prompt references
   `test-data/` with no parseable subpath, mount the whole tree.
2. **`e04` — streaming progress is structurally unobservable.** The OnlineIngestor rubric
   requires mid-run per-doc progress markers ("10/368 done"); those stream *during* the run and
   cannot appear in the *final* message we grade. Same class as the tracer-log artifact fixed in
   the smoke. Fix: drop the mid-run-observability clause from the `e04` rubric (the artifact
   facts already prove ingestion).
3. **Output-shape pedantry.** `q13`/`d04` etc.: the agent returns a **correct** answer with
   extra structure (a table with citations, or a count plus explanation) and the judge fails it
   for not being a "simple deduplicated list" / "single integer." These are mostly correct
   answers penalized on format — heaviest on codex (16). Partial artifact (some genuinely ignore
   a stated output contract; most are over-strict).

### Real findings (genuine signal)

1. **Query-planning / adaptive-retrieval capability gap (`p02`/`p03`, 16+ fails).** These tests
   deliberately probe capabilities the library lacks natively — explicit query **decomposition**
   (`p02`) and **adaptive dense→BM25 fallback** with a score threshold (`p03`). Both agents
   mostly don't implement the skill-level workaround (claude fails 10, codex 6). **The skill has
   no query-planning / adaptive-strategy scaffolding** — a concrete enhancement target.
2. **codex doesn't reliably persist the KB (`m*`, 7 fails).** codex produces a *correct answer*
   (judge=PASS) but the LanceDB table is absent from the workdir (`rows=None`), so the ingest
   gate fails. Same root cause as its timeouts: codex's 1-second `exec_command` yield window
   backgrounds the long `retriever ingest`, so the cold-start ingest never finalizes even though
   it answers from a transient/partial state. **codex-specific orchestration weakness with
   long-running foreground commands** (the original skill_eval finding, reconfirmed).
3. **Exact-count queries are unreliable (`q12`/`q14`, ~11 fails).** Agents return 366/270/275
   instead of 368 — counting 10-Ks only, or unique embedded docs, vs the ground-truth file
   count. Real accuracy gap on aggregation/metadata-count queries (UC-5d / UC-8a).
4. **Semantic-scan recall misses (`d06`/`d07`, ~14 fails).** Semantic "list documents that
   mention X" omits companies the validation signal expects (Microsoft, Walmart). Real
   recall/coverage gap on semantic-filter-then-list queries.

## Timeouts (infra, excluded from pass-rate)

| agent | n | by base |
|---|---|---|
| claude | 26 | `m02`×6, `m03`×6, `s01`×4, `e04`×4, `m01`×3, `m04` `i11` `i12` |
| codex | 8 | `i01`×3, `i02` `i11` `i12` `m03` `e04` |

All are **financebench in-turn ingest** (368 PDFs ingested + queried within one turn, 60-min
cap). The asymmetry is informative: **claude times out *more* (26)** because it correctly waits
on the heavy ingest until the wall; **codex times out *less* (8)** because it backgrounds the
ingest and "finishes" — but then fails the KB-persistence gate instead (see real-finding #2).
Two faces of the same long-command-orchestration issue.

## Artifact-adjusted view

Excluding the `i12` harness bug (non-result) and crediting the `e04` + output-shape artifacts
as passes:

| agent | raw | adjusted ≈ | drivers of the gap |
|---|---|---|---|
| claude | 134/177 = 76% | **~141/172 ≈ 82%** | +e04(2) +shape(5); −i12(5 excluded) |
| codex | 128/195 = 66% | **~148/191 ≈ 77%** | +e04(4) +shape(16); −i12(4 excluded) |

If codex's "answer-correct-but-KB-not-persisted" cases are also credited on *answer quality*
(they were judge=PASS), codex rises another ~7 to **~81%** — i.e. **answer quality is near
parity; codex's real deficit is durable ingestion, not retrieval/answering.**

## What's solid
- **Ingest (UC-4 core): ~84% both agents, at parity** — bulk + multimodal (OCR/charts/audio)
  in-turn ingest works.
- **Multi-modal in-turn ingest confirmed** (from the smoke + run): chartqa images, librispeech
  audio, financebench PDFs all produce real LanceDB rows.
- **claude ≥ codex on retrieval/answering** once artifacts are removed, with codex's headline
  gap explained almost entirely by output-shape grading + KB-persistence, not answer quality.

## Recommended next steps (cheap — grading is concurrent + cached)
1. **Fix the `i12` mount** (bare-`test-data/` → mount whole tree) and **soften the `e04`
   rubric** (drop mid-run-observability), then **re-grade** — cache reuses unaffected verdicts,
   so only ~15 re-run (minutes). Cleaner, fairer pass-rate.
2. **Recover the timeouts** — re-run the 26+8 financebench in-turn-ingest queries with a longer
   per-query timeout (or a prebuilt shared financebench index for the `m*`/`i01` ingest-type
   queries) to fill in claude's thin `ingest_plus_answer` cell and stabilize the comparison.
3. (Product) Treat **query-planning/adaptive-retrieval** (`p02`/`p03`) and **codex long-command
   persistence** as the two real capability gaps worth addressing in the skill.

## UC coverage mapping (these prompts → UC-1…UC-8)

The 35 task-bases (×~6 phrasing variants) use an internal **Job taxonomy** — Job B = bulk PDF
ingest, Job I = multimodal extract/ingest, Job E = structural/semantic query, Job F = multi-hop
— that maps cleanly onto the platform use-cases.

| UC | Capability | Coverage | Bases (≈ prompts) |
|---|---|---|---|
| **UC-1** Discover | retriever in plugin/catalog/onboarding; build.nvidia.com as NIM source | ❌ none | — (s-family invocation is tangential) |
| **UC-2** Set up | (a) hosted NIM via API key (b) HF on GPU (c) local NIM containers | 🟡 incidental | `e04` (online endpoint ≈2a); `i01` ("setup if needed") |
| **UC-3** Extract a doc | single doc → structured JSON/MD/HTML w/ tables/charts/lineage; agentic | 🟡 modalities only | `f02 f05 i05 i06 i07 i08` (≈36) |
| **UC-4** Ingest a folder | bulk async resumable idempotent extract→dedupe→embed→index; multimodal | ✅ core/heaviest | `i01 i02 i05–i09 i11 i12` `e04 e06` `f02 f05` `m01–m04` `s01 s03` (≈114) |
| **UC-5** Ask questions | (a–e) single/multi-source/multi-hop/aggregation/comparative + citations; planning | ✅ heavy | `q05 q07 q12–q16` `d04 d06–d10` `p02 p03` `e08` `m01–m04` `s06` (≈110) |
| **UC-6** Evaluate accuracy | extraction/retrieval/answer benchmarks (F1, TEDS, Recall, nDCG, RAGAS); CI | ❌ none | — (`e08`/`p03` emit comparison signals, not benchmark eval) |
| **UC-7** Integrate as KB / memory | persist across sessions, write-back, multi-tenant, memory typing | 🟡 persistence only | `e06` `i11` + cold-start in `m01–m04` (≈42 partial) |
| **UC-8** Query KB structurally | (a) list/filter by metadata (b) full content by ID/filename/path + neighbors | 🟡 (a) yes, (b) no | `d04 d06 d10` `q07 q12 q13 q14` `s06` (≈48) |

Legend: ✅ well-covered · 🟡 partial · ❌ not covered. ~80% of prompts exercise **UC-4 (ingest)**
and **UC-5 (Q&A)**; **UC-1, UC-2, UC-6 are essentially untested**; UC-7's agentic-memory half
and UC-8(b) content-fetch are gaps. Full per-UC reasoning is in `functional_uc_mapping.md`.

### Full base → UC table

| Base | ×n | functional_type | Job | Capability (one-liner) | Maps to |
|---|---|---|---|---|---|
| `i01` | 6 | ingest | B | bulk PDF ingest pipeline | **UC-4** |
| `i02` | 6 | ingest | B | bulk ingest, "build a KB" synonym | **UC-4** |
| `i05` | 6 | ingest | I | scanned-form OCR + tables | **UC-4**, UC-3 |
| `i06` | 6 | ingest | I | same, casual "load up" phrasing | **UC-4**, UC-3 |
| `i07` | 6 | ingest | I | chart extraction (graphic-elements) | **UC-4**, UC-3 |
| `i08` | 6 | ingest | I | doc-image OCR | **UC-4**, UC-3 |
| `i09` | 6 | ingest | I | audio ingest (parakeet ASR) | **UC-4** (multimodal) |
| `i11` | 6 | ingest | B | idempotent incremental dedup | **UC-4**, UC-7 |
| `i12` | 6 | ingest | B+I | mixed-modality auto-routing → tables | **UC-4** (right structure) |
| `e04` | 6 | ingest | B | OnlineIngestor, streaming/async | **UC-4**, UC-2a |
| `e06` | 6 | **stateful** | B | SIGKILL → idempotent resume | **UC-4**, UC-7 |
| `f02` | 6 | retrieval_answer | B | DOCX folder ingest + multi-doc Q&A | UC-3, UC-4, **UC-5b** |
| `f05` | 6 | retrieval_answer | B | HTML extraction + multi-page synthesis | UC-3, UC-4, **UC-5b** |
| `m01` | 6 | ingest_plus_answer | B+E | cold-start ingest + per-company synthesis | **UC-4 + UC-5b**, UC-7 |
| `m02` | 6 | ingest_plus_answer | B+E | cold-start + scoped Apple supply-chain | **UC-4 + UC-5a** |
| `m03` | 6 | ingest_plus_answer | B+F | cold-start + YoY-growth ranking | **UC-4 + UC-5c/d** |
| `m04` | 6 | ingest_plus_answer | I+E | cold-start audio + semantic scan | **UC-4 + UC-5** |
| `q05` | 6 | retrieval_answer | E | year filter + risk aggregation/synthesis | **UC-5b/d** |
| `q07` | 6 | retrieval_answer | E | metadata-only list filenames+companies | **UC-8a**, UC-5d |
| `q12` | 6 | retrieval_answer | E | count unique source docs | **UC-5d**, UC-8a |
| `q13` | 6 | retrieval_answer | E | dedup company grouping | **UC-5d**, UC-8a |
| `q14` | 6 | retrieval_answer | E | metadata filter + count | **UC-5d**, UC-8a |
| `q15` | 6 | retrieval_answer | F | multi-hop acquisition→revenue | **UC-5c** |
| `q16` | 6 | retrieval_answer | F | multi-hop revenue-peak→R&D | **UC-5c** |
| `d04` | 6 | retrieval_answer | E | per-company count via `source LIKE` | **UC-8a**, UC-5d |
| `d06` | 6 | retrieval_answer | E | semantic + list source docs | **UC-8a/UC-5a** |
| `d07` | 6 | retrieval_answer | E | semantic scan (tariffs) | **UC-5a/b** |
| `d08` | 6 | retrieval_answer | F | multi-hop dividend-rank→R&D | **UC-5c** |
| `d10` | 6 | retrieval_answer | E | industry-classification filter→list | **UC-8a**, UC-5b |
| `p02` | 6 | retrieval_answer | F | query planning / decomposition (RET-8) | **UC-5 depth** |
| `p03` | 6 | retrieval_answer | E | adaptive dense→BM25 fallback (RET-9) | **UC-5 depth** |
| `e08` | 5 | retrieval_answer | E | hybrid vs dense comparison (RET-10) | **UC-5e**, ~UC-6b |
| `s01` | 6 | retrieval_answer | B | slash `ingest <path>` → Job B | **UC-4** (invocation) |
| `s03` | 6 | retrieval_answer | I | slash "build me a KB" → Job I | **UC-4** (invocation) |
| `s06` | 6 | retrieval_answer | E | slash `count` + filter → Job E | **UC-5d/UC-8a** (invocation) |

**Cross-reading findings × UC:** the real capability gaps land on **UC-5 depth** (`p02`/`p03` =
query-planning/adaptive) and **UC-4/UC-7** (codex KB-not-persisted = durable ingest); the
exact-count (`q12`) and semantic-recall (`d06`/`d07`) misses are **UC-5d / UC-8a**. The
artifacts cluster on **UC-4** (`i12` mount bug, `e04` streaming) and the **UC-8a/UC-5d**
structural-query output-shape pedantry.

## Provenance / repro
- Runs: `/raid/agent_eval_func/agenteval_func_{claude,codex}_20260603_010229_UTC/`
  (skill profile, all 203; claude GPUs 0–3, codex 4–7; per-query timeout 3600s).
- Grading: `eval_functional.py` (concurrent judge, incremental/resumable),
  judge = `nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5`, PASS/FAIL rubrics with the
  internal-signal-agnostic fix (ignore tracer/job/endpoint clauses; ingest rubric trusts the
  programmatic rows>0 gate).
- Raw report: `/raid/agent_eval_func/functional_report_full.md`.
- UC mapping of these prompts: `agent_eval/functional_uc_mapping.md`.
