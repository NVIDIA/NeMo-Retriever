# Functional prompts → UC-1…UC-8 coverage mapping

Maps the **`functional_corpus_variants`** track of `agent_corpus_level_batch_4`
(209 prompts = **35 distinct task-bases** × ~6 phrasing variants each; 203 single-turn +
6 stateful) onto the eight platform use-cases (UC-1…UC-8).

The manifest encodes an internal **"Job" taxonomy** that maps almost 1:1 onto the UCs:

| Internal job | Meaning | Primary UC |
|---|---|---|
| **Job B** | Bulk PDF ingest: `.files().extract().dedup().embed().vdb_upload()` | UC-4 |
| **Job I** | Multimodal extract/ingest (OCR, charts, page-elements, audio/ASR) | UC-4 (+UC-3) |
| **Job E** | Structural / metadata / semantic-scan query over the index | UC-5 / UC-8 |
| **Job F** | Multi-hop chained retrieval | UC-5c |

Each base ships ×6 **phrasing variants** (v1–v6: imperative / casual / dev-voice / synonym
triggers). That variant axis is itself a cross-cutting **invocation-robustness** test — it
checks the skill fires on diverse phrasings of the *same* capability, not new capabilities.

---

## Coverage summary

| UC | Capability | Coverage | Bases (≈ prompts) |
|---|---|---|---|
| **UC-1** Discover | Find retriever in plugin list / catalog / onboarding; build.nvidia.com as NIM source | ❌ **None** | — (s-family invocation-routing is tangential) |
| **UC-2** Set up | (a) hosted NIM via API key (b) HF on local GPU (c) local NIM containers | 🟡 **Incidental** | `e04` (online endpoint ≈ 2a); `i01` ("setup if needed") |
| **UC-3** Extract a doc | Single doc → structured output (JSON/MD/HTML) w/ tables/charts/lineage; agentic/schema-guided | 🟡 **Modalities only** | `f02` `f05` `i05` `i06` `i07` `i08` (≈36) |
| **UC-4** Ingest a folder | Bulk async resumable idempotent extract→dedupe→embed→index; multimodal; right structure | ✅ **Core / heaviest** | `i01 i02 i05–i09 i11 i12` `e04 e06` `f02 f05` `m01–m04` `s01 s03` (≈114) |
| **UC-5** Ask questions | (a–e) single/multi-source/multi-hop/aggregation/comparative + citations; planning depth | ✅ **Heavy** | `q05 q07 q12–q16` `d04 d06–d10` `p02 p03` `e08` `m01–m04` `s06` (≈110) |
| **UC-6** Evaluate accuracy | Extraction/retrieval/answer benchmarks (F1, TEDS, Recall@5, nDCG, RAGAS); CI | ❌ **None** | — (`e08`/`p03` emit comparison signals but aren't benchmark eval) |
| **UC-7** Integrate as KB / memory | Persist across sessions, runtime write-back, multi-tenant, memory typing/lifecycle | 🟡 **Persistence only** | `e06` `i11` + cold-start in `m01–m04` (≈42 partial) |
| **UC-8** Query KB structurally | (a) list/filter by metadata (b) full content by ID/filename/path + neighbors | 🟡 **(a) yes, (b) no** | `d04 d06 d10` `q07 q12 q13 q14` `s06` (≈48) |

Legend: ✅ well-covered · 🟡 partial · ❌ not covered.

**Shape of the suite:** ~80% of prompts exercise **UC-4 (ingest)** and **UC-5 (Q&A)** — the
runtime read/write loop. UC-8(a) structural query is solidly covered. UC-3 covers extraction
*modalities* but not the single-doc-structured-output framing. **UC-1, UC-2, UC-6 are
essentially untested**, and UC-7's "agentic memory" half and UC-8(b) content-fetch are gaps.

---

## Per-UC detail

### UC-1 — Discover · ❌ not covered
No prompt asks the agent to locate NeMo Retriever in a plugin/model catalog or onboarding
flow, or to identify build.nvidia.com as a NIM source. These are platform-surface tasks; the
functional track is entirely runtime corpus work. The **`s` (slash-trigger)** family is the
nearest neighbor — it verifies the router *fires* nemo-retriever deterministically on
`/`-invocation — but that's invocation plumbing, not discovery.

### UC-2 — Set up the retriever · 🟡 incidental
- **`e04`** uses `create_ingestor(run_mode='online', base_url=…)` — a **hosted/online endpoint**
  path, the closest analog to **UC-2(a)** (connect to hosted NIM).
- **`i01`** says "Run Step 1-3 setup if needed" — setup is a precondition, not the graded act.
- **No dedicated tests** for (a) API-key-only hosted NIM, (b) loading embed+rerank from HF onto
  a local GPU, or (c) deploying NIM containers locally. Setup is assumed, not evaluated.

### UC-3 — Extract a document · 🟡 extraction modalities, not the UC-3 framing
Extraction *capabilities* are exercised, but always inside a **folder-ingest** framing rather
than "submit one doc → get agent-ready structured JSON/MD/HTML with lineage":
- **`f02`** DOCX extraction · **`f05`** HTML via `.extract_html()`
- **`i05`/`i06`** scanned-form OCR + **table** extraction (`extract_tables`, nemotron-ocr)
- **`i07`** **chart** extraction (`extract_charts`, nemotron-graphic-elements-v1)
- **`i08`** mixed-format doc-image OCR
**Gaps:** no single-document structured-output-with-lineage test; no **schema-guided / agentic
extraction with self-correction** depth check.

### UC-4 — Ingest a folder · ✅ core, the heaviest-tested UC
- **Bulk pipeline (Job B):** `i01`, `i02` (`.files().extract().dedup().embed().vdb_upload()`).
- **Multimodal (Job I):** `i05/i06` OCR forms, `i07` charts, `i08` doc-images, `i09` **audio**
  (parakeet ASR, `[asr]` extra) — lands each in the index. → "handle multimodal (image/audio)".
- **Idempotent / resumable:** `i11` dedup-incremental ("only new files re-chunked"); **`e06`**
  SIGKILL-mid-ingest then resume with dedup (stateful) → "resumable, idempotent".
- **Async / streaming:** `e04` OnlineIngestor with mid-run progress → "bulk, async".
- **Right structure / auto-routing:** `i12` walks `test-data/`, routes PDFs→Job B,
  images/audio→Job I, into per-corpus or consolidated tables → "land each source type in the
  right structure".
- **Source types:** `f02` DOCX, `f05` HTML.
- **Cold-start ingest** as the first half of every `m0x`.
**Gaps inside UC-4:** **structured/SQL/tabular via GSF** and **web-mediated** sources are not
tested (HTML in `f05` is local files, not fetched from the web; no SQL/Spider2-style source).

### UC-5 — Ask questions of the corpus · ✅ heavily covered, all sub-modes present
- **(a) single-source Q&A:** `m02` (Apple-only supply-chain), `d07` (scoped semantic scan).
- **(b) multi-source synthesis:** `q05` (cross-doc risk aggregation), `f02`/`f05` (multi-doc),
  `m01` (per-company AI synthesis), `d10`.
- **(c) multi-hop / chained:** `q15` (acquisition→revenue), `q16` (revenue-peak→R&D), `d08`
  (dividend-rank→R&D), `m03` (YoY ranking), `p02` (orchestrated decomposition).
- **(d) aggregation (count/list/sum):** `q07` `q12` `q13` `q14` `d04` `s06` (counts, dedup
  lists, grouped tallies), `m03` (computed YoY).
- **(e) comparative:** `e08` (hybrid vs dense retrieval comparison); `m03` cross-company ranking.
- **Structured citations:** required by `f02`/`f05` (per-source), `q15` (cite the 2017 10-K),
  `m01`.
- **Depth — planning / decomposition / adaptive (the `p` family):** `p02` agent-orchestrated
  **query decomposition** (RET-8, "library has no native decomposer"); `p03` **adaptive
  strategy** — dense→BM25 **score-based fallback** (RET-9).
**Possible thin spot:** UC-5(e) is mostly *method*-comparative (`e08`) rather than
*content*-comparative ("compare company A vs B on X") — lighter than the other sub-modes.

### UC-6 — Evaluate accuracy · ❌ not covered
No prompt tasks the agent with running OmniDocBench/OCRBench/RD-TableBench (extraction),
BEIR/BRIGHT/ViDoRe/Spider2/BirdSQL (retrieval), or RAGAS/answer-citation quality — nor with
F1/TEDS/Recall@5/nDCG@10 or CI wiring. (`e08` hybrid-vs-dense and `p03` score-fallback emit
*comparison* signals, but that's runtime strategy selection, not benchmark measurement.)
*Note: the `agent_eval` harness in this repo is itself the UC-6 surface — but the prompts
don't ask the agent to evaluate.*

### UC-7 — Integrate as agent KB / memory · 🟡 persistence yes, agentic-memory no
- **Persists across turns + no re-ingestion:** `i11` incremental dedup (KB survives, only new
  docs added); **`e06`** idempotent resume after interruption; `m01–m04` **cold-start detect
  "KB missing"** (the agent reasons about whether the KB already exists).
**Gaps:** runtime **write-back** (events / tool-calls / outcomes), **multi-tenant isolation**,
and the depth check — **memory typing** (episodic/semantic/procedural/KG) and **lifecycle**
(capture→classify→consolidate→promote→expire) and **multi-scope** (user/project/org) — are
not exercised. This UC is tested as *durable corpus*, not as *agentic memory*.

### UC-8 — Query the KB structurally · 🟡 (a) covered, (b) not
- **(a) list/filter by metadata:** `q07` (list filenames+companies by year=2022), `q12` (count
  unique sources), `q13` (dedup company list), `q14` (filter+count), `d04` (per-company source
  count via `source LIKE`), `d06`/`d10` (semantic-then-list sources), `s06` (filtered count via
  slash). Strong coverage of metadata listing/filtering/counting.
**Gaps:** UC-8(b) — **return the full content of a doc by ID/filename/path**, or a **section by
chunk-ID + neighbors** — has no dedicated prompt. The depth check (**filter by memory
type/scope**) is also untested (follows from the UC-7 memory-typing gap).

---

## Full base → UC table

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

---

## Gaps & recommendations

1. **UC-1 (Discover)** and **UC-6 (Evaluate)** — zero coverage. If they matter, they need new
   prompt families: a discovery/onboarding surface (catalog/plugin-list/build.nvidia.com), and
   a benchmark-eval surface (run ViDoRe/OmniDocBench/RAGAS, report metrics). UC-6 is partly
   served *out-of-band* by this `agent_eval` harness, but no prompt tasks the agent with it.
2. **UC-2 (Setup)** — only the online-endpoint path (`e04`) is touched. Add explicit tests for
   the three setup modes (hosted-API-key / HF-on-GPU / local-NIM-container) if setup is in scope.
3. **UC-3 depth** — extraction modalities are covered but not "single doc → structured
   agent-ready output + lineage" nor **schema-guided / agentic extraction with self-correction**.
4. **UC-4 sub-gaps** — **structured (SQL/tabular via GSF)** and **web-mediated** sources are
   untested; HTML here is local, not web-fetched.
5. **UC-5(e)** — comparative queries are mostly *method*-comparative (`e08`); consider
   *content*-comparative prompts ("compare A vs B on metric X with citations").
6. **UC-7 (agentic memory)** — the durable-corpus half is covered (`e06`, `i11`, cold-start);
   the **memory** half (runtime write-back, multi-tenant isolation, memory typing/lifecycle,
   multi-scope) is entirely a gap. The 6 stateful `e06` tests are the only multi-turn
   persistence probes and are currently deferred in the runner.
7. **UC-8(b)** — no "fetch full content / section-by-chunk-ID + neighbors by doc ID" prompt;
   and filter-by-memory-type/scope follows the UC-7 typing gap.

*Source: `agent_corpus_level_manifest.json` (`functional_corpus_variants` track). Mapping by
task-base; each base = ~6 phrasing variants exercising the same capability.*
