# Plan — evaluating the functional (domain-less) batch4 prompts

## Background

When we extracted `agent_corpus_level_batch_4`, **209 of 1210 exported prompts had no
`domain`** and were skipped from the corpus-level runs. They are the
**`functional_corpus_variants`** track — behavioral/functional tests, not Q&A-with-gold-pages:

- **No `domain`** (so `domain → corpus` mount doesn't apply) and **no exact `answer`**.
- They reference their corpus **by path in the prompt** (`test-data/financebench/pdfs`,
  FUNSD, LibriSpeech, ChartQA, DocVQA, multiformat docx/html — all present under
  `/raid/retriever-sdg-v3/test-data/`).
- Pass/fail is defined by `expected_behavior` + `validation_signal` +
  `expected_output_shape` + `tests` + `contract`, keyed by `ground_truth_kind`:

| functional_type | n | ground_truth_kind | pass/fail basis |
|---|---|---|---|
| `retrieval_answer` | 119 | answer_reference | answer satisfies `validation_signal` |
| `ingest` | 60 | action_contract | LanceDB table built (rows>0) AND no Q&A answer |
| `ingest_plus_answer` | 24 | answer_reference | ingest happened AND answer correct |
| `stateful_orchestration` | 6 | stateful_contract | multi-turn: SIGKILL ingest ~50%, re-ingest, dedup resumes |

These are **skill-profile** tests (they assume the retriever ingest/query pipeline),
graded **pass/fail**, not recall@k.

## Decisions (confirmed)

- **Scope:** skill profile, **claude + codex**.
- **Stateful (6):** **deferred** — they need a multi-turn + mid-ingest-SIGKILL driver;
  build/run the **203 single-turn** tests first.
- **Grader:** **LLM PASS/FAIL rubric** (not 1–5) for answer-reference tests.

## Phases

**Phase 1 — Extract** (`extract_functional.py`)
`functional_queries.json` = `[{query_id, prompt, functional_type, corpus_refs}]`,
excluding stateful. `corpus_refs` parsed from the prompt's `test-data/...` paths (for
mounting). Pass criteria stay in the manifest (answer-free file, as for the recall runs).
Corpus root map: `test-data/` → `/raid/retriever-sdg-v3/test-data/`.

**Phase 2 — Runner: arbitrary-corpus mount**
Add a **functional mount mode** to `run_agent_eval.py`: symlink each prompt's referenced
`test-data/...` dir into the workdir preserving the relative path (so the prompt's literal
paths resolve), instead of `domain → corpus`. Skill profile, claude (GPUs 0–3) + codex
(4–7) via the existing `--gpu-list` pinning. Reuse one financebench index across the many
financebench prompts (don't re-ingest 368 PDFs per query). Verify OCR/ASR extras for the
image/audio subsets; separate "infra-fail" (can't ingest) from "logic-fail" in the report.

**Phase 3 — Pass/fail evaluator** (`eval_functional.py`)
Re-read manifest criteria; grade by `ground_truth_kind`:
- **answer_reference** → `LLMJudge` with a PASS/FAIL rubric vs `validation_signal` +
  `expected_output_shape` → `{pass, reason}`.
- **action_contract (ingest)** → programmatic: workdir LanceDB table with rows>0 AND
  output is an ingest confirmation (not a Q&A answer).
- **ingest_plus_answer** → both (artifact check AND answer grader).

**Phase 4 — Report** (`functional_report.md`)
Pass-rate overall + by `functional_type` + by agent, with failing IDs + reasons.
(No recall@k — these have no gold pages.)

## Risks
- **Multi-modal ingest** (FUNSD/DocVQA/ChartQA images, LibriSpeech audio) needs the
  retriever OCR/ASR extras; subsets that can't ingest fail at setup, not logic — report
  them separately.
- **Cost/time:** 203 × 2 agents, skill profile; financebench is a heavy 368-PDF ingest —
  share its index across financebench prompts.
- **Stateful driver** (deferred) is the one genuinely new capability (process injection +
  state assertions).

## Status
- [x] Investigation + plan
- [x] Phase 1 — extractor (`extract_functional.py`) → 203 queries (119 retrieval_answer /
  60 ingest / 24 ingest_plus_answer; 6 stateful excluded); all 7 corpus_refs resolve on
  disk; leak-free. Output: `functional_queries.json` in the batch4 run dir.
- [x] Phase 2 — runner (`run_functional.py`, sibling to run_agent_eval.py). Mounts each
  prompt's `corpus_refs` at literal `test-data/...` paths via `profiles.mount_corpus_refs`;
  skill profile; GPU pinning (`--gpu-list`); shared per-corpus prebuilt index for
  retrieval_answer (built once), in-turn ingest for ingest/ingest_plus_answer. Reuses the
  base AgentAdapter `run()`. **Dry-run verified** (368 financebench PDFs visible at literal
  path, skill copied, functional suffix appended, ingest-type has no prebuilt index).
  **Real GPU run still pending** (heavy: 203×2 agents + 368-PDF ingest).
- [x] Phase 3 — evaluator (`eval_functional.py`). Re-reads manifest criteria; routes by
  ground_truth_kind: answer_reference→LLM PASS/FAIL rubric (custom prompt via
  `LLMJudge._client.complete`, not the 1–5 reference template); action_contract→programmatic
  LanceDB rows>0 gate + rubric (ingest-not-Q&A); ingest_plus_answer→gate AND rubric. Caches
  verdicts. Unit-verified (parser, manifest loader, complete() signature).
- [x] Phase 4 — report (`functional_report.py`). Merges N per-agent functional_eval.json →
  `functional_report.md`: overall + by-type matrix + failing IDs, separating non-results
  (run_timeout/judge_error/...) from logic FAILs. End-to-end smoke-tested.
- [x] **Smoke run** (claude, 4 queries: financebench-query / docx-query / chartqa-ingest /
  librispeech-ingest+answer) → **4/4 PASS** after fixes. Validated the full Phase 2→3→4 chain
  on real artifacts incl. live judge + lancedb row count. Findings:
  - **financebench shared index = 1764s (~29min)** to build once; amortized across all 119
    retrieval_answer fb queries in the full run. **Pre-build it once and reuse** (don't rebuild
    per run dir): point the runner at a persisted index dir.
  - **Multi-modal in-turn ingest WORKS**: chartqa images → 30 rows, librispeech audio → 147
    rows (the agent's skill handled OCR/ASR). The OCR/ASR-extras risk did NOT materialize for
    the agent's own `retriever ingest`.
  - **docx/html shared-index pre-build FAILS** (`retriever ingest test-data/multiformat/docx`
    errored in 35s — pipeline is PDF/image/audio, not Office/HTML). f02/f05 (12 retrieval_answer
    queries) run with no index; the agent answered by reading the docx natively and PASSED — so
    those test native-fallback, not the retriever query path. Acceptable for v1; flag in report.
  - **Rubric fix (critical):** manifest `validation_signal`s mix observable answer/output-shape
    contract with UNOBSERVABLE internal signals ("Tracer logs job=B/I", "graphic-elements
    endpoint hit"). Grading strictly false-FAILed i07 + m04. Fixed both rubrics
    (`_RUBRIC_SYSTEM` + `_RUBRIC_SYSTEM_INGEST`) to IGNORE internal tracer/job/endpoint clauses
    and grade only observable content + the verified artifact facts. Without this, most of the
    ~84 ingest/ingest_plus tests would false-FAIL.
- [ ] **Full run** — execute run_functional.py for claude (GPUs 0–3) + codex (4–7) over all 203,
  then eval_functional.py on each, then functional_report.py merging both. Watch for OOM. The
  financebench shared index dominates wall-clock — build/persist it once first.
- [ ] (later) stateful driver (the 6 stateful_orchestration tests)
