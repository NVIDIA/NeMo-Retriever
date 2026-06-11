# EVALUATE — two rows with NO test suite (unsupported in NRL 26.05)

The JTBD "functional tests" tab lists three EVALUATE user tasks. Only the first has a test
suite; the other two are marked **N/A** in the spec because the capability does not exist in
NeMo Retriever Library 26.05. This file documents that on purpose, so the coverage map has
no silent holes.

| EVALUATE user task | Priority | Spec success criteria | Suite |
|---|---|---|---|
| Evaluate **Retrieval** quality for customer datasets | P1 | operational pass (Recall@k / nDCG@k surfaced) | ✅ `evaluate_01_retrieval_quality/` |
| Evaluate **Extraction** quality for custom datasets | P1 | **N/A** — "currently unsupported in NRL 26.05" | ❌ none (this file) |
| Evaluate **Answer + citation** quality | P1 | **N/A** — "currently unsupported in NRL 26.05" | ❌ none (this file) |

## Why no suite for these two

- **Evaluate Extraction quality** — NRL 26.05 ships no harness that scores extraction
  output against a labeled extraction ground truth (e.g. per-field / per-cell / layout
  accuracy). There is no `retriever` subcommand whose contract is "grade extraction vs
  gold," so there is no correct subcommand for an agent to select — the functional
  success criterion is `N/A`.
- **Evaluate Answer + citation quality** — NRL is retrieval-only; it does not generate a
  final answer (answer generation is explicitly out of scope — see negative-test category
  (d), "Generate a final answer with citations…"). With no answer produced by the library,
  there is nothing for the library to self-grade for answer/citation quality, so the
  criterion is `N/A`. (Answer-quality grading is what the **performance-eval** track does
  at the *agent* layer via RAGAS — that is a different artifact, not a library capability.)

## When these become testable

If a future NRL release adds an extraction-eval harness or a generate-with-citations
runtime, add `evaluate_02_extraction_quality/` and `evaluate_03_answer_citation_quality/`
here following `CONVENTIONS.md`, and update the coverage index in
`functional_cases/README.md`. Until then, an agent asked to "evaluate extraction quality"
or "grade my answer + citations" should state the capability is **not supported in 26.05**
(this is the correct, in-scope behavior — it overlaps the negative-test discipline).
