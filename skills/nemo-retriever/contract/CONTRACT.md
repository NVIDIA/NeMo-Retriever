# retriever skill↔engine contract

`contract_version` (see `cli-contract.json`) is the semver the **skill** asserts
about the installed **engine**. Maintainers can run `scripts/doctor.py` after a
CLI or evidence-schema change to verify the installed `retriever` satisfies it.

The skill's one primitive is **`retriever query <question> --format evidence --retrieval-mode hybrid`** →
`{ evidence, coverage }`. The skill opts into `--format evidence`
(fidelity-tagged evidence + coverage) and `--retrieval-mode hybrid` (vector+BM25) explicitly.
`query` also exposes `--rerank`, `--candidate-k`,
`--content-types`, `--page-dedup` (unused by the skill); the contract gates the skill's
invocation + result shape, not the full flag surface.

## Files
- `cli-contract.json` — the executable surface checked by `doctor.py`: required
  subcommands, required query/ingest flags, forbidden ingest flags, and the query
  result schema path.
- `query-result.schema.json` — the shape `retriever query --format evidence` emits and the
  skill reasons over: `evidence[]` (each with `text, source, locator, modality,
  fidelity, score, citation`) + `coverage`. This is THE contract the skill relies on.

## Versioning
- Bump **patch** for clarifications, **minor** for additive engine capabilities the
  skill can use, **major** when the engine changes something the skill relies on
  (a `query` evidence/coverage field, a required flag, or the gated primitive). A
  major bump means the skill must be updated in the same change.
- `doctor.py` fails if the installed engine no longer matches `cli-contract.json` /
  `query-result.schema.json`.

## What the doctor verifies
`doctor.py` checks that the required subcommands (`ingest`, `query`) and their
contracted local flags exist. It then performs a live probe: ingest a tiny
built-in document, invoke the public
`retriever query --format evidence` workflow, and validate `{evidence, coverage}`
(including the `fidelity` enum) against `query-result.schema.json`. Any
divergence (a renamed evidence field, a missing `fidelity`, a dropped `--format`,
or `--input-type` reappearing on ingest) fails loudly with a remediation hint.

## Changelog
- **1.0.0** — replace deprecated index/retrieval aliases with canonical
  `--index-mode hybrid` for ingest and `--retrieval-mode hybrid` for query; remove
  descriptive JSON fields that the doctor did not enforce.
- **0.1.0** — initial skill-first `{ evidence, coverage }` query contract
  (validated against `query-result.schema.json`). The gated subcommands are
  `ingest` and `query`; evidence output and hybrid retrieval are explicit opt-ins.
  `query` may expose extra knobs (`--rerank`, `--candidate-k`, …) — they're allowed but unused
  by the skill, so the contract gates the invocation + result shape, not the full flag surface.
