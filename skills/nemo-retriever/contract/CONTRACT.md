# retriever skill↔engine contract

`contract_version` (see `cli-contract.json`) is the semver the **skill** asserts
about the installed **engine**. Run `scripts/doctor.py` to verify the installed
`retriever` satisfies it.

The skill's one primitive is **`retriever query <question> --format evidence --hybrid`** →
`{ evidence, coverage }`. The `query` engine defaults are `--format hits` (a flat ranked
list) and vector-only (`--hybrid` off); the skill opts into `--format evidence`
(fidelity-tagged evidence + coverage) and `--hybrid` (vector+BM25) **explicitly**, so
plain `query` callers are unaffected. `query` *also* exposes `--rerank`, `--candidate-k`,
`--content-types`, `--page-dedup` (unused by the skill); the contract gates the skill's
invocation + result shape, not the full flag surface.

## Files
- `cli-contract.json` — the gated surface: required subcommands, `query`'s required
  flags + default format/hybrid, and `ingest`'s flags. `default_table_name` is the
  engine's table-name constant (operator config), not the skill name.
- `query-result.schema.json` — the shape `retriever query --format evidence` emits and the
  skill reasons over: `evidence[]` (each with `text, source, locator, modality,
  fidelity, score, citation`) + `coverage`. This is THE contract the skill relies on.

## Versioning
- Bump **patch** for clarifications, **minor** for additive engine capabilities the
  skill can use, **major** when the engine changes something the skill relies on
  (a `query` evidence/coverage field, the default `--format`/`--hybrid` behavior, or
  the gated primitive). A major bump means the skill must be updated in the same change.
- `doctor.py` fails if the installed engine no longer matches `cli-contract.json` /
  `query-result.schema.json`.

## How drift gets caught
`doctor.py` runs on the skill's setup turn. It
performs a LIVE probe — ingest a tiny built-in document, run `retriever query --format evidence`,
validate `{evidence, coverage}` (including the `fidelity` enum)
against `query-result.schema.json` — plus static `--help` checks: the required
subcommands (`ingest`, `query`) exist and `query` exposes its required
flags (`--top-k`, `--hybrid`, `--format`). Any divergence (a renamed evidence field, a
missing `fidelity`, a dropped `--format`, `--input-type` reappearing on `ingest`) fails
loudly with a remediation hint.

## Changelog
- **0.1.0** — skill-first contract built around **`retriever query --format evidence --hybrid`**
  → `{evidence, coverage}` (validated against `query-result.schema.json`). The gated
  subcommands are `ingest` and `query`; `query`'s engine defaults are `--format hits` and
  vector-only, and the skill passes `--format evidence`/`--hybrid` explicitly.
  `query` may expose extra knobs (`--rerank`, `--candidate-k`, …) — they're allowed but unused
  by the skill, so the contract gates the invocation + result shape, not the full flag surface.
