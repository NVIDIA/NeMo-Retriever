# retriever skill↔engine contract

`contract_version` (see `cli-contract.json`) is the semver the **skill** asserts
about the installed **engine**. Run `scripts/doctor.py` to verify the installed
`retriever` satisfies it.

The skill's one primitive is **`retriever query <question> --format evidence --hybrid`** →
`{ evidence, coverage }`. `query`'s **engine defaults are unchanged** from legacy
(`--format hits` = flat ranked list, `--hybrid` off = vector-only); the skill opts into
`--format evidence` (fidelity-tagged evidence + coverage) and `--hybrid` (vector+BM25)
**explicitly**, so existing `query` callers are unaffected. `query` *also* exposes
`--rerank`, `--candidate-k`, `--content-types`, `--page-dedup` (unused by the skill); the
contract gates the skill's invocation + result shape, not the full flag surface. `verify`
still ships but the skill does not depend on it (Legacy, not gated).

## Files
- `cli-contract.json` — the gated surface: required subcommands, `query`'s required
  flags + default format/hybrid, `ingest`'s flags, and a `legacy` block for the
  ungated commands. `default_table_name` is the engine's table-name constant
  (operator config), not the skill name.
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
`doctor.py` runs on the skill's setup turn and in CI (`tests/test_contract.py`). It
performs a LIVE probe — ingest a tiny fixture, run `retriever query --format evidence`,
validate `{evidence, coverage}` (including the `fidelity` enum)
against `query-result.schema.json` — plus static `--help` checks: the required
subcommands (`ingest`, `query`, `serve-models`) exist and `query` exposes its required
flags (`--top-k`, `--hybrid`, `--format`). Any divergence (a renamed evidence field, a
missing `fidelity`, a dropped `--format`, `--input-type` reappearing on `ingest`) fails
loudly with a remediation hint.

## Legacy (not gated)
`verify` still ships for callers that want claim-evidence lookups, but the `retriever`
skill routes everything through `query`. If a future skill revision adopts `verify` as a
first-class move, promote it out of the `legacy` block and add a gated check + schema.

## Changelog
- **0.2.0** — consolidated the skill primitive onto **`retriever query --format evidence --hybrid`**.
  Removed the separate `retrieve` subcommand; its answer-ready `{evidence, coverage}` output is
  available via the **opt-in** `--format evidence` flag (with `--hybrid` for vector+BM25). `query`'s
  engine defaults are **unchanged** (`--format hits` flat list, vector-only) so existing callers are
  unaffected; the skill passes the flags explicitly. `doctor.py` now live-probes `query --format
  evidence` and validates
  against `query-result.schema.json` (renamed from `retrieve-result.schema.json`). The
  forbidden-strategy-knob check is dropped — `query` is both the skill primitive and the
  power-user tool, so knobs like `--rerank`/`--candidate-k` are allowed (just unused by
  the skill). **Breaking** vs 0.1.0: the `retrieve` subcommand no longer exists.
- **0.1.0** — initial skill-first contract, defined around `retriever retrieve <question>`
  → `{evidence, coverage}` with a knob-free primitive (strategy knobs forbidden on
  `retrieve`). Superseded by 0.2.0.
