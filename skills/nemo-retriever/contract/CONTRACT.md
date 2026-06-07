# retriever skill↔engine contract

`contract_version` (see `cli-contract.json`) is the semver the **skill** asserts
about the installed **engine**. Run `scripts/doctor.py` to verify the installed
`retriever` satisfies it.

The skill's one primitive is **`retriever retrieve <question>`** → `{ evidence, coverage }`.
The contract is defined around that primitive — not around the engine's CLI flag
surface. `query`/`verify` still ship but the skill does not depend on them;
they are documented under "Legacy" and are NOT gated by `doctor.py`.

## Files
- `cli-contract.json` — the gated surface: required subcommands, `retrieve`'s
  required + forbidden flags, `ingest`'s flags, and a `legacy` block for the
  ungated commands. `default_table_name` is the engine's table-name constant
  (operator config), not the skill name.
- `retrieve-result.schema.json` — the shape `retriever retrieve` emits and the
  skill reasons over: `evidence[]` (each with `text, source, locator, modality,
  fidelity, score, citation`) + `coverage`. This is THE contract the skill relies on.

## Versioning
- Bump **patch** for clarifications, **minor** for additive engine capabilities the
  skill can use, **major** when the engine changes something the skill relies on
  (a `retrieve` evidence/coverage field, the `retrieve` flag surface, or the gated
  primitive). A major bump means the skill must be updated in the same change.
- `doctor.py` fails if the installed engine no longer matches `cli-contract.json` /
  `retrieve-result.schema.json`.

## How drift gets caught
`doctor.py` runs on the skill's setup turn and in CI (`tests/test_contract.py`). It
performs a LIVE probe — ingest a tiny fixture, run `retrieve`, validate
`{evidence, coverage}` (including the `fidelity` enum) against
`retrieve-result.schema.json` — plus static `--help` checks: the required
subcommands exist, `retrieve` exposes its required flags, and `retrieve` does NOT
expose strategy knobs (`--content-types`, `--rerank`, …). Any divergence (a renamed
evidence field, a missing `fidelity`, a strategy knob leaking onto `retrieve`,
`--input-type` reappearing on `ingest`) fails loudly with a remediation hint.

## Legacy (not gated)
`query` and `verify` still exist for callers that want raw hits, but the `retriever`
skill routes everything through `retrieve`. If a future skill revision adopts `verify`
as a first-class move, promote it out of the `legacy` block and add a gated check + schema.

## Changelog
- **0.1.0** — initial contract for the skill-first `retriever`. Defined around the one
  primitive **`retriever retrieve <question>`** → `{evidence, coverage}` (see
  `retrieve-result.schema.json`): `doctor.py` live-probes `retrieve` and validates the
  shape (evidence + coverage, `fidelity` enum), and static-checks that the required
  subcommands (`ingest`, `retrieve`, `serve-models`) exist and that `retrieve` hides
  strategy knobs (`--content-types`, `--rerank`, …). `query`/`verify` ship but are
  ungated legacy.
