# Agent Instructions — NeMo Retriever

These standards apply to AI coding agents working in this repository (Cursor, Claude Code, and compatible hosts).

## Project Overview

NVIDIA NeMo Retriever Library (NRL) is the multimodal extraction and retrieval library formerly known in docs as NV-Ingest. Published customer documentation lives primarily under `docs/docs/extraction/` and builds with MkDocs (`docs/mkdocs.yml`). Library and Helm sources live under `nemo_retriever/`.

## Repository Map

| Path | Purpose |
|------|---------|
| `docs/docs/extraction/` | Published NRL extraction documentation (MkDocs) |
| `docs/mkdocs.yml` | Nav, redirects, and MkDocs config |
| `nemo_retriever/` | Library source, CLI, Helm chart, and in-repo README examples |
| `nemo_retriever/tests/` | Library tests |
| `.github/workflows/` | CI, including NRL docs publish workflows |

## Documentation

- Treat `docs/docs/extraction/` and related MkDocs pages as the source of truth for user-facing NRL documentation. Follow [`docs/AGENTS.md`](docs/AGENTS.md).
- Before completing a code change, determine whether it changes a **user-visible** surface. This includes a public API, CLI, configuration, Helm values or defaults, workflow, error message or error contract, supported file type, or other supported product behavior.
- When it does and the host supports subagents, start a documentation authoring subagent while the primary agent continues the implementation. Direct it to read `docs/AGENTS.md`, update the affected docs, and run validation. Give it the changed sources and user-visible impact.
- Reconcile the authoring subagent's documentation changes and validation evidence before completing the implementation. Include the required documentation in the same change when the repository workflow allows a combined PR. If policy requires a **docs-only** follow-up PR, open that PR in the same task and link it from the code PR.
- If the host cannot run subagents, read `docs/AGENTS.md` in the primary task, complete the documentation work, and run its documented validation. Do not omit required documentation because parallel execution is unavailable.
- Do not document defaults or behavior that `main` does not have yet.
- Documentation PRs that change published NRL prose must stay docs-scoped. Do not change `nemo_retriever/src/**`, tests, Helm chart behavior, lockfiles, or runtime CI env on a docs PR unless the user explicitly requests eng work.
- Verified product surfaces that often need docs updates: Python `create_ingestor` / `GraphIngestor` APIs, `retriever` CLI, Helm chart README and values, support matrix and NIM defaults, authentication and environment variables, error and troubleshoot guidance, and release notes.

### NVIDIA DORI Routing

Select the documentation path from current host capabilities.
Do not ask the user to classify themselves or store repository-scoped identity
state during a normal documentation task.

1. Check whether the current agent exposes `dori_handle` or `dori_route` and
   `dori_collections`.
   If the user explicitly asks not to use DORI, use the
   [Writing Style Guide](docs/AGENTS.md#writing-style-guide) instead.
2. When those tools are available, list the installed collections.
   - If a collection source contains `tech-docs/skill-library`, use DORI for
     task routing.
   - If the collection is missing, inaccessible, or cannot be verified,
     continue with the
     [Writing Style Guide](docs/AGENTS.md#writing-style-guide).
3. When the DORI tools are unavailable, continue with the Writing Style Guide.
   Do not inspect a shell-visible CLI, install software, or configure the host
   during a normal documentation task.
4. Use [NVIDIA DORI Setup](docs/DORI_SETUP.md) only when the user explicitly
   asks to install or configure DORI.

Capability detection does not approve installation or host configuration.
DORI unavailability must not block documentation work.

## Engineering Guardrails

- Prefer small, focused diffs that match existing style.
- Do not invent APIs, CLI flags, Helm keys, or defaults. Verify against checked-in source or tests.
- Never commit secrets, API keys, or credentials.
- Do not add lint, hooks, or CI from agent guidance alone. Those require a separately reviewed repository change.
- Do not create or modify `CLAUDE.md` as part of documentation-agent setup.

## Validation Shortcuts

| Change type | Validation |
|-------------|------------|
| Docs under `docs/` | From `docs/`: `python -m mkdocs build --strict --config-file mkdocs.yml` when the environment supports it |
| Library code | Run the targeted tests that cover the changed modules |
| Docs-only PR scope | `git diff --name-only upstream/main...HEAD` (or `origin/main...HEAD`) and confirm no runtime/out-of-scope paths |
