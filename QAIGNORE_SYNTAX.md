# `.qaignore` Syntax Reference

The `.qaignore` file controls what the QA automation agent **skips** during analysis, test generation, and validation. It lives at the repository root and uses a line-oriented, plain-text format inspired by `.gitignore`.

---

## General Rules

| Rule | Detail |
|------|--------|
| **One pattern per line** | Each non-blank, non-comment line is a single ignore rule. |
| **Comments** | Lines starting with `#` that are **not** a valid PR pattern (see §4) are treated as comments and ignored by the parser. Use them freely for documentation. |
| **Blank lines** | Blank lines are ignored; use them to visually separate sections. |
| **Ordering** | Rules are evaluated top-to-bottom. The **first** matching rule wins. |
| **Whitespace** | Leading and trailing whitespace on a line is trimmed before parsing. |

---

## §1 — Files & Folders (`path:` prefix)

Ignore entire files, directories, or sets of files using glob patterns. Paths are **relative to the repository root**.

### Syntax

```
path: <glob-pattern>
```

### Glob Rules

These follow the same conventions as `.gitignore`:

| Pattern | Meaning |
|---------|---------|
| `path: portal/` | Ignore the entire `portal` directory and everything inside it. |
| `path: portal/src/legacy.py` | Ignore a single file. |
| `path: **/*.generated.ts` | Ignore all files ending in `.generated.ts` anywhere in the tree. |
| `path: docs/internal/` | Ignore the `docs/internal/` directory recursively. |
| `path: *.log` | Ignore all `.log` files at any depth. |

### Details

- A trailing `/` indicates a directory — the agent will skip the directory and all of its contents.
- `*` matches anything except a path separator.
- `**` matches zero or more directories.
- `?` matches exactly one character (not a path separator).
- Patterns without a `/` are matched against the **filename only** at any depth (like `.gitignore`).
- Patterns with a `/` (other than a trailing slash) are matched against the **full relative path**.

### Examples

```text
# Ignore the entire portal UI code
path: portal/

# Ignore auto-generated protobuf stubs
path: **/*_pb2.py
path: **/*_pb2_grpc.py

# Ignore a specific config used only in dev
path: config/local_overrides.yaml
```

---

## §2 — Functions & Invocation Chains (`fn:` prefix)

Ignore a specific function **and every function in its call chain** (callers and callees traced by the agent).

### Syntax

```
fn: <function-name>
```

### Matching Rules

| Pattern | What It Matches |
|---------|-----------------|
| `fn: page_elements_testxyz` | Any function named `page_elements_testxyz` in any module. |
| `fn: mypackage.views.page_elements_testxyz` | Only the function at that exact module path. |
| `fn: TestSuite.test_legacy_*` | Wildcard — any method on `TestSuite` starting with `test_legacy_`. |

### Details

- **Bare names** (no dots) match the function name regardless of which module it resides in.
- **Qualified names** (with dots) are matched against the fully-qualified path: `<module>.<class>.<method>` or `<module>.<function>`.
- The wildcard `*` is supported at the **end** of the name to match a prefix.
- When a function is ignored, the agent also ignores:
  - All functions **called by** the ignored function (downstream chain).
  - All test functions whose **sole purpose** is to exercise the ignored function.

### Examples

```text
# Ignore a deprecated test helper and everything it calls
fn: page_elements_testxyz

# Ignore all legacy migration functions in a specific module
fn: mypackage.db.migrations.legacy_*

# Ignore a known-flaky test class method
fn: RegressionTests.test_flaky_timeout
```

---

## §3 — CLI Commands (`cmd:` prefix)

Tell the agent to skip validation, testing, or analysis of a specific CLI command or subcommand.

### Syntax

```
cmd: <command> [subcommand] [args...]
```

### Matching Rules

| Pattern | What It Matches |
|---------|-----------------|
| `cmd: retriever xyz` | The exact invocation `retriever xyz` (with any trailing arguments). |
| `cmd: retriever migrate --dry-run` | Only when `--dry-run` is present. |
| `cmd: retriever *` | Any subcommand of `retriever`. |

### Details

- Matching is **prefix-based by default**: `cmd: retriever xyz` matches `retriever xyz`, `retriever xyz --verbose`, etc.
- Use `*` as a wildcard for any single token (subcommand or argument).
- Quoting is **not required** — tokens are split on whitespace.
- The agent will skip:
  - Generating tests that invoke the ignored command.
  - Validating output or exit codes of the ignored command.
  - Analyzing code paths that are exclusively reachable through the ignored command.

### Examples

```text
# Ignore the experimental "xyz" subcommand entirely
cmd: retriever xyz

# Ignore a specific migration dry-run variant
cmd: retriever migrate --dry-run

# Ignore all retriever subcommands (nuclear option)
cmd: retriever *
```

---

## §4 — Text / Verbs (`text:` prefix)

Ignore any file whose **contents** match a free-text keyword or phrase. The agent wraps the term in implicit wildcards (`*<term>*`) and scans file contents — any file that contains a match is excluded from QA analysis.

This is useful for ignoring everything related to a concept, feature flag, internal codename, or identifier that appears across many files without having to enumerate each path.

### Syntax

```
text: <term>
```

### Matching Rules

| Pattern | What It Matches |
|---------|-----------------|
| `text: vlm_embedder` | Any file containing the substring `vlm_embedder` anywhere in its contents (case-sensitive). |
| `text: EXPERIMENT_FLAG` | Any file containing `EXPERIMENT_FLAG`. |
| `text: deprecated_pipeline` | Any file containing `deprecated_pipeline`. |

### Details

- The match is **substring / wildcard** — the agent treats `text: foo` as `*foo*` applied to the full text of every file in scope.
- Matching is **case-sensitive** by default. Append `/i` to make it case-insensitive: `text: vlm_embedder/i`.
- The term can contain spaces; everything after `text: ` (up to end-of-line) is the search term.
- When a file matches, the **entire file** is ignored — not just the matching line.
- Binary files and files already excluded by `path:` rules are never scanned.
- Use this sparingly on large repos — broad terms may unintentionally exclude many files. Prefer specific identifiers over common words.

### Examples

```text
# Ignore all files that reference the VLM embedder subsystem
text: vlm_embedder

# Ignore anything touching a deprecated pipeline by codename
text: deprecated_pipeline

# Ignore files with an experimental feature flag (case-insensitive)
text: USE_EXPERIMENTAL_RERANKER/i

# Ignore files mentioning a specific internal tracking tag
text: NEMO-SKIP-QA
```

---

## §5 — Pull Requests (`#` prefix)

Ignore all changes introduced by a specific pull request number. The agent resolves the PR's diff and excludes every file and hunk touched by it.

### Syntax

```
#<PR-number>
```

> **Note:** There is no space between `#` and the number. This distinguishes PR references from comments (which have a space or text after `#`).

### Disambiguation From Comments

| Line | Interpretation |
|------|----------------|
| `#1829` | **PR reference** — ignore PR 1829. |
| `# 1829` | **Comment** — the space after `#` makes it a comment. |
| `# Ignore portal code` | **Comment**. |
| `#portal` | **Comment** — not a number, so not a PR reference. |

### Details

- The PR number must be a positive integer immediately following `#`.
- The agent will look up the PR's merge commit (or head commit if unmerged) and determine which files were changed.
- All files and diff hunks from that PR are excluded from QA analysis.
- Multiple PRs can be listed, one per line.

### Examples

```text
# Ignore the large refactor in PR 1829
#1829

# Also ignore the experimental feature PR
#2044
```

---

## Full Example `.qaignore` File

```text
# =============================================================================
# .qaignore — QA Automation Ignore File
# =============================================================================

# --- Files & Folders ---
path: portal/
path: **/*_pb2.py
path: config/local_overrides.yaml

# --- Functions ---
fn: page_elements_testxyz
fn: mypackage.db.migrations.legacy_*

# --- CLI Commands ---
cmd: retriever xyz
cmd: retriever migrate --dry-run

# --- Text / Verbs ---
text: vlm_embedder
text: deprecated_pipeline
text: USE_EXPERIMENTAL_RERANKER/i

# --- Pull Requests ---
#1829
#2044
```

---

## Quick-Reference Cheat Sheet

| Prefix | Purpose | Example |
|--------|---------|---------|
| `path:` | Ignore files/folders by glob | `path: portal/` |
| `fn:` | Ignore a function + call chain | `fn: page_elements_testxyz` |
| `cmd:` | Ignore a CLI command | `cmd: retriever xyz` |
| `text:` | Ignore files containing a term | `text: vlm_embedder` |
| `#N` | Ignore a PR by number | `#1829` |
| `# text` | Comment (ignored by parser) | `# this is a note` |
