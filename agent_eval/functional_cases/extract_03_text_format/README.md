# Functional test suite — EXTRACT: standalone text-format extraction (HTML / Markdown / .txt / JSON / SH)

An agent-driven functional test suite for the **NeMo Retriever Library skill**, built
against the real CLI in `nemo_retriever/nemo_retriever/src` (`retriever ingest` /
`retriever query`).

This suite covers the **cheapest, fully-local extraction path**: text-native files
(`.txt`, HTML, Markdown, JSON, shell scripts) carry no images, no scanned pages and no
tables-as-pixels, so extraction needs **no OCR, no page-elements, no table-structure, and
no GPU embedding**. The agent must route these to the CPU-only `txt` / `html` ingest
branches and **never** invoke a GPU NIM. Each test is a self-contained triple — a prompt,
a per-case `data/` folder, and an expected output naming the right `retriever` subcommand(s)
and asserting the no-GPU-NIM path.

---

## The user task under test

> **JTBD: EXTRACT.** "Standalone Text-format extraction (HTML / Markdown / `.txt` / JSON /
> SH)." — **P0**

**Success criteria for the row:**

| Dimension | Target for this row |
|---|---|
| Quality (operational) | binary pass: (1) skill routes to the **CPU-only text-extract path — no GPU NIM** (no page-elements / OCR / table-structure / GPU embedding), (2) ingest produces non-zero rows, (3) a query recovers the expected text / heading / table row with a citation |
| Time | **fast — ≤ 30s** wall clock (cheapest, fully-local path) |
| Trigger rate | ≥ 95% — a "make this text/HTML/markdown folder searchable" prompt must fire the skill |
| Subcommand accuracy | ≥ 90% — `retriever ingest <file/dir>` (auto-detect, **no** `--input-type`, **no** GPU/batch flags) then `retriever query`; `--dry-run` may be used to prove the plan has no GPU NIM stage |
| Token usage | tracked, not gated |

Seed queries this suite is derived from (paraphrased):
- *"Make this folder of HTML docs searchable for Q&A."*
- *"From aurora_README.md, what's the SLO for eu-central-1? Quote the exact row from the capacity table."*
- *"Pull the content out of api_changelog.txt."*

---

## How the CLI extracts text formats (verified against the shipped code)

Grounded by `nemo_retriever/common/input_files.py` and `retriever ingest … --dry-run`
(offline, no network):

- **Auto-detection.** Format is auto-detected from extension — there is **no
  `--input-type` flag**. `INPUT_TYPE_PATTERNS` registers `.txt` → the `txt` branch and
  `.html` → the `html` branch (alongside pdf/docx/pptx/images/audio/video).
- **CPU-only, no GPU NIM.** `retriever ingest <txt/html> --dry-run` resolves to
  `branch_summary` like `"txt:1"` / `"html:1"`, `extraction_mode` `"text"` / `"html"`,
  `family` `"txt"` / `"html"`. In that plan **every GPU field is null**
  (`gpu_embed`, `gpu_page_elements`, `gpu_ocr`, `gpu_table_structure`,
  `gpu_nemotron_parse`) and `embed` is null for a pure-extract dry-run. The whole visual
  stack (page-elements → OCR → table-structure on `ai.api.nvidia.com`) is **bypassed** —
  these formats never need it to extract. This is the no-GPU/NIM emphasis of the row.
- **Clean text.** HTML is parsed and stripped to readable text (doctype/head/title/nav/
  anchors/tags removed); Markdown structure (headings, pipe tables, code fences) is
  preserved verbatim as text; `.txt` passes through as-is.
- **Row metadata.** Each emitted row carries `source_id`, `path` (source filename),
  `page_number = 1` for a single text/html block (or `-1` where a page index is N/A), and
  a hierarchy block. At the CLI layer, each query hit surfaces exactly
  `{source, page_number, text}`.
- **Ingest success line:** `Ingested N file(s) → M row(s) in LanceDB lancedb/<table>.`

### Supported-extension caveat (IMPORTANT — grounded against the shipped CLI)

The seed task names **Markdown / JSON / SH** as text formats, but the **shipped** CLI's
`INPUT_TYPE_PATTERNS` registers only `.txt` and `.html` for text — **not** `.md`, `.json`
or `.sh`. `retriever ingest file.md` errors:

```
Error: Unsupported input file type(s) for retriever ingest: …/file.md
```

(verified live with the venv binary on `test.md`, `test.json`, `test.sh`.)

Since Markdown, JSON and shell scripts are all plain UTF-8 text, the **grounded path** for
them today is to ingest the byte-identical content under a **`.txt`** name — the package's
own text-glob example uses `[".txt", ".html"]`. A `.txt`-named copy round-trips through the
**identical** CPU text branch with all structure (headings, pipe tables, code fences, the
JSON object, the shell command) preserved verbatim. Rungs 3–5 therefore ingest md/json/sh
content under a `.txt` name and carry a `known_caveat`; a live run should re-check whether a
newer CLI build registers `.md` / `.json` / `.sh` natively (in which case drop the rename).

Convention in every command: `RETRIEVER=$RETRIEVER_VENV/bin/retriever`.

---

## The five tests (increasing complexity)

| # | id | What it adds over the previous rung | Subcommands |
|---|---|---|---|
| 1 | `extract-txt-001` | **Baseline.** One plain `.txt` → CPU `txt` branch, non-zero rows, no GPU NIM. | `ingest`, `query` |
| 2 | `extract-txt-002` | **HTML branch.** Markup-aware parse; boilerplate + tags stripped to clean text. | `ingest`, `query` |
| 3 | `extract-txt-003` | **Markdown structure.** Headings + a pipe table survive; recover an exact table row. (md ingested as `.txt`.) | `ingest`, `query` |
| 4 | `extract-txt-004` | **Structured / script text.** JSON + SH ride the same CPU text branch; literal content round-trips. (ingested as `.txt`.) | `ingest`, `query` |
| 5 | `extract-txt-005` | **Acceptance gate.** Mixed HTML+MD+TXT folder in ONE ingest, all on CPU (no GPU NIM, ≤30s), exact row recovered with citation. | `ingest` (+`--dry-run`), `query` |

The ladder adds one dimension per rung: rung 1 proves the simplest text path runs with no
GPU NIM; rung 2 adds markup-aware HTML parsing; rung 3 adds structured Markdown (heading +
table) and an exact-row recovery; rung 4 adds the remaining non-prose text formats (JSON,
SH); rung 5 composes a heterogeneous one-call folder ingest into the row's operational-pass
gate (CPU-only, fast, exact row + citation).

---

### T1 — `extract-txt-001` · plain `.txt` extraction  *(complexity 1)*
- **Satisfies:** the text-format EXTRACT task in its simplest form.
- **Data:** `data/api_changelog.txt` (Aurora API changelog; v2.4.0 added `/v1/embed/batch`,
  deprecated `/v1/embed/single`, added the `enterprise-plus` 5000 req/min tier).
- **Expected:** `RETRIEVER ingest data/api_changelog.txt` (branch `txt:1`, all GPU fields
  null) → `RETRIEVER query "What did the v2.4.0 release add?" --top-k 5` → the v2.4.0
  additions, citing `api_changelog.txt` p1.

### T2 — `extract-txt-002` · HTML → clean text  *(complexity 2)*
- **Satisfies:** the **HTML** clause + the "boilerplate stripped" validation bullet.
- **Data:** `data/architecture.html` (Aurora 3-tier architecture; p99 SLO < 350ms).
- **Adds:** markup-aware parsing (the `html` branch) — the queried text must be clean
  (no `<html>/<head>/<title>/<a href>`), recovering the three tiers and the <350ms SLO.

### T3 — `extract-txt-003` · Markdown structure + exact table row  *(complexity 3)*
- **Satisfies:** the **Markdown** clause + the seed "quote the exact eu-central-1 row".
- **Data:** `data/aurora_README.md` (provided for reference) **and** `data/aurora_README.txt`
  (byte-identical, the ingested copy). Capacity table includes
  `| eu-central-1 | 192 H200 | 310ms |`.
- **Adds:** structured Markdown — headings + a pipe table must survive extraction so the
  query recovers SLO p99 = **310ms** and quotes the exact eu-central-1 row.
- **Caveat (built in):** `.md` is not a registered extension — ingest the `.txt` copy.

### T4 — `extract-txt-004` · JSON / SH structured-script text  *(complexity 4)*
- **Satisfies:** the **JSON** and **SH** clauses.
- **Data:** `data/test.json` (`{"a":4,"b":2}`) and `data/test.sh` (`echo "Hello World!"`)
  for reference, plus `data/test.json.txt` / `data/test.sh.txt` (the ingested `.txt`-named
  copies; `Path.suffix` resolves to `.txt`).
- **Adds:** non-prose structured/script text rides the same CPU `txt` branch; query returns
  `a = 4` (and `b = 2`) and the echoed `Hello World!`, citing both sources at p1.
- **Caveat (built in):** `.json` / `.sh` are not registered extensions — ingest `.txt` copies.

### T5 — `extract-txt-005` · acceptance: mixed folder, CPU-only, exact row + citation  *(complexity 5)*
- **Satisfies:** the complete operational-pass row + the no-GPU / fast / exact-row-with-
  citation validation path.
- **Data:** `data/` = `index.html` + `runbook.html` + `aurora_README.txt` +
  `api_changelog.txt` (mixed HTML + Markdown-as-txt + plain `.txt`).
- **Expected:** `RETRIEVER ingest data/ --dry-run` (inspect `branch_summary "txt:2, html:2"`,
  all `gpu_*` null, `embed` null) → `RETRIEVER ingest data/` (4 files, one call) →
  `RETRIEVER query "eu-central-1 capacity SLO row" --top-k 5` → quotes
  `| eu-central-1 | 192 H200 | 310ms |` attributed to `aurora_README.txt` p1.
- **Adds (the gate):** heterogeneous one-call folder ingest + explicit no-GPU-NIM / ≤30s
  assertions + exact-row-with-citation.

---

## Running / grading

Mount each case's `data/` folder into the agent workdir, give it the `prompt`, and grade
against `pass_when` in `cases.json`. The defining checks unique to this suite:
**(a)** the agent uses bare `retriever ingest` auto-detection (no `--input-type`, no GPU /
batch flags); **(b)** the resolved plan / trace shows the **CPU `txt`/`html` branch with
every GPU field null and no page-elements/OCR/table-structure/visual-NIM call**;
**(c)** the queried text is clean (HTML) / structure-preserving (Markdown) and the exact
expected row/heading is recovered with a citation; **(d)** wall clock ≤ 30s.

**Note on live runs — NOT YET RUN LIVE.** Expected outputs are grounded in the CLI source
(`common/input_files.py` extension map) and `--dry-run` (offline, no network); the suite has
**not** been executed live end-to-end. That said, **this is the cheapest path to actually
run**: it is fully local / CPU-only, needs no GPU and no hosted API key for extraction, and
is the obvious candidate to validate for real first. A live run would capture concrete row
counts per file, the exact clean-text/structure-preserved output, wall-clock latency (to
confirm the ≤30s bucket), and token baselines — and should re-confirm the
supported-extension caveat (whether a newer CLI registers `.md` / `.json` / `.sh` natively,
which would let rungs 3–5 drop the `.txt` rename).
