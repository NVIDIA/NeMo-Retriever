# Synopsis — EXTRACT: standalone text-format extraction (HTML / Markdown / .txt / JSON / SH)

**What user task this covers.** A developer has text-native files — plain `.txt`, HTML
pages, Markdown READMEs, JSON configs, shell scripts — and wants their content pulled out
and made searchable for Q&A. Because these formats carry no images, scanned pages or
pixel-tables, extracting them needs **no OCR, no page-elements, no table-structure and no
GPU**. This is the **cheapest, fully-local extraction path**. Success means the skill
ingests the files on the CPU-only text branch (never firing a GPU NIM), produces non-zero
searchable rows, and a query recovers the expected text / heading / table row with a
citation — fast (under 30 seconds).

**How we test it.** Five agent prompts, each handing the agent a small set of text files
and checking that the agent drives the `retriever` CLI correctly: bare `retriever ingest`
auto-detection (no `--input-type`, no GPU/batch flags) then `retriever query`, with the
resolved plan / trace showing the **CPU `txt` / `html` branch and every GPU field null** —
no page-elements/OCR/table-structure/embedding on a GPU. HTML must come out clean
(boilerplate and tags stripped); Markdown structure (headings, tables, code blocks) must be
preserved; each row carries its source, path and page number.

**The five tests, simplest to hardest:**

1. **Plain `.txt`** — ingest one text file and answer from it. Baseline: proves `.txt`
   routes to the CPU text branch with non-zero rows and no GPU NIM.
2. **HTML → clean text** — parse an HTML page, strip the page chrome and tags, and answer
   from the clean body text.
3. **Markdown structure** — ingest a README whose capacity table has a row per region, and
   quote the exact `eu-central-1` SLO row — proving headings and the table survived.
4. **JSON / SH** — pull the literal content out of a JSON config and a shell script, riding
   the same CPU text branch.
5. **Acceptance gate** — ingest a mixed folder (HTML + Markdown + `.txt`) in one call, all
   on CPU with no GPU NIM and under 30 seconds, and recover the exact capacity-table row
   with a citation. This is the test the others build up to.

**Why this order.** Each rung adds one dimension: first the simplest text path, then
markup-aware HTML parsing, then structured Markdown (heading + table) with an exact-row
recovery, then the remaining non-prose formats (JSON, SH), then everything composed into a
heterogeneous one-call folder ingest that is the row's real pass/fail gate.

**One important caveat (grounded against the shipped code).** The shipped CLI's extension
map registers only `.txt` and `.html` for text formats — **not** `.md`, `.json` or `.sh`;
ingesting those raw errors with "Unsupported input file type(s)". Since all three are plain
UTF-8 text, the grounded path is to ingest the byte-identical content under a `.txt` name
(it round-trips through the identical CPU text branch with structure preserved). Rungs 3–5
do this and flag it; a live run should re-check whether a newer CLI build adds those
extensions natively.

**Status.** Tests are authored and grounded in the real CLI extension map plus offline
`--dry-run` plans; **not yet run live**. Of all the suites this is the cheapest to actually
run — it is fully local / CPU-only and needs no GPU or hosted API key for extraction. See
`README.md` for the full spec and `cases.json` for the machine-gradable definitions.
