# Synopsis — INGEST: a mixed-format, multi-modal folder in one invocation

**What user task this covers.** A user has a folder full of different kinds of documents —
PDFs, a Word file, a scanned image, an HTML page, "the works" — and wants NeMo Retriever to
**ingest the whole folder in a single command** and make it searchable. Success means: point
the skill at the directory once, have every file auto-routed to the right extractor by its
extension, have unsupported files quietly skipped (not crash the run), get a clear
end-of-run summary, and end up with one combined index that loads and answers questions
spanning multiple source files.

**How we test it.** Five agent prompts, each handing the agent one small real **folder** and
a question that can only be answered if the folder was ingested in one shot and the right
files were routed correctly. We check the agent drives the `retriever` CLI properly: a
**single** `retriever ingest <folder>/` (the folder is globbed automatically — there is **no**
`--input-type` flag, and **no** per-file loop), then a `retriever query` to prove the combined
index. We ground the routing with `retriever ingest <folder>/ --dry-run`, which prints a
`branch_summary` (e.g. `pdf:2, image:1, html:1`) showing each file landing in its modality
branch.

**The five tests, simplest to hardest:**

1. **Baseline folder** — a folder of two PDFs ingested in one call → one combined index.
   (Ground truth: James, 2019 → **978**.) Proves the directory is ingested in a single
   invocation.
2. **Mixed formats** — add a Word doc and an HTML page; each is auto-detected and routed by
   extension in the same call (`pdf:2, html:1`; the docx is converted to PDF by libreoffice).
   (Ground truth: HTML "My First Heading"; Giraffe → Driving a car → At the beach.)
3. **Multi-modal** — add a scanned **PNG**, so the one ingest now fans out across three
   branch families at once (`pdf:2, image:1, html:1`) and the image goes through its own
   image pipeline, not the PDF one.
4. **Unsupported file** — drop an Excel `.xlsx` into the folder; the ingest **skips** it and
   still indexes the supported files (`pdf:1, html:1`) instead of aborting the whole run.
5. **Acceptance gate** — a mixed multi-modal folder ingested in **one** invocation into a
   named index, returning an end-of-run summary (files by type, total rows, index path),
   landing the canonical LanceDB schema, loadable in a fresh process, and answering a query
   that spans **two** different source files (New Hampshire / 1923 from one PDF **and** James
   2019 = 978 from another). This is the test the others build up to.

**Why this order.** Each rung adds exactly one new thing: first "does a folder ingest in one
call at all," then multi-**format** (routing by extension), then multi-**modal** (an image
branch beside the docs), then graceful handling of an **unsupported** member, then everything
composed — one-shot ingest, the summary, the canonical schema, a fresh-process load, and a
cross-file query.

**Status.** Tests are authored and grounded in the real CLI (`--dry-run`, offline) and the
shipped source; **not yet run live**. Live runs need libreoffice (for the DOCX/PPTX members),
the `[multimedia]` extra + ffmpeg only if a folder includes audio/video members (this suite's
fixtures do not), and a reachable extraction backend (hosted `ai.api.nvidia.com` with an API
key, or a local GPU). One known gap to flag: the shipped CLI skips an unsupported folder
member by *exclusion* (it never enters the plan, no error) rather than emitting an explicit
per-file "skipped: reason" log line — see `README.md` / the rung-4 caveat. See `README.md`
for the full spec and `cases.json` for the machine-gradable definitions.
