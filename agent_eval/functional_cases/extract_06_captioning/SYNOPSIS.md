# Synopsis — EXTRACT: image / picture-region captioning

**What user task this covers.** A user wants NeMo Retriever to *describe images in plain
English* and make those descriptions searchable. The image can be a **standalone file**
(a chart, a table image) or a **picture region sitting inside a document** (a chart on a
page of a PDF / DOCX / PPTX). Success means: the library generates a natural-language
**caption** for each visual via a vision-language model, indexes it alongside the other
extracted chunks, and then lets the user **find that visual by typing a plain-text
description** — so a chart nobody transcribed is still retrievable by what it shows.

**How we test it.** Five agent prompts, each handing the agent a small set of images / a
visual PDF and checking that the agent drives the `retriever` CLI correctly: it runs
`retriever ingest <input> --caption` (the VLM caption stage — which is **off by default and
must be explicitly requested**), the captioned visual rows carry a **non-empty
natural-language caption**, and a later `retriever query` with a plain-English description
surfaces those visuals. We assert that a caption was produced and is retrievable — we do
**not** assert the exact caption wording, since VLM output varies.

**The five tests, simplest to hardest:**

1. **Standalone chart caption** — caption one image; confirm a non-empty caption row exists.
2. **Caption is retrievable** — a plain-English description query surfaces the image via its
   caption (closes the ingest→query loop).
3. **Region inside a document** — caption picture regions inside a PDF; the caption row
   records *where on the page* the region sits (a normalized bounding box).
4. **Also caption infographics** — extend captioning to infographic crops with one flag, and
   search their takeaways in plain English.
5. **Acceptance gate** — a mixed corpus (standalone images + a charts PDF) into a fresh named
   index, every visual captioned and text-queryable, in-document regions localized, **and** a
   check that the captioning actually ran — not silently skipped.

**Why this order.** Each rung adds one thing: first "is a caption produced," then "is it
searchable," then move from a standalone image to a region *inside a document* (the second
half of the task), then the infographic dimension, then everything composed into the
pass/fail gate.

**The trap built into the suite.** Captioning is never turned on by a profile, and the
plain in-process ingestor's caption method is a **stub that does nothing**. If the agent
forgets `--caption`, or the work falls through to that stub instead of a caption-capable
ingestor, captioning **silently no-ops** — the visuals keep empty descriptions and stay
invisible to text search. Test 5 is built to catch exactly that.

**Status.** Tests are authored and grounded in the real CLI + a `--caption --dry-run`
(offline); **not yet run live**. A live run needs a reachable VLM endpoint (hosted via
`NVIDIA_API_KEY`/`--caption-invoke-url` on CPU, or a local VLM container on GPU). See
`README.md` for the full spec and `cases.json` for the machine-gradable definitions.
