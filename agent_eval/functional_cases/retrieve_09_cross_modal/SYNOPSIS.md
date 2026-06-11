# Synopsis — RETRIEVE: cross-modal retrieval (text query surfaces non-text content)

**What user task this covers.** A developer wants to type a plain-English question and have
NeMo Retriever hand back the **chart, image, infographic, or table** that answers it —
*without* first transcribing those visuals into text by hand, and *without* building a
separate visual-search system. "Find me the chart that shows revenue growth." "What does
this chart show — what's the most striking number?" "Which images discuss this topic?" The
answer often lives in a picture that carries no native text, so the library has to cross the
modality boundary: a **text** query has to reach **non-text** content.

**How it works (two routes).** NeMo Retriever bridges text and visuals in one of two ways,
and either one satisfies this task. (A) It embeds the visual with a **vision-language model**
so a text query lands in the same space and matches the chart/image directly. (B) `retriever
ingest --caption` runs a vision-language model that writes a plain-English **caption** into
the visual's row, which an ordinary text query then matches. Either way, the developer never
hand-extracts the picture.

**How we test it.** Five agent prompts, each handing the agent a small set of real visuals
(a standalone bar chart, a table image, a 3-page multimodal PDF) and checking that the agent
drives the `retriever` CLI correctly: it ingests the corpus (with `--caption` and/or VL
embedding so visuals become text-reachable), then runs a **text** `retriever query` and uses
`--content-types chart,image,infographic,table` to keep only **non-text** hits. We grade by
an **operational pass** (not a RAGAS judge score): the text query must return a hit whose row
type is a chart/image/infographic/table that **semantically matches** the question, with a
file + page citation.

**The five tests, simplest to hardest:**

1. **Text → table** — a typed question pulls back a `content_type=table` row (the Giraffe row
   of a table inside a PDF). The floor: text reaches a non-text, typed row.
2. **Text → chart** — the query surfaces a `content_type=chart` hit (a gadget-cost bar
   chart). A chart has no native text, so this needs the cross-modal route.
3. **Visual-only results** — across a mixed folder, the `--content-types` filter strips out
   every plain-text passage, so the only hits returned are charts/images/tables.
4. **Image-by-caption** — a plain-English *description* (not the chart's labels) retrieves a
   standalone image via its auto-generated caption (ties to the captioning task).
5. **Acceptance gate** — one plain query into a fresh index returns the *right* chart (the
   gadget-cost one, not the car-color table) as the top visual hit, with a content_type and a
   citation, with no hand pre-extraction, in under a minute.

**Why this order.** Each rung adds one thing: first text → a typed table that still carries
text, then text → a chart with *no* native text, then the visual-only filter that proves the
results are genuinely non-text, then the caption route for a standalone image, then everything
composed into the pass/fail gate — including that the *correct* visual comes back by intent.

**The trap built into the suite.** The cross-modal route is not on by default — `--caption`
is never enabled by a profile, and the bare in-process ingestor's caption method is a **stub
that does nothing**. If the agent forgets `--caption` (and VL embedding isn't configured), the
visuals have no text to match, and a visual-only query comes back **empty** — that empty
result is the exact fail signal rungs 2–5 check for.

**Status.** Tests are authored and grounded in the real CLI + a `--caption --dry-run`
(offline); **not yet run live**. A live run needs a working cross-modal route: `--caption` /
VL embedding reachable via `NVIDIA_API_KEY` or a pinned `--caption-invoke-url` /
`--embed-invoke-url` (CPU/hosted), or a local VLM / VL container (GPU). See `README.md` for
the full spec and `cases.json` for the machine-gradable definitions.
