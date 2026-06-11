# Synopsis — SETUP #1: CPU-only setup via build.nvidia.com

**What user task this covers.** A developer on a laptop or server with **no GPU** wants to
get NeMo Retriever working using only an **NVIDIA API key** and NVIDIA's **hosted models**
(build.nvidia.com) — nothing runs locally. Success means: the library installs cleanly,
they can load a PDF (extraction/embedding happen over hosted endpoints), and they can ask a
question and get a grounded answer — all end-to-end in **under 10 minutes**.

**How we test it.** Five agent prompts that each hand the agent a small set of real PDFs
and check that the agent drives the `retriever` CLI correctly: the right subcommands
(`ingest` then `query`), the right CPU/hosted flags (API key + hosted endpoints, **never**
local-GPU flags), and a correct, cited answer.

**The five tests, simplest to hardest:**

1. **Smoke loop** — load one trivial text PDF and answer a question. Proves the install
   works and the ingest→query loop closes on a no-GPU box.
2. **Prove the routing** — before any real call, the agent uses a dry-run to confirm it
   will use hosted endpoints + the API key and *not* a GPU, then runs for real.
3. **Extraction reachable** — load a table PDF and read a specific cell, exercising the
   full hosted extraction stack (layout → OCR → table structure).
4. **Reranking reachable** — load a 3-PDF folder in one shot and answer with reranking on,
   exercising the hosted reranker and best-match-first ordering.
5. **Acceptance gate** — a clean end-to-end run into a custom-named index, with a cited
   answer, a check that no local GPU was used, and the ≤ 10-minute deadline. This is the
   test the others build up to.

**Why this order.** Each rung adds exactly one new thing — first just "does it run," then
"does it run the *right* way," then each of the three hosted model classes (embedding →
extraction → reranking) in turn, then everything composed into the real pass/fail gate.

**Status.** Tests are authored and grounded in the real CLI; **not yet run live** (live
runs make small billable build.nvidia.com calls). See `README.md` for the full spec and
`cases.json` for the machine-gradable definitions.
