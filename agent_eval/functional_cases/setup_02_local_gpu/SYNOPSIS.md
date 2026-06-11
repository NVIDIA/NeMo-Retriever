# Synopsis — SETUP #2: local GPU setup (models on the card)

**What user task this covers.** A developer with a **CUDA-capable GPU** (e.g. an H100)
wants to run NeMo Retriever **entirely on-device** — extraction, embedding, and reranking
models loaded **directly onto the local GPU**, with **no calls to the cloud**. Success
means: the library installs with local GPU support, the models load onto the card, they can
ingest a PDF and ask a question, and the answer is served by GPU-resident models — all
end-to-end in **under 30 minutes**.

**How we test it.** Five agent prompts that each hand the agent a small set of PDFs and
check that the agent drives the `retriever` CLI on-device: the right subcommands (`ingest`
then `query`), the **local-GPU** flags (`--query-embed-backend hf --reranker-backend hf
--rerank` — *not* hosted endpoints or an API key), the models actually sitting in GPU
memory, and **zero** outbound calls to build.nvidia.com.

**The five tests, simplest to hardest:**

1. **On-device smoke loop** — install with local GPU support, confirm a GPU is visible,
   load one text PDF and answer. Closes the loop with no network at all.
2. **Prove it's on the card** — confirm the models are actually resident in GPU memory and
   that nothing leaked to the cloud during ingest or query.
3. **Local GPU reranker** — load a 3-PDF folder and answer with reranking done on-device,
   exercising the reranker model on the GPU.
4. **Local GPU extraction** — load a table PDF and read a specific cell with the document
   parsing (page-elements/OCR/table) running on the GPU too. This is the hardest part: by
   default that parsing calls the cloud, so the test is built to catch a silent fallback.
5. **Acceptance gate** — a clean end-to-end run into a custom-named index, with a cited
   answer, a check that all three model classes ran on the GPU with no cloud calls, and the
   ≤ 30-minute deadline. This is the test the others build up to.

**Why this order.** Each rung moves one more thing on-device: first "does the on-device
loop run," then "are the models *really* on the card and nothing leaked," then each model
class in turn (reranking → the hard one, extraction), then everything composed into the
real pass/fail gate.

**Relationship to suite 1.** This is the GPU mirror of `setup_01_cpu_hosted`. Same prompts'
spirit and same small PDFs — the only variable under test is **where the models run**
(local GPU here vs. hosted endpoints there).

**Status.** Tests are authored and grounded in the real CLI + skill install/query
references; **not yet run live** (live runs need an actual CUDA-13 GPU host). See
`README.md` for the full spec and `cases.json` for the machine-gradable definitions.
