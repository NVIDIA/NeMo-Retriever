# Synopsis — RETRIEVE: embed then rerank

**What user task this covers.** A user has a small set of documents and wants NeMo
Retriever to **find the relevant chunks with the embedder and then put the most relevant
passages first by reordering them with the reranker**. Success means: the system runs a
dense vector search, hands the candidates to the reranker
(`nvidia/llama-nemotron-rerank-1b-v2`), and returns a top-k list whose order has actually
changed because of reranking — with the correct, grounded answer sitting in that top-k.

This is an **operational-pass** suite: we check the *mechanics* of the embed→rerank chain
(right flags, the order changed, the gold passage is in the top-k with a citation). It is
**not** RAGAS judge grading — that quality-score grading lives in the separate
performance-eval suites.

**How we test it.** Five agent prompts, each handing the agent one or a few small real
PDFs and a question that can only be answered well if the right passage was retrieved and
floated to the top. We check the agent drives the `retriever` CLI correctly: `retriever
ingest <paths>` then `retriever query "<q>" --rerank --top-k K`. Reranking is **off by
default**, so the agent must add `--rerank` to turn it on; and to *prove* reranking did
something, we run the same query with `--no-rerank` and with `--rerank` and confirm the
ordering changed.

**The five tests, simplest to hardest:**

1. **Baseline dense retrieve** — one document, a `--no-rerank` query, just confirm the
   embedder returns the relevant passage (the owner's "house is in the village"). This is
   the dense-only ordering the next rung will reorder.
2. **Turn on the reranker** — same document and query, now with `--rerank`. The gold passage
   ("miles to go before I sleep") is promoted and the order differs from the dense-only run.
   The before/after pair is the proof the reranker did something.
3. **Across multiple documents** — load three PDFs; dense search pulls competing candidates
   from all of them, and the reranker has to surface the single best passage across docs
   (which Frost collection was published in 1923 → New Hampshire).
4. **Wide pool, narrow answer** — set the candidate pool (`--candidate-k`) bigger than the
   final `--top-k`, so the reranker reorders a broad set of near-duplicate numeric rows and
   *promotes* the exact cell (James, 2019 → **978**) into a small top-k.
5. **Acceptance gate** — a small labeled set of three questions (one per document) over the
   whole corpus; each must return its gold row in the reranked top-k with a citation, and we
   spot-check that reranking changed the order. This is the test the others build up to.

**Why this order.** Each rung adds exactly one new thing: first "does dense retrieval find
the chunk," then "does turning on the reranker change the order," then "can it win across
multiple documents," then "can it promote an answer from a wider candidate pool into a
narrow top-k," then everything composed into a small labeled mini-eval.

**Status.** Tests are authored and grounded in the real CLI (`retriever query --help`
confirms `--rerank/--no-rerank`, `--candidate-k`, `--top-k`) and the fixtures' verified
text; **not yet run live**. Live runs need a reachable embedder + reranker backend (hosted
`integrate.api.nvidia.com` / `build.nvidia.com` with an API key, or a local GPU with the
`hf` backends). See `README.md` for the full spec and `cases.json` for the machine-gradable
definitions.
