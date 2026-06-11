# Synopsis — DX: tinker with any exposed configuration knob across the pipeline

**What user task this covers.** A developer wants to **try a different setting** without
ceremony: a bigger chunk size, a different embedding model, more reranked results — just for
this run. Success means the override **takes effect on the very next command with no rebuild
or reinstall**, the new value is **actually visible in the pipeline's resolved plan** (not
silently ignored in favor of the default), and the change **does not stick** to later runs
unless the developer deliberately persists it.

**How we test it.** Five agent prompts, each asking to override an exposed knob "for this
run." We lean on the fact that `retriever ingest … --dry-run` prints the **fully resolved
plan as JSON, offline** — so for ingest-time knobs (chunk size/overlap, embedding model) we
can read the field back and confirm the override reached the operator. `retriever query` has
no dry-run, so the query-time knob (number of reranked results) is proven by its **observed
effect** on the next query, with no re-ingest. We check the agent uses the **real** flags
(`--text-chunk-max-tokens` / `--text-chunk-overlap-tokens`, `--embed-model-name`,
`--top-k` / `--rerank`) and never invents one.

**The five tests, simplest to hardest:**

1. **Chunk-size override** — load a PDF with a 1000-token chunk / 100-token overlap and prove
   the `--dry-run` plan shows **1000 / 100** instead of the default **1024 / 150**. The DX
   floor: one knob, visible in the trace, no rebuild.
2. **Embedding-model swap** — ingest with a different embedder (`nvidia/nv-embedqa-e5-v5`) and
   prove it lands in the plan with no reinstall. (Model swaps are the case people wrongly
   assume need a rebuild.)
3. **Query-time top-k** — bump reranked results from the default up to **20** on the next
   query, reusing the already-built index (no re-ingest), with reranking on.
4. **Default restored** — run the overridden call, then a plain one, and confirm the chunk
   settings revert to **1024 / 150**: the override is not sticky.
5. **Acceptance gate** — in one workflow, override knobs on two stages (chunking at ingest +
   top-k/rerank at query), each visible in its own trace, then show a follow-up plain call is
   back to defaults. The test the others build up to.

**Why this order.** Each rung adds exactly one new thing: first "does one ingest-time knob
reach the trace," then a different knob on a different stage (the embedding model), then a
query-time knob proven by effect with no re-ingest, then non-stickiness (default restored),
then everything composed — multiple knobs across the pipeline in one workflow, each traced,
defaults restored after.

**Verified default vs override values** (from `--dry-run` / `--help` on 2026.06.10.devXXXX):
chunk **1024/150 → 1000/100** (`split_config.pdf.{max_tokens,overlap_tokens}`); embedding
model default unset/bundled **→ `nvidia/nv-embedqa-e5-v5`** (`embed.embed_model_name`);
query **`--top-k` 10, `--rerank` off → `--top-k 20 --rerank` on**.

**Status.** Tests are authored and grounded in the real CLI (`--dry-run` + `--help`); **not
yet run live**. The four `--dry-run` steps are offline; the two real `ingest`/`query` steps
(T3, T5) would need a reachable embedding/reranker backend (hosted endpoints with an API key,
or a local GPU). A live run would capture real row counts at the overridden chunking, the
hit-array growth as top-k rises toward 20 on a larger corpus, latencies, and token baselines.
See `README.md` for the full spec and `cases.json` for the machine-gradable definitions.
