---
name: nemo-retriever-evaluate
description: Use when the user asks to measure NeMo Retriever retrieval quality, recall, QA answer quality, compare retrieval outputs, export evaluation JSON, build page markdown indexes, or run `retriever recall` / `retriever eval` workflows. Do not use for ad-hoc answering from a single index; use `nemo-retriever-query`.
---

# nemo-retriever-evaluate

Use this skill for repeatable retrieval or QA evaluation, not one-off question
answering.

## Orientation

1. Verify the installed surface: `retriever recall --help`,
   `retriever recall vdb-recall run --help`, `retriever eval export --help`,
   and `retriever eval run --help`.
2. Decide the evaluation type:
   - Recall metrics over labeled query/page data: `retriever recall vdb-recall run`.
   - QA generation and judging: `retriever eval export` plus `retriever eval run`,
     or `retriever eval run --from-env`.
   - End-to-end ingest plus QA: `retriever pipeline run --evaluation-mode qa`.
3. If the installed CLI is absent but this is a source checkout, use
   `uv run --project nemo_retriever retriever ...`. Retry dependency downloads
   before choosing another evaluation validation path.
4. If neither path works, use `nemo-retriever-setup` before debugging evaluation
   behavior.

## References

- `references/EVALUATE.md`: recall, QA export/run, page markdown, config, and
  artifact contracts.
- `PITFALLS.md`: table mismatches, retrieval JSON schema, coverage failures,
  missing LLM extras, and expensive reruns.

## Workflow

1. Identify the corpus/index and ground-truth data. Query CSVs commonly need
   `query,pdf_page` or `query,pdf,page`.
2. For recall:

   ```bash
   retriever recall vdb-recall run \
     --query-csv ./queries.csv \
     --lancedb-uri ./lancedb \
     --table-name nv-ingest \
     --top-k 10
   ```

3. For QA evaluation with reusable retrieval JSON:

   ```bash
   retriever eval export \
     --lancedb-uri ./lancedb \
     --lancedb-table nv-ingest \
     --query-csv ./qa.csv \
     --output ./eval/retrieval.json

   retriever eval run --config ./eval_sweep.yaml
   ```

4. If local HuggingFace query embeddings are required, add
   `--local-query-embed-backend hf`, `--local-hf-cache-dir`, and
   `--local-hf-device` to `eval export`.
5. If `eval export` cannot use the required remote embedding endpoint, build the
   retrieval JSON with the Python `Retriever` API and `write_retrieval_json`
   using explicit `embed_kwargs`.
6. Use `retriever eval build-page-index` when full-page markdown is needed from
   Parquet produced during ingest.
7. Preserve generated artifacts and report exact paths so the user can rerun
   generation/judging without re-querying LanceDB.

## Success Checks

- Recall prints `recall@...` metrics or rich recall output.
- QA eval reports coverage and writes result JSON.
- Retrieval JSON has a top-level `queries` object keyed by ground-truth question
  strings.

## Evaluation Scenarios

- "Measure recall@10 for this LanceDB table and query CSV." Use this skill.
- "Export retrieval JSON and run QA judging with a config." Use this skill.
- "What does the index say about a single question?" Use `nemo-retriever-query`.
