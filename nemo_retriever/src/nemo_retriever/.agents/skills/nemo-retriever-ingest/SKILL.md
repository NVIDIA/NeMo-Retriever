---
name: nemo-retriever-ingest
description: Use when the user asks to ingest, index, embed, or make documents searchable with NeMo Retriever, including `retriever ingest`, `retriever pipeline run`, LanceDB creation, extraction outputs, or ingestion validation. Do not use for querying an existing index; use `nemo-retriever-query` instead.
---

# nemo-retriever-ingest

Use this skill to build a searchable NeMo Retriever corpus. It teaches the
current CLI/SDK behavior, the defaults that matter across tasks, and validation
checks that distinguish a real index from a false-positive run.

## Orientation

1. From the active environment, verify the public surface: `retriever --help`,
   then `retriever ingest --help` or `retriever pipeline run --help`.
2. If `retriever` is not on PATH but this is a source checkout, bootstrap the
   CLI with `uv run --project nemo_retriever retriever --help`. If dependencies
   need to download, retry the command and continue from the validated command
   surface.
3. If neither an installed command nor a source checkout is available, this is
   an environment setup blocker, not an ingest failure. Use the setup workflow
   first; do not invent a package name or private command.
4. Choose the simplest ingestion path that satisfies the task:
   - `retriever ingest ...` for one-shot ingest into LanceDB.
   - `retriever pipeline run ...` when the task needs saved Parquet, image
     storage, evaluation mode, service run mode, or lower-level tuning.
   - Python `create_ingestor(...)` when the user explicitly wants SDK code.
5. Record the `lancedb_uri`, table name, run mode, and any remote NIM endpoints
   because query and evaluation tasks must match them exactly.

## References

- `references/INGEST.md`: command choices, defaults, remote inference, SDK notes,
  and validation checks.
- `PITFALLS.md`: install gaps, table mismatches, slow startup, empty corpora,
  model downloads, and stale docs.

## Workflow

1. Identify input paths and supported file types. For directories, expect
   `retriever ingest` to expand supported files; for `pipeline run`, confirm the
   desired `--input-type` when the corpus is not obvious.
2. Decide local versus remote inference before running:
   - Remote NIM inference: set `NVIDIA_API_KEY` when using build.nvidia.com and
     pass explicit `--*-invoke-url`, `--embed-invoke-url`, and model flags.
   - Local inference: confirm the environment has the needed extras, CUDA stack,
     and model cache. Route HuggingFace downloads to `~/models` when preparing a
     new environment.
3. Run ingest with explicit index settings when the index will be reused:

   ```bash
   retriever ingest ./data/corpus --lancedb-uri ./lancedb --table-name nv-ingest
   ```

   From this repo checkout, use:

   ```bash
   uv run --project nemo_retriever retriever ingest ./data/corpus --lancedb-uri ./lancedb --table-name nv-ingest
   ```

4. Use `retriever pipeline run` when saved intermediates matter:

   ```bash
   retriever pipeline run ./data/corpus --input-type pdf --save-intermediate ./processed_docs
   ```

5. Validate the table before declaring success. A zero exit and "Ingested N
   document(s)" are not sufficient:

   ```python
   import lancedb

   db = lancedb.connect("./lancedb")
   print(db.table_names())
   table = db.open_table("nv-ingest")
   print(table.count_rows())
   ```

   Then run a smoke query with the same URI, table, embedding endpoint, and
   model used for ingest.

## Success Checks

- The command reports the target LanceDB URI and table, or the expected Parquet
  directory exists for `--save-intermediate`.
- A focused `retriever query ... --lancedb-uri ... --table-name ...` returns at
  least one JSON hit for a query that should match the corpus.
- The query skill can reuse the recorded URI/table/model settings without
  guessing.

## Evaluation Scenarios

- "Index the PDFs in `data/reports` with NeMo Retriever." Use this skill.
- "Run a tuned batch ingestion and save Parquet for later page markdown." Use
  this skill and prefer `retriever pipeline run`.
- "Answer a question from an existing LanceDB table." Use `nemo-retriever-query`,
  not this skill.
