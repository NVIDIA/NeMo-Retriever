# Ingest Pitfalls

## No Installed Surface

First check the installed user surface:

```bash
retriever --help
python -c "import importlib.util; print(importlib.util.find_spec('nemo_retriever'))"
```

If `retriever` is missing and this is explicitly a source checkout, use the
developer fallback:

```bash
uv run --project nemo_retriever retriever --help
```

If that fails because dependencies need to download, retry the command.

Only report the environment as missing after checking the installed command and,
when a source checkout is actually available, the repo-local fallback.

If there is no installed command, no importable `nemo_retriever` package, and no
source checkout, switch to environment setup. Do not guess alternate command
names such as `nemo-retriever` or proceed with a custom RAG implementation.

## False-Positive Ingest Output

`retriever ingest` can exit zero and print `Ingested N document(s)` even when no
uploadable LanceDB rows were produced. Always validate with `db.table_names()`,
`table.count_rows()`, and a smoke query before reporting success.

## TXT Requires Transformers

TXT ingestion uses a HuggingFace tokenizer. In a lean environment, missing
`transformers` can lead to empty extraction or direct
`ModuleNotFoundError: No module named 'transformers'`. Recover by installing or
transiently adding it. Installed environment:

```bash
uv pip install transformers
retriever ingest ./docs/*.txt --input-type txt ...
```

Source checkout:

```bash
uv run --project nemo_retriever --with transformers retriever ingest ./docs/*.txt --input-type txt ...
```

## Table Defaults Drift

The root CLI defaults are `--lancedb-uri lancedb` and `--table-name nv-ingest`.
Some older docs and examples mention `nemo-retriever`. Always match the table
that was actually written.

## Overwrite Is Default

`retriever ingest` overwrites the target table unless `--append` is passed. Do
not append on reruns unless duplicates are acceptable.

## First Run Can Be Slow

Local GPU model loading, CUDA graph capture, Ray startup, and first-time model
downloads can dominate the first run. This is not automatically a failed ingest.
Look for a non-zero exit or explicit validation error.

## Remote Endpoints Need Matching Query Settings

If ingest used `--embed-invoke-url` or a non-default `--embed-model-name`, query
and evaluation must use the same endpoint/model pair. Mismatched embeddings can
return empty or irrelevant hits.

## Single-PDF And Tiny Corpora

Tiny LanceDB tables can emit partition/index warnings or produce weak nearest
neighbors for unrelated queries. Validate with a query that should be present in
the corpus and inspect rows before tuning thresholds.

## Stale Documentation

If a flag from docs is rejected, run the command-specific `--help` and adapt to
the installed CLI. Teach the mismatch in your final answer rather than hiding it.
