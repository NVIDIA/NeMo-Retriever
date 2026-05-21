# Evaluate Pitfalls

## Missing Evaluation Surface

If `retriever recall` or `retriever eval` is unavailable, first try the
repo-local CLI when a source checkout exists:

```bash
uv run --project nemo_retriever retriever eval --help
```

Retry dependency downloads before choosing another validation path.

## Table Name Drift

Root CLI ingest defaults to table `nv-ingest`. Some evaluation docs and older
graph-pipeline examples mention `nemo-retriever`. Export/recall must point at
the table that was actually written.

## Retrieval JSON Contract

`retriever eval run` in file mode needs a retrieval JSON whose top-level
`queries` object maps each ground-truth question string to retrieved `chunks`.
If query strings differ from the ground truth loader's normalization, coverage
will drop even if retrieval quality is good.

## Coverage Failures

`retriever eval run` checks retrieval coverage before generation. Low coverage
usually means the retrieval JSON and QA dataset keys do not align, the wrong
dataset loader was selected, or the wrong table/query CSV was used.

## LLM Extras And Keys

QA eval needs the `[llm]` extra and generator/judge API configuration. Missing
`litellm`, `NVIDIA_API_KEY`, `GEN_API_KEY`, or `JUDGE_API_KEY` should be reported
as setup gaps, not retrieval failures.

## Eval Export Remote Endpoint Gap

`retriever eval export` supports local-HF query embedding with
`--local-query-embed-backend hf`, `--local-hf-cache-dir`, and
`--local-hf-device`, but it still does not expose `--embed-invoke-url` /
`--embedding-http-endpoint` for remote/self-hosted embedding services. If export
must use a remote endpoint, use `retriever recall ... --embedding-http-endpoint ...`
for recall metrics or build the retrieval JSON with the Python `Retriever` API
and explicit `embed_kwargs`.

## Requery Cost

Do not re-ingest or re-query LanceDB when changing only generator/judge models.
Save and reuse retrieval JSON whenever the retrieval stage is unchanged.
