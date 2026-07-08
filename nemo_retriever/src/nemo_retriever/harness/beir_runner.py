# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from nemo_retriever.harness.artifact_writer import append_jsonl, ArtifactWriter
from nemo_retriever.harness.contracts import (
    EXIT_EVALUATION_FAILURE,
    EXIT_INVALID,
    EXIT_MISSING_INPUT,
    EXIT_QUERY_FAILURE,
    FailurePayload,
    HarnessRunError,
)
from nemo_retriever.harness.json_io import artifact_write_error, write_json
from nemo_retriever.query.options import QueryRequest
from nemo_retriever.query.workflow import ResolvedQueryPlan
from nemo_retriever.tools.recall.beir import (
    BeirDataset,
    build_beir_run_from_hits,
    build_beir_run_from_ranked_doc_ids,
    compute_beir_metrics,
    load_beir_dataset,
)


def _write_trec_run(path: Path, run: Mapping[str, Mapping[str, float]]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for query_id, docs in run.items():
                ordered = sorted(docs.items(), key=lambda item: (-item[1], item[0]))
                for rank, (doc_id, score) in enumerate(ordered, start=1):
                    handle.write(f"{query_id} Q0 {doc_id} {rank} {float(score):.6f} retriever-harness\n")
    except OSError as exc:
        raise artifact_write_error(exc) from exc


def _write_query_result(
    path: Path,
    *,
    query_id: str,
    query_text: str,
    latency_ms: float,
    hits: Sequence[Mapping[str, Any]],
) -> None:
    ranked_hits = []
    for rank, hit in enumerate(hits, start=1):
        ranked = dict(hit)
        ranked["rank"] = rank
        ranked_hits.append(ranked)
    append_jsonl(
        path,
        {
            "query_id": query_id,
            "query": query_text,
            "latency_ms": round(latency_ms, 3),
            "hits": ranked_hits,
        },
    )


def run_beir_queries(
    writer: ArtifactWriter,
    resolved: dict[str, Any],
    query_plan: ResolvedQueryPlan | None,
    query_request: QueryRequest,
) -> tuple[list[float], dict[str, float], int]:
    evaluation = resolved.get("evaluation") or {}
    loader = evaluation.get("loader")
    dataset_name = evaluation.get("dataset_name")
    doc_id_field = evaluation.get("doc_id_field") or "pdf_basename"
    if not loader:
        raise HarnessRunError(
            EXIT_EVALUATION_FAILURE,
            FailurePayload(
                failed_phase="evaluate",
                failure_reason="evaluation_failed",
                retryable=False,
                message="BEIR evaluation requires evaluation.loader.",
            ),
        )
    ks = tuple(evaluation.get("ks") or (1, 3, 5, 10))

    # Load the queries and qrels once; dense and agentic retrieval score against
    # the same dataset. doc_id_field keys the qrels here and keys the run doc-ids
    # back to them later, for both strategies.
    try:
        dataset = load_beir_dataset(
            str(loader),
            dataset_name=str(dataset_name),
            split=str(evaluation.get("split") or "test"),
            query_language=evaluation.get("query_language"),
            doc_id_field=str(doc_id_field),
        )
    except FileNotFoundError as exc:
        raise HarnessRunError(
            EXIT_MISSING_INPUT,
            FailurePayload(
                failed_phase="query_plan",
                failure_reason="dataset_missing",
                retryable=False,
                message=str(exc),
                debug_artifacts=("resolved_benchmark.json",),
            ),
        ) from exc
    except Exception as exc:
        raise HarnessRunError(
            EXIT_EVALUATION_FAILURE,
            FailurePayload(
                failed_phase="evaluate",
                failure_reason="evaluation_failed",
                retryable=False,
                message=str(exc),
                debug_artifacts=("resolved_benchmark.json",),
            ),
        ) from exc

    # Retrieve. Dense runs a per-query retriever loop; agentic runs the whole
    # dataset as one concurrent ReAct batch. They differ in how they retrieve and
    # how they turn results into a run, so each returns the per-query latencies and
    # a finished run mapping (query_id -> {doc_id: score}) that scores identically.
    writer.status(status="running", phase="query")
    if query_request.agentic.enabled:
        latencies_ms, run = _agentic_retrieve(writer, dataset, query_request, ks=ks, doc_id_field=str(doc_id_field))
    else:
        # Only the dense path uses query_plan; execution.py always resolves one for it.
        assert query_plan is not None, "dense BEIR retrieval requires a resolved query_plan"
        latencies_ms, run = _dense_retrieve(writer, dataset, query_plan, doc_id_field=str(doc_id_field))

    # Score the run and write artifacts -- identical for both retrieval strategies.
    writer.status(status="running", phase="evaluate")
    writer.event("evaluate", "evaluate_start", "Computing BEIR metrics")
    try:
        metrics = compute_beir_metrics(dataset.qrels, run, ks=ks)
    except Exception as exc:
        raise HarnessRunError(
            EXIT_EVALUATION_FAILURE,
            FailurePayload(
                failed_phase="evaluate",
                failure_reason="evaluation_failed",
                retryable=False,
                message=str(exc),
                debug_artifacts=("query_results.jsonl", "run.log"),
            ),
        ) from exc
    write_json(writer.path("beir_metrics.json"), metrics)
    _write_trec_run(writer.path("beir_run.trec"), run)
    return latencies_ms, metrics, len(dataset.queries)


def _dense_retrieve(
    writer: ArtifactWriter,
    dataset: BeirDataset,
    query_plan: ResolvedQueryPlan,
    *,
    doc_id_field: str,
) -> tuple[list[float], dict[str, dict[str, float]]]:
    """Run each query through the dense retriever, recording per-query latency."""
    writer.event("query", "query_start", f"Running {len(dataset.queries)} BEIR queries")
    retriever = query_plan.create_retriever()
    query_kwargs = query_plan.query_kwargs()
    raw_hits: list[list[dict[str, Any]]] = []
    latencies_ms: list[float] = []
    query_results_path = writer.path("query_results.jsonl")
    for query_id, query_text in zip(dataset.query_ids, dataset.queries):
        start = time.perf_counter()
        try:
            hits = retriever.query(query_text, **query_kwargs)
        except Exception as exc:
            raise HarnessRunError(
                EXIT_QUERY_FAILURE,
                FailurePayload(
                    failed_phase="query",
                    failure_reason="query_failed",
                    retryable=False,
                    message=str(exc),
                    debug_artifacts=("query_plan.json", "query_results.jsonl", "run.log"),
                ),
            ) from exc
        latency_ms = (time.perf_counter() - start) * 1000.0
        hit_dicts = [dict(hit) for hit in hits]
        raw_hits.append(hit_dicts)
        latencies_ms.append(latency_ms)
        _write_query_result(
            query_results_path,
            query_id=query_id,
            query_text=query_text,
            latency_ms=latency_ms,
            hits=hit_dicts,
        )
    run = build_beir_run_from_hits(dataset.query_ids, raw_hits, doc_id_field=doc_id_field)
    return latencies_ms, run


def _agentic_retrieve(
    writer: ArtifactWriter,
    dataset: BeirDataset,
    query_request: QueryRequest,
    *,
    ks: tuple[int, ...],
    doc_id_field: str,
) -> tuple[list[float], dict[str, dict[str, float]]]:
    """Run the whole dataset through one concurrent agentic (ReAct) batch.

    The agent targets the deepest BEIR ``k`` so recall at the largest cutoff is
    computable. Because the queries run as a batch, per-query latency is not
    measured individually; the wall-clock total is spread evenly across queries.
    """
    from nemo_retriever.query.agentic import agentic_beir_retrieve
    from nemo_retriever.query.agentic_options import agentic_target_top_k
    from nemo_retriever.query.workflow import build_agentic_config

    try:
        cfg = build_agentic_config(query_request, top_k=agentic_target_top_k("beir", list(ks)))
    except (ValueError, TypeError) as exc:
        # Invalid agentic config (e.g. out-of-range temperature, backend_top_k < top_k)
        # surfaces as a structured harness failure rather than an unhandled exception.
        raise HarnessRunError(
            EXIT_INVALID,
            FailurePayload(
                failed_phase="query_plan",
                failure_reason="invalid_agentic_config",
                retryable=False,
                message=str(exc),
                debug_artifacts=("resolved_benchmark.json",),
            ),
        ) from exc
    writer.event("query", "query_start", f"Running {len(dataset.queries)} agentic BEIR queries")
    start = time.perf_counter()
    try:
        _result, ranked_doc_ids = agentic_beir_retrieve(dataset, cfg, doc_id_field=doc_id_field)
    except Exception as exc:
        raise HarnessRunError(
            EXIT_QUERY_FAILURE,
            FailurePayload(
                failed_phase="query",
                failure_reason="query_failed",
                retryable=False,
                message=str(exc),
                debug_artifacts=("query_plan.json", "query_results.jsonl", "run.log"),
            ),
        ) from exc
    total_ms = (time.perf_counter() - start) * 1000.0

    # The agent runs as a batch, so spread the wall-clock total evenly per query.
    query_count = len(dataset.queries)
    per_query_ms = total_ms / query_count if query_count else 0.0
    query_results_path = writer.path("query_results.jsonl")
    for query_id, query_text, doc_ids in zip(dataset.query_ids, dataset.queries, ranked_doc_ids):
        _write_query_result(
            query_results_path,
            query_id=str(query_id),
            query_text=str(query_text),
            latency_ms=per_query_ms,
            hits=[{"doc_id": doc_id} for doc_id in doc_ids],
        )
    run = build_beir_run_from_ranked_doc_ids(dataset.query_ids, ranked_doc_ids)
    return [per_query_ms] * query_count, run
