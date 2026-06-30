# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from nemo_retriever.harness.artifact_writer import ArtifactWriter
from nemo_retriever.harness.contracts import (
    EXIT_EVALUATION_FAILURE,
    EXIT_MISSING_INPUT,
    EXIT_QUERY_FAILURE,
    FailurePayload,
    HarnessRunError,
)
from nemo_retriever.query.workflow import ResolvedQueryPlan
from nemo_retriever.tools.recall.beir import (
    build_beir_run_from_hits,
    compute_beir_metrics,
    load_beir_dataset,
)


def _trec_run_text(run: Mapping[str, Mapping[str, float]]) -> str:
    lines: list[str] = []
    for query_id, docs in run.items():
        ordered = sorted(docs.items(), key=lambda item: (-item[1], item[0]))
        for rank, (doc_id, score) in enumerate(ordered, start=1):
            lines.append(f"{query_id} Q0 {doc_id} {rank} {float(score):.6f} retriever-harness\n")
    return "".join(lines)


def _write_query_result(
    writer: ArtifactWriter,
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
    writer.append_jsonl(
        "query_results.jsonl",
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
    query_plan: ResolvedQueryPlan,
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

    writer.status(status="running", phase="query")
    writer.event("query", "query_start", f"Running {len(dataset.queries)} BEIR queries")
    retriever = query_plan.create_retriever()
    query_kwargs = query_plan.query_kwargs()
    raw_hits: list[list[dict[str, Any]]] = []
    latencies_ms: list[float] = []
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
            writer,
            query_id=query_id,
            query_text=query_text,
            latency_ms=latency_ms,
            hits=hit_dicts,
        )

    writer.status(status="running", phase="evaluate")
    writer.event("evaluate", "evaluate_start", "Computing BEIR metrics")
    try:
        run = build_beir_run_from_hits(dataset.query_ids, raw_hits, doc_id_field=str(doc_id_field))
        metrics = compute_beir_metrics(dataset.qrels, run, ks=tuple(evaluation.get("ks") or (1, 3, 5, 10)))
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
    writer.write_json("beir_metrics.json", metrics)
    writer.write_text("beir_run.trec", _trec_run_text(run))
    return latencies_ms, metrics, len(dataset.queries)
