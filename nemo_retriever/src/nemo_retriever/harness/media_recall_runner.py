# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audio/video segment-recall execution for the artifact-first harness."""

from __future__ import annotations

import csv
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from nemo_retriever.harness.artifact_writer import ArtifactWriter, append_jsonl
from nemo_retriever.harness.contracts import (
    EXIT_EVALUATION_FAILURE,
    EXIT_MISSING_INPUT,
    EXIT_QUERY_FAILURE,
    FailurePayload,
    HarnessRunError,
)
from nemo_retriever.harness.json_io import write_json
from nemo_retriever.query.options import QueryRequest, ServiceQueryRequest
from nemo_retriever.query.service import query_documents as query_service_documents
from nemo_retriever.query.workflow import ResolvedQueryPlan
from nemo_retriever.tools.recall.core import (
    _hit_to_audio_segment_key,
    _normalize_audio_media_id,
    is_hit_at_k,
)


@dataclass(frozen=True)
class MediaRecallDataset:
    query_ids: list[str]
    queries: list[str]
    golden_answers: list[str]


def _media_id(value: str) -> str:
    # Reuse the established audio/video matcher normalization so video source
    # paths and both historical ground-truth CSV shapes have identical IDs.
    return _normalize_audio_media_id(value)


def load_media_recall_dataset(path: str | Path) -> MediaRecallDataset:
    """Load either supported segment-recall CSV shape into a common dataset."""
    csv_path = Path(path).expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Media recall query file does not exist: {csv_path}")
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise FileNotFoundError(f"Could not read media recall query file: {csv_path}") from exc

    query_ids: list[str] = []
    queries: list[str] = []
    golden_answers: list[str] = []
    for index, row in enumerate(rows):
        query = str(row.get("query") or row.get("question") or "").strip()
        media = str(row.get("expected_media_id") or row.get("name") or "").strip()
        raw_start = row.get("expected_start_time", row.get("start_time"))
        raw_end = row.get("expected_end_time", row.get("end_time"))
        if not query or not media or raw_start is None or raw_end is None:
            raise ValueError(f"Media recall row {index + 2} in {csv_path} is missing query or segment metadata")
        try:
            start, end = float(raw_start), float(raw_end)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Media recall row {index + 2} in {csv_path} has invalid segment metadata") from exc
        if end < start:
            raise ValueError(f"Media recall row {index + 2} in {csv_path} ends before it starts")
        query_ids.append(str(index))
        queries.append(query)
        golden_answers.append(f"{_media_id(media)}\t{start:.6f}\t{end:.6f}")

    if not query_ids:
        raise ValueError(f"Media recall query file has no rows: {csv_path}")
    return MediaRecallDataset(query_ids=query_ids, queries=queries, golden_answers=golden_answers)


def _write_query_result(
    path: Path,
    *,
    query_id: str,
    query_text: str,
    latency_ms: float,
    hits: Sequence[Mapping[str, Any]],
) -> None:
    append_jsonl(
        path,
        {
            "query_id": query_id,
            "query": query_text,
            "latency_ms": round(latency_ms, 3),
            "hits": [{**dict(hit), "rank": rank} for rank, hit in enumerate(hits, start=1)],
        },
    )


def _score(
    dataset: MediaRecallDataset,
    raw_hits: Sequence[Sequence[Mapping[str, Any]]],
    ks: Sequence[int],
    *,
    audio_match_tolerance_secs: float,
) -> dict[str, float]:
    keys = [[key for hit in hits if (key := _hit_to_audio_segment_key(dict(hit))) is not None] for hits in raw_hits]
    return {
        f"recall@{int(k)}": sum(
            is_hit_at_k(
                golden,
                retrieved,
                int(k),
                match_mode="audio_segment",
                audio_match_tolerance_secs=audio_match_tolerance_secs,
            )
            for golden, retrieved in zip(dataset.golden_answers, keys, strict=True)
        )
        / len(dataset.golden_answers)
        for k in ks
    }


def _run_queries(
    writer: ArtifactWriter,
    dataset: MediaRecallDataset,
    query: Callable[[str], Sequence[Mapping[str, Any]]],
) -> tuple[list[float], list[list[dict[str, Any]]]]:
    writer.status(status="running", phase="query")
    writer.event("query", "query_start", f"Running {len(dataset.queries)} media recall queries")
    query_results_path = writer.path("query_results.jsonl")
    latencies_ms: list[float] = []
    raw_hits: list[list[dict[str, Any]]] = []
    for query_id, query_text in zip(dataset.query_ids, dataset.queries, strict=True):
        start = time.perf_counter()
        try:
            hits = query(query_text)
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
        latencies_ms.append(latency_ms)
        raw_hits.append(hit_dicts)
        _write_query_result(
            query_results_path,
            query_id=query_id,
            query_text=query_text,
            latency_ms=latency_ms,
            hits=hit_dicts,
        )
    return latencies_ms, raw_hits


def _run_media_recall(
    writer: ArtifactWriter,
    resolved: dict[str, Any],
    query: Callable[[str], Sequence[Mapping[str, Any]]],
) -> tuple[list[float], dict[str, float], int]:
    evaluation = resolved.get("evaluation") or {}
    dataset_name = evaluation.get("dataset_name") or (resolved.get("dataset") or {}).get("query_file")
    if not dataset_name:
        raise HarnessRunError(
            EXIT_EVALUATION_FAILURE,
            FailurePayload(
                failed_phase="evaluate",
                failure_reason="evaluation_failed",
                retryable=False,
                message="Media recall evaluation requires evaluation.dataset_name.",
            ),
        )
    try:
        dataset = load_media_recall_dataset(str(dataset_name))
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
    except ValueError as exc:
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

    latencies_ms, raw_hits = _run_queries(writer, dataset, query)
    writer.status(status="running", phase="evaluate")
    writer.event("evaluate", "evaluate_start", "Computing media recall metrics")
    try:
        metrics = _score(
            dataset,
            raw_hits,
            tuple(evaluation.get("ks") or (1, 3, 5, 10)),
            audio_match_tolerance_secs=float(evaluation.get("audio_match_tolerance_secs", 2.0)),
        )
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
    write_json(writer.path("media_recall_metrics.json"), metrics)
    return latencies_ms, metrics, len(dataset.queries)


def run_media_recall_queries(
    writer: ArtifactWriter,
    resolved: dict[str, Any],
    query_plan: ResolvedQueryPlan,
    _query_request: QueryRequest,
) -> tuple[list[float], dict[str, float], int]:
    """Run media segment recall against an in-process or batch Retriever."""
    retriever = query_plan.create_retriever()
    query_kwargs = query_plan.query_kwargs()
    return _run_media_recall(
        writer,
        resolved,
        lambda query_text: retriever.query(query_text, **query_kwargs),
    )


def run_service_media_recall_queries(
    writer: ArtifactWriter,
    resolved: dict[str, Any],
    query_request: ServiceQueryRequest,
) -> tuple[list[float], dict[str, float], int]:
    """Run media segment recall against a deployed Retriever service."""
    return _run_media_recall(
        writer,
        resolved,
        lambda query_text: query_service_documents(replace(query_request, query=query_text)),
    )
