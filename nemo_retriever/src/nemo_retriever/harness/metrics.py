# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


def _safe_pdf_page_count(path: Path) -> int | None:
    try:
        import pypdfium2 as pdfium  # type: ignore

        doc = pdfium.PdfDocument(str(path))
        try:
            return int(len(doc))
        finally:
            try:
                doc.close()
            except Exception:
                pass
    except Exception:
        return None


def _count_pages(paths: Sequence[str]) -> int | None:
    total = 0
    counted = False
    for raw_path in paths:
        path = Path(raw_path)
        if path.suffix.lower() != ".pdf":
            continue
        count = _safe_pdf_page_count(path)
        if count is None:
            continue
        counted = True
        total += count
    return total if counted else None


def _normalize_metric_key(key: str) -> str:
    return str(key).lower().replace("@", "_").replace("-", "_")


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    value = ordered[lower] * (1 - fraction) + ordered[upper] * fraction
    return round(value, 3)


def build_summary_metrics(
    resolved: dict[str, Any],
    *,
    documents: Sequence[str],
    ingest_summary: Mapping[str, Any] | None = None,
    ingest_secs: float | None = None,
    query_latencies_ms: Sequence[float] = (),
    beir_metrics: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "files": len(documents),
        "pages": _count_pages(documents),
        "rows_processed": None,
        "ingest_secs": ingest_secs,
        "pages_per_sec_ingest": None,
        "query_count": len(query_latencies_ms) if query_latencies_ms else 0,
        "query_latency_p50_ms": _percentile(query_latencies_ms, 0.50),
        "query_latency_p95_ms": _percentile(query_latencies_ms, 0.95),
    }
    if ingest_summary:
        for source_key in ("result_n_rows", "n_rows", "rows_processed", "num_rows", "rows"):
            if source_key in ingest_summary and ingest_summary[source_key] is not None:
                metrics["rows_processed"] = ingest_summary[source_key]
                break
        for source_key in ("processed_pages", "num_pages", "input_pages"):
            if source_key in ingest_summary and ingest_summary[source_key] is not None:
                metrics["pages"] = ingest_summary[source_key]
                break
        if ingest_secs is None:
            for source_key in ("ingest_secs", "ingestion_only_secs", "elapsed_secs"):
                if source_key in ingest_summary and ingest_summary[source_key] is not None:
                    metrics["ingest_secs"] = ingest_summary[source_key]
                    break
    if metrics["pages"] is not None and metrics["ingest_secs"] not in {None, 0, 0.0}:
        try:
            metrics["pages_per_sec_ingest"] = round(float(metrics["pages"]) / float(metrics["ingest_secs"]), 3)
        except (TypeError, ValueError, ZeroDivisionError):
            metrics["pages_per_sec_ingest"] = None
    for key, value in (beir_metrics or {}).items():
        metrics[_normalize_metric_key(key)] = value
    return {key: metrics.get(key) for key in resolved.get("summary_keys", [])}
