# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Human-readable and structured run summaries for evaluation actors."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _fmt_time(seconds: float) -> str:
    """Format *seconds* as ``raw / H:MM:SS.mmm``."""
    ms = int(round(seconds * 1000))
    h, remainder = divmod(ms, 3_600_000)
    m, remainder = divmod(remainder, 60_000)
    s, millis = divmod(remainder, 1000)
    return f"{seconds:.2f}s / {h}:{m:02d}:{s:02d}.{millis:03d}"


def _evaluation_metric_sort_key(item: tuple[str, float]) -> tuple[str, int, str]:
    """Sort metrics like ndcg@1, ndcg@3, ..., recall@1, recall@3, ... ."""
    key, _value = item
    metric_name, sep, suffix = str(key).partition("@")
    if sep:
        try:
            return metric_name, int(suffix), str(key)
        except ValueError:
            pass
    return metric_name, 0, str(key)


def print_run_summary(
    processed_pages: Optional[int],
    input_path: Path,
    vdb_op: str,
    vdb_kwargs: Optional[Dict[str, Any]],
    total_time: float,
    ingest_only_total_time: float,
    ray_dataset_download_total_time: float,
    vdb_upload_total_time: float,
    evaluation_total_time: float = 0.0,
    evaluation_metrics: Optional[Dict[str, float]] = None,
    recall_total_time: float = 0.0,
    recall_metrics: Optional[Dict[str, float]] = None,
    processed_files: Optional[int] = None,
    evaluation_label: str = "Recall",
    evaluation_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Print a human-readable run summary and return all metrics as a dict.

    The returned dict is the authoritative structured representation of every
    metric collected during the run.  Callers should persist it to a JSON file
    so that the harness can read it directly instead of parsing stdout.
    """
    if recall_metrics is None:
        recall_metrics = {}
    if evaluation_metrics is None:
        evaluation_metrics = {}
    pages = processed_pages if processed_pages is not None else 0

    ingest_only_pps = pages / ingest_only_total_time if ingest_only_total_time > 0 else 0
    ingest_write_denom = ingest_only_total_time + vdb_upload_total_time
    ingest_and_vdb_upload_pps = pages / ingest_write_denom if ingest_write_denom > 0 else 0
    recall_qps = pages / recall_total_time if recall_total_time > 0 else 0
    total_pps = pages / total_time if total_time > 0 else 0
    vdb_kwargs = dict(vdb_kwargs or {})

    print(f"===== Run Summary - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC =====")

    print("Run Configuration:")
    print(f"\tInput path: {input_path}")
    print(f"\tVDB op: {vdb_op}")
    if vdb_kwargs:
        print(f"\tVDB kwargs: {json.dumps(vdb_kwargs, default=str, sort_keys=True)}")

    print("Runtimes:")
    if processed_files is not None:
        print(f"\tTotal files processed: {processed_files}")
    print(f"\tTotal pages processed: {pages} from {input_path}")
    print(f"\tIngestion only time: {_fmt_time(ingest_only_total_time)}")
    print(f"\tRay dataset download time: {_fmt_time(ray_dataset_download_total_time)}")
    print(f"\tVDB upload time: {_fmt_time(vdb_upload_total_time)}")
    if recall_total_time > 0:
        print(f"\tRecall time: {_fmt_time(recall_total_time)}")
    if evaluation_total_time > 0:
        print(f"\t{evaluation_label} time: {_fmt_time(evaluation_total_time)}")

    print("PPS:")
    print(f"\tIngestion only PPS: {ingest_only_pps:.2f}")
    print(f"\tIngestion + VDB upload PPS: {ingest_and_vdb_upload_pps:.2f}")
    if recall_total_time > 0:
        print(f"\tRecall QPS: {recall_qps:.2f}")
    print(f"\tTotal - Processed: {pages} pages in {_fmt_time(total_time)} @ {total_pps:.2f} PPS")

    if recall_metrics:
        print("Recall metrics:")
        for k, v in sorted(recall_metrics.items(), key=_evaluation_metric_sort_key):
            print(f"  {k}: {v:.4f}")
    elif not evaluation_metrics:
        print("Recall metrics: skipped (no query CSV configured)")

    if evaluation_metrics:
        print(f"{evaluation_label} metrics:")
        for k, v in sorted(evaluation_metrics.items(), key=_evaluation_metric_sort_key):
            print(f"  {k}: {v:.4f}")

    return {
        "pages": pages,
        "files": processed_files,
        "ingest_secs": round(ingest_only_total_time, 4),
        "pages_per_sec_ingest": round(ingest_only_pps, 4),
        "total_time_secs": round(total_time, 4),
        "total_pps": round(total_pps, 4),
        "ray_dataset_download_secs": round(ray_dataset_download_total_time, 4),
        "vdb_op": str(vdb_op),
        "vdb_kwargs": vdb_kwargs,
        "vdb_upload_secs": round(vdb_upload_total_time, 4),
        "recall_time_secs": round(recall_total_time, 4),
        "evaluation_time_secs": round(evaluation_total_time, 4),
        "evaluation_label": evaluation_label,
        "evaluation_count": evaluation_count,
        "recall_metrics": recall_metrics,
        "evaluation_metrics": evaluation_metrics,
    }
