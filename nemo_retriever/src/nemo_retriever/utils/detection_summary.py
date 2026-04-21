# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared detection summary logic.

Provides a single function that accumulates per-page detection counters from
an iterable of ``(page_key, metadata_dict, row_dict)`` tuples.  Both the
batch pipeline (reading from LanceDB) and inprocess pipeline (reading from
a DataFrame) can produce these tuples, allowing the summary computation to
be shared.
"""

from __future__ import annotations

from datetime import datetime
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def compute_detection_summary(
    rows: Iterable[Tuple[Any, Dict[str, Any], Dict[str, Any]]],
) -> Dict[str, Any]:
    """Compute deduped detection totals from an iterable of page data.

    Each element is ``(page_key, metadata_dict, row_dict)`` where:

    - *page_key* is a hashable value used to deduplicate exploded content rows
      (e.g. ``(source_id, page_number)``).
    - *metadata_dict* is the parsed JSON metadata (may contain counters from the
      LanceDB metadata column or from direct DataFrame columns).
    - *row_dict* is the raw row dict, used as fallback for counters stored as
      top-level DataFrame columns (e.g. ``table``, ``chart`` lists).
    """
    per_page: dict[Any, dict] = {}

    for page_key, meta, raw_row in rows:
        entry = per_page.setdefault(
            page_key,
            {
                "pe": 0,
                "ocr_table": 0,
                "ocr_chart": 0,
                "ocr_infographic": 0,
                "pe_by_label": defaultdict(int),
            },
        )

        pe = _safe_int(meta.get("page_elements_v3_num_detections") or raw_row.get("page_elements_v3_num_detections"))
        entry["pe"] = max(entry["pe"], pe)

        for field, meta_key, col_key in [
            ("ocr_table", "ocr_table_detections", "table"),
            ("ocr_chart", "ocr_chart_detections", "chart"),
            ("ocr_infographic", "ocr_infographic_detections", "infographic"),
        ]:
            val = _safe_int(meta.get(meta_key))
            if val == 0:
                col_val = raw_row.get(col_key)
                if isinstance(col_val, list):
                    val = len(col_val)
            entry[field] = max(entry[field], val)

        label_counts = meta.get("page_elements_v3_counts_by_label") or raw_row.get("page_elements_v3_counts_by_label")
        if isinstance(label_counts, dict):
            for label, count in label_counts.items():
                entry["pe_by_label"][str(label)] = max(
                    entry["pe_by_label"][str(label)],
                    _safe_int(count),
                )

    pe_by_label_totals: dict[str, int] = defaultdict(int)
    pe_total = ocr_table_total = ocr_chart_total = ocr_infographic_total = 0
    for e in per_page.values():
        pe_total += e["pe"]
        ocr_table_total += e["ocr_table"]
        ocr_chart_total += e["ocr_chart"]
        ocr_infographic_total += e["ocr_infographic"]
        for label, count in e["pe_by_label"].items():
            pe_by_label_totals[label] += count

    return {
        "pages_seen": len(per_page),
        "page_elements_v3_total_detections": pe_total,
        "page_elements_v3_counts_by_label": dict(sorted(pe_by_label_totals.items())),
        "ocr_table_total_detections": ocr_table_total,
        "ocr_chart_total_detections": ocr_chart_total,
        "ocr_infographic_total_detections": ocr_infographic_total,
    }


def iter_lancedb_rows(uri: str, table_name: str):
    """Yield ``(page_key, meta, row_dict)`` tuples from a LanceDB table."""
    import lancedb  # type: ignore

    db = lancedb.connect(uri)
    table = db.open_table(table_name)
    df = table.to_pandas()[["source_id", "page_number", "metadata"]]

    for row in df.itertuples(index=False):
        source_id = str(getattr(row, "source_id", "") or "")
        page_number = _safe_int(getattr(row, "page_number", -1), default=-1)
        raw_metadata = getattr(row, "metadata", None)
        meta: dict = {}
        if isinstance(raw_metadata, str) and raw_metadata.strip():
            try:
                parsed = json.loads(raw_metadata)
                if isinstance(parsed, dict):
                    meta = parsed
            except Exception:
                pass
        yield (source_id, page_number), meta, {}


def iter_dataframe_rows(df):
    """Yield ``(page_key, meta, row_dict)`` tuples from a pandas DataFrame."""
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        path = str(row_dict.get("path") or row_dict.get("source_id") or "")
        page_number = _safe_int(row_dict.get("page_number", -1), default=-1)

        meta = row_dict.get("metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        if not isinstance(meta, dict):
            meta = {}

        yield (path, page_number), meta, row_dict


def collect_detection_summary_from_lancedb(uri: str, table_name: str) -> Optional[Dict[str, Any]]:
    """Collect detection summary from a LanceDB table."""
    try:
        return compute_detection_summary(iter_lancedb_rows(uri, table_name))
    except Exception:
        return None


def collect_detection_summary_from_df(df) -> Dict[str, Any]:
    """Collect detection summary from a pandas DataFrame."""
    return compute_detection_summary(iter_dataframe_rows(df))


def print_detection_summary(summary: Optional[Dict[str, Any]]) -> None:
    """Print a detection summary to stdout."""
    if summary is None:
        print("Detection summary: unavailable (could not read metadata).")
        return
    print("\nDetection summary (deduped by source_id/page_number):")
    print(f"  Pages seen: {summary['pages_seen']}")
    print(f"  PageElements v3 total detections: {summary['page_elements_v3_total_detections']}")
    print(f"  OCR table detections: {summary['ocr_table_total_detections']}")
    print(f"  OCR chart detections: {summary['ocr_chart_total_detections']}")
    print(f"  OCR infographic detections: {summary['ocr_infographic_total_detections']}")
    print("  PageElements v3 counts by label:")
    by_label = summary.get("page_elements_v3_counts_by_label") or {}
    if not by_label:
        print("    (none)")
    else:
        for label, count in by_label.items():
            print(f"    {label}: {count}")


def write_detection_summary(path: Path, summary: Optional[Dict[str, Any]]) -> None:
    """Write a detection summary dict to a JSON file."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = summary if summary is not None else {"error": "Detection summary unavailable."}
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def print_pages_per_second(processed_pages: Optional[int], ingest_elapsed_s: float) -> None:
    """Print pages-per-second throughput to stdout."""
    if ingest_elapsed_s <= 0:
        print("Pages/sec: unavailable (ingest elapsed time was non-positive).")
        return
    if processed_pages is None:
        print("Pages/sec: unavailable (could not estimate processed pages). " f"Ingest time: {ingest_elapsed_s:.2f}s")
        return

    pps = processed_pages / ingest_elapsed_s
    print(f"Pages processed: {processed_pages}")
    print(f"Pages/sec (ingest only; excludes Ray startup and recall): {pps:.2f}")


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
    hybrid: bool,
    lancedb_uri: str,
    lancedb_table_name: str,
    total_time: float,
    ingest_only_total_time: float,
    ray_dataset_download_total_time: float,
    lancedb_write_total_time: float,
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
    ingest_write_denom = ingest_only_total_time + lancedb_write_total_time
    ingest_and_lancedb_write_pps = pages / ingest_write_denom if ingest_write_denom > 0 else 0
    recall_qps = pages / recall_total_time if recall_total_time > 0 else 0
    total_pps = pages / total_time if total_time > 0 else 0

    print(f"===== Run Summary - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC =====")

    print("Run Configuration:")
    print(f"\tInput path: {input_path}")
    print(f"\tHybrid: {hybrid}")
    print(f"\tLancedb URI: {lancedb_uri}")
    print(f"\tLancedb Table: {lancedb_table_name}")

    print("Runtimes:")
    if processed_files is not None:
        print(f"\tTotal files processed: {processed_files}")
    print(f"\tTotal pages processed: {pages} from {input_path}")
    print(f"\tIngestion only time: {_fmt_time(ingest_only_total_time)}")
    print(f"\tRay dataset download time: {_fmt_time(ray_dataset_download_total_time)}")
    print(f"\tLanceDB Write Time: {_fmt_time(lancedb_write_total_time)}")
    if recall_total_time > 0:
        print(f"\tRecall time: {_fmt_time(recall_total_time)}")
    if evaluation_total_time > 0:
        print(f"\t{evaluation_label} time: {_fmt_time(evaluation_total_time)}")

    print("PPS:")
    print(f"\tIngestion only PPS: {ingest_only_pps:.2f}")
    print(f"\tIngestion + LanceDB Write PPS: {ingest_and_lancedb_write_pps:.2f}")
    if recall_total_time > 0:
        print(f"\tRecall QPS: {recall_qps:.2f}")
    print(f"\tTotal - Processed: {pages} pages in {_fmt_time(total_time)} @ {total_pps:.2f} PPS")

    if recall_metrics:
        print("Recall metrics:")
        for k, v in sorted(recall_metrics.items(), key=_evaluation_metric_sort_key):
            print(f"  {k}: {v:.4f}")
    else:
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
        "lancedb_write_secs": round(lancedb_write_total_time, 4),
        "recall_time_secs": round(recall_total_time, 4),
        "evaluation_time_secs": round(evaluation_total_time, 4),
        "evaluation_label": evaluation_label,
        "evaluation_count": evaluation_count,
        "recall_metrics": recall_metrics,
        "evaluation_metrics": evaluation_metrics,
    }
