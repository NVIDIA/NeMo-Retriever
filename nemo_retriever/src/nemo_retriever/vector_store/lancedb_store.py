# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from datetime import timedelta
import time

from nv_ingest_client.util.vdb.lancedb import LanceDB
from nemo_retriever.vector_store.lancedb_utils import lancedb_schema
import pandas as pd
import lancedb

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LanceDBConfig:
    """
    Minimal config for writing embeddings into LanceDB.

    This module is intentionally lightweight: it can be used by the text-embedding
    stage (`nemo_retriever.text_embed.stage`) and by the vector-store CLI (`nemo_retriever.vector_store.stage`).
    """

    uri: str = "lancedb"
    table_name: str = "nv-ingest"
    overwrite: bool = True

    # Optional index creation (recommended for recall/search runs).
    create_index: bool = True
    index_type: str = "IVF_HNSW_SQ"
    metric: str = "l2"
    num_partitions: int = 16
    num_sub_vectors: int = 256

    hybrid: bool = False
    fts_language: str = "English"


def ensure_lancedb_table(uri: str, table_name: str) -> None:
    """Ensure the LanceDB URI exists and the target table can be opened."""
    Path(uri).mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(uri=uri)
    try:
        db.open_table(table_name)
        return
    except Exception:
        pass

    import pyarrow as pa  # type: ignore

    schema = lancedb_schema()
    empty = pa.table({field.name: [] for field in schema}, schema=schema)
    db.create_table(table_name, data=empty, schema=schema, mode="create")


def open_lancedb_table_with_retry(
    uri: str,
    table_name: str,
    *,
    retries: int = 3,
    sleep_seconds: float = 2.0,
) -> Any:
    db = lancedb.connect(uri=uri)
    open_err: Exception | None = None
    for _ in range(max(1, retries)):
        try:
            return db.open_table(table_name)
        except Exception as exc:
            open_err = exc
            ensure_lancedb_table(uri, table_name)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    raise RuntimeError(f"Could not open LanceDB table {table_name!r} at {uri!r}") from open_err


def count_lancedb_rows(uri: str, table_name: str) -> int | None:
    try:
        table = open_lancedb_table_with_retry(uri, table_name, retries=1, sleep_seconds=0.0)
        return int(table.count_rows())
    except Exception:
        return None


def estimate_processed_pages(uri: str, table_name: str) -> int | None:
    """Estimate processed pages from unique `(source_id, page_number)` pairs."""
    try:
        table = open_lancedb_table_with_retry(uri, table_name, retries=1, sleep_seconds=0.0)
        df = table.to_pandas()[["source_id", "page_number"]]
        return int(df.dropna(subset=["source_id", "page_number"]).drop_duplicates().shape[0])
    except Exception:
        return count_lancedb_rows(uri, table_name)


def _read_text_embeddings_json_df(path: Path) -> pd.DataFrame:
    """
    Read a `*.text_embeddings.json` file emitted by `nemo_retriever.text_embed.stage`.

    Expected wrapper shape:
      {
        ...,
        "df_records": [ { "document_type": ..., "metadata": {...}, ... }, ... ],
        ...
      }
    """
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        raise ValueError(f"Failed reading JSON {path}: {e}") from e

    if isinstance(obj, dict):
        recs = obj.get("df_records")
        if isinstance(recs, list):
            return pd.DataFrame([r for r in recs if isinstance(r, dict)])
        # Fall back to a single record.
        return pd.DataFrame([obj])

    if isinstance(obj, list):
        return pd.DataFrame([r for r in obj if isinstance(r, dict)])

    return pd.DataFrame([])


def _iter_text_embeddings_json_files(input_dir: Path, *, recursive: bool) -> List[Path]:
    """
    Return sorted list of `*.text_embeddings.json` files.

    The stage5 default naming is: `<input>.text_embeddings.json` (where `<input>` is
    typically a stage4 output filename).
    """
    if recursive:
        files = list(input_dir.rglob("*.text_embeddings.json"))
    else:
        files = list(input_dir.glob("*.text_embeddings.json"))
    return sorted([p for p in files if p.is_file()])


def _parse_metadata_dict(raw_metadata: Any) -> Dict[str, Any]:
    if isinstance(raw_metadata, dict):
        return dict(raw_metadata)
    if not isinstance(raw_metadata, str):
        return {}
    text = raw_metadata.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def _extract_detection_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    pe_num = row.get("page_elements_v3_num_detections")
    if pe_num is not None:
        try:
            out["page_elements_v3_num_detections"] = int(pe_num)
        except Exception:
            pass

    pe_counts = row.get("page_elements_v3_counts_by_label")
    if isinstance(pe_counts, dict):
        normalized_counts: Dict[str, int] = {}
        for label, count in pe_counts.items():
            if count is None:
                continue
            try:
                normalized_counts[str(label)] = int(count)
            except Exception:
                continue
        if normalized_counts:
            out["page_elements_v3_counts_by_label"] = normalized_counts

    for field in ("table", "chart", "infographic"):
        entries = row.get(field)
        if isinstance(entries, list):
            out[f"ocr_{field}_detections"] = len(entries)

    return out


def _build_lancedb_rows_from_df(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transform an embeddings-enriched primitives DataFrame into LanceDB rows.

    Rows include:
      - vector (embedding)
      - pdf_basename
      - page_number
      - pdf_page (basename_page)
      - source_id
      - path
    """
    out: List[Dict[str, Any]] = []

    for row in rows:
        metadata = _parse_metadata_dict(row.get("metadata"))
        if not metadata:
            continue

        embedding = metadata.get("embedding")
        if embedding is None:
            continue

        # Normalize embedding to list[float]
        if not isinstance(embedding, list):
            try:
                embedding = list(embedding)  # type: ignore[arg-type]
            except Exception:
                continue
        metadata.pop("embedding", None)  # Remove embedding from metadata to save space in LanceDB.
        metadata.update(_extract_detection_metadata(row))
        path = str(row.get("path", "") or "")
        source_id = str(metadata.get("source_path") or path)
        page_number = row.get("page_number", -1)
        p = Path(path) if path else None
        filename = p.name if p is not None else ""
        pdf_basename = p.stem if p is not None else ""
        pdf_page = f"{pdf_basename}_{page_number}" if (pdf_basename and page_number >= 0) else ""

        if page_number == -1:
            logger.debug("Unable to determine page number for %s", path)

        out.append(
            {
                "vector": embedding,
                "pdf_page": pdf_page,
                "filename": filename,
                "pdf_basename": pdf_basename,
                "page_number": int(page_number),
                "source": source_id,
                "source_id": source_id,
                "path": path,
                "text": row.get("text", ""),
                "metadata": json.dumps(metadata, ensure_ascii=False, default=str),
            }
        )

    return out


def _infer_vector_dim(rows: Sequence[Dict[str, Any]]) -> int:
    for r in rows:
        v = r.get("vector")
        if isinstance(v, list) and v:
            return int(len(v))
    return 0


def create_lancedb_index(table: Any, *, cfg: LanceDBConfig, text_column: str = "text") -> None:
    """Create vector (IVF_HNSW_SQ) and optionally FTS indices on a LanceDB table."""
    try:
        table.create_index(
            index_type=cfg.index_type,
            metric=cfg.metric,
            num_partitions=int(cfg.num_partitions),
            num_sub_vectors=int(cfg.num_sub_vectors),
            vector_column_name="vector",
        )
    except TypeError:
        table.create_index(vector_column_name="vector")

    if cfg.hybrid:
        try:
            table.create_fts_index(text_column, replace=True, language=cfg.fts_language)
        except Exception:
            logger.warning(
                "FTS index creation failed on column %r; continuing with vector-only search.",
                text_column,
                exc_info=True,
            )

    for index_stub in table.list_indices():
        table.wait_for_index([index_stub.name], timeout=timedelta(seconds=600))


def _write_rows_to_lancedb(rows: Sequence[Dict[str, Any]], *, cfg: LanceDBConfig) -> None:
    if not rows:
        logger.warning("No embeddings rows provided; nothing to write to LanceDB.")
        return

    dim = _infer_vector_dim(rows)
    if dim <= 0:
        raise ValueError("Failed to infer embedding dimension from rows.")

    try:
        import lancedb  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "LanceDB write requested but dependencies are missing. "
            "Install `lancedb` and `pyarrow` in this environment."
        ) from e

    db = lancedb.connect(uri=cfg.uri)

    schema = lancedb_schema(vector_dim=dim)

    mode = "overwrite" if cfg.overwrite else "append"
    table = db.create_table(cfg.table_name, data=list(rows), schema=schema, mode=mode)

    if cfg.create_index:
        create_lancedb_index(table, cfg=cfg)


def write_embeddings_to_lancedb(df_with_embeddings: pd.DataFrame, *, cfg: LanceDBConfig) -> None:
    """
    Write embeddings found in `df_with_embeddings.metadata.embedding` to LanceDB.

    This is used programmatically by `nemo_retriever.text_embed.stage.embed_text_from_primitives_df(...)`.
    """
    rows = _build_lancedb_rows_from_df(df_with_embeddings)
    _write_rows_to_lancedb(rows, cfg=cfg)


def write_text_embeddings_dir_to_lancedb(
    input_dir: Path,
    *,
    cfg: LanceDBConfig,
    recursive: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Read `*.text_embeddings.json` files from `input_dir` and upload their embeddings to LanceDB.
    """
    input_dir = Path(input_dir)
    files = _iter_text_embeddings_json_files(input_dir, recursive=bool(recursive))
    if limit is not None:
        files = files[: int(limit)]

    processed = 0
    skipped = 0
    failed = 0

    lancedb = LanceDB(uri=cfg.uri, table_name=cfg.table_name, overwrite=cfg.overwrite)

    results = []

    for p in files:
        df = _read_text_embeddings_json_df(p)
        rows = df.to_dict(orient="records")
        results.append(rows)

    if not results:
        logger.warning("No *.text_embeddings.json files found in %s; nothing to write.", input_dir)
        return {
            "input_dir": str(input_dir),
            "n_files": 0,
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "lancedb": {"uri": cfg.uri, "table_name": cfg.table_name, "overwrite": cfg.overwrite},
        }

    lancedb.run(results)

    return {
        "input_dir": str(input_dir),
        "n_files": len(files),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        # "rows_written": len(all_rows),
        "lancedb": {"uri": cfg.uri, "table_name": cfg.table_name, "overwrite": cfg.overwrite},
    }


def handle_lancedb(
    rows: Sequence[Dict[str, Any]],
    uri: str,
    table_name: str,
    hybrid: bool = False,
    mode: str = "overwrite",
) -> None:
    """
    Write materialized embedding rows to LanceDB.

    Used by batch/inprocess pipelines after ingestion has produced in-memory row dicts.
    """
    normalized_mode = str(mode or "overwrite").strip().lower()
    if normalized_mode not in {"overwrite", "append"}:
        raise ValueError(f"Unsupported LanceDB mode: {mode!r}")

    lancedb_config = LanceDBConfig(
        uri=uri,
        table_name=table_name,
        hybrid=hybrid,
        overwrite=(normalized_mode == "overwrite"),
    )
    cleaned_rows = _build_lancedb_rows_from_df(list(rows))
    _write_rows_to_lancedb(cleaned_rows, cfg=lancedb_config)
