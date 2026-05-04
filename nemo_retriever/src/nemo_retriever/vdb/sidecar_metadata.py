# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sidecar document metadata merged into each chunk's ``content_metadata``.

Parity with ``nv_ingest_client`` ingestion: ``meta_dataframe``, ``meta_source_field``,
and ``meta_fields`` attach user columns to ``content_metadata`` so LanceDB (and
Milvus-style filters) can query fields like ``meta_a``, ``meta_b``, etc.
"""

from __future__ import annotations

import ast
import json
import logging
import os
from pathlib import Path
from typing import Any, Literal, Mapping

import pandas as pd

logger = logging.getLogger(__name__)

MetaJoinKey = Literal["auto", "source_id", "source_name"]


def normalize_sidecar_cell_value(field: Any) -> Any:
    """Normalize a dataframe cell for JSON / vector DB storage (matches client behavior)."""
    if field is None:
        return None
    if pd.isna(field):
        return None
    if isinstance(field, str):
        if field == "":
            return None
        if field.startswith("["):
            try:
                return ast.literal_eval(field)
            except (SyntaxError, ValueError):
                return field
        return field
    try:
        import numpy as np

        if isinstance(field, (np.int32, np.int64)):
            return int(field)
        if isinstance(field, (np.float32, np.float64)):
            return float(field)
        if isinstance(field, np.bool_):
            return bool(field)
    except ImportError:
        pass
    return field


def _canonical_lookup_key(value: str) -> str:
    s = (value or "").strip()
    if not s:
        return ""
    try:
        return os.path.normpath(os.path.abspath(s))
    except Exception:
        return os.path.normpath(s)


def load_meta_dataframe(meta_dataframe: str | os.PathLike[str] | pd.DataFrame) -> pd.DataFrame:
    """Load a metadata table from a path (csv/json/parquet) or return a copy of a DataFrame."""
    if isinstance(meta_dataframe, pd.DataFrame):
        return meta_dataframe.copy()
    path = Path(meta_dataframe)
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in {".json", ".ndjson"}:
        return pd.read_json(path)
    if suf in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported metadata file type: {path} (use .csv, .json, or .parquet)")


def _build_lookup(
    meta_df: pd.DataFrame,
    meta_source_field: str,
    meta_fields: list[str],
) -> dict[str, dict[str, Any]]:
    if meta_source_field not in meta_df.columns:
        raise ValueError(f"meta_source_field {meta_source_field!r} not in metadata columns: {list(meta_df.columns)}")
    missing = [c for c in meta_fields if c not in meta_df.columns]
    if missing:
        raise ValueError(f"meta_fields missing from metadata columns: {missing}")

    lookup: dict[str, dict[str, Any]] = {}
    dup_logged = False
    for _, row in meta_df.iterrows():
        raw_key = row[meta_source_field]
        if pd.isna(raw_key):
            continue
        key = _canonical_lookup_key(str(raw_key))
        if not key:
            continue
        payload = {c: row[c] for c in meta_fields}
        if key in lookup and not dup_logged:
            logger.info("Duplicate metadata row for key %s; using first match (same as nv-ingest-client).", key)
            dup_logged = True
        if key not in lookup:
            lookup[key] = payload
    return lookup


def _resolve_join_keys(source_metadata: Mapping[str, Any], join_key: MetaJoinKey) -> list[str]:
    out: list[str] = []
    if join_key == "auto":
        for k in (source_metadata.get("source_id"), source_metadata.get("source_name")):
            if isinstance(k, str) and k.strip():
                ck = _canonical_lookup_key(k)
                if ck and ck not in out:
                    out.append(ck)
    elif join_key == "source_id":
        k = source_metadata.get("source_id")
        if isinstance(k, str) and k.strip():
            out.append(_canonical_lookup_key(k))
    else:
        k = source_metadata.get("source_name")
        if isinstance(k, str) and k.strip():
            out.append(_canonical_lookup_key(k))
    return out


def apply_sidecar_metadata_to_client_batches(
    batches: list[list[dict[str, Any]]],
    *,
    meta_df: pd.DataFrame,
    meta_source_field: str,
    meta_fields: list[str],
    join_key: MetaJoinKey = "auto",
) -> list[list[dict[str, Any]]]:
    """For each NV-Ingest record, merge sidecar columns into ``metadata['content_metadata']``."""
    lookup = _build_lookup(meta_df, meta_source_field, meta_fields)
    new_batches: list[list[dict[str, Any]]] = []
    for batch in batches:
        new_batch: list[dict[str, Any]] = []
        for element in batch:
            if not isinstance(element, dict):
                continue
            meta = element.get("metadata")
            if not isinstance(meta, dict):
                new_batch.append(element)
                continue
            src = meta.get("source_metadata")
            if not isinstance(src, dict):
                new_batch.append(element)
                continue
            row = None
            for jk in _resolve_join_keys(src, join_key):
                row = lookup.get(jk)
                if row is not None:
                    break
            if row is None:
                new_batch.append(element)
                continue
            content_meta = meta.get("content_metadata")
            if not isinstance(content_meta, dict):
                content_meta = {}
                meta["content_metadata"] = content_meta
            for col in meta_fields:
                content_meta[col] = normalize_sidecar_cell_value(row.get(col))
            new_batch.append(element)
        new_batches.append(new_batch)
    return new_batches


def split_sidecar_from_vdb_kwargs(
    vdb_kwargs: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Remove sidecar keys from ``vdb_kwargs`` for the concrete VDB constructor.

    Returns
    -------
    (clean_kwargs, sidecar_spec_or_none)

    ``sidecar_spec`` contains ``meta_dataframe`` (path, or DataFrame), ``meta_source_field``,
    ``meta_fields`` (list[str]), and ``meta_join_key`` (str).
    """
    raw = dict(vdb_kwargs or {})
    meta_df = raw.pop("meta_dataframe", None)
    meta_sf = raw.pop("meta_source_field", None)
    meta_fields = raw.pop("meta_fields", None)
    meta_join = raw.pop("meta_join_key", "auto")
    if meta_join not in ("auto", "source_id", "source_name"):
        raise ValueError("meta_join_key must be one of: auto, source_id, source_name")

    present = sum(1 for x in (meta_df, meta_sf, meta_fields) if x is not None)
    if present == 0:
        return raw, None
    if present != 3:
        raise ValueError(
            "Sidecar metadata requires all of: meta_dataframe, meta_source_field, meta_fields "
            "(matching nv-ingest-client / metadata_and_filtered_search.ipynb)."
        )
    if isinstance(meta_fields, str):
        meta_fields = [x.strip() for x in meta_fields.split(",") if x.strip()]
    if not isinstance(meta_fields, list) or not meta_fields:
        raise ValueError("meta_fields must be a non-empty list of column names.")

    return raw, {
        "meta_dataframe": meta_df,
        "meta_source_field": str(meta_sf),
        "meta_fields": [str(x) for x in meta_fields],
        "meta_join_key": meta_join,
    }


def materialize_sidecar_dataframe(spec: dict[str, Any]) -> pd.DataFrame:
    """Load ``meta_dataframe`` from path or validate an in-memory DataFrame."""
    mdf = spec["meta_dataframe"]
    if isinstance(mdf, pd.DataFrame):
        return mdf
    return load_meta_dataframe(mdf)


def parse_hit_content_metadata(hit: Mapping[str, Any]) -> dict[str, Any]:
    """Parse ``hit['metadata']`` JSON string into a dict (post-retrieval helper)."""
    raw = hit.get("metadata")
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def filter_hits_by_content_metadata(
    hits: list[dict[str, Any]],
    predicate,
) -> list[dict[str, Any]]:
    """Keep hits where ``predicate(content_metadata_dict)`` is true."""
    out: list[dict[str, Any]] = []
    for h in hits:
        cm = parse_hit_content_metadata(h)
        try:
            if predicate(cm):
                out.append(h)
        except Exception:
            continue
    return out
