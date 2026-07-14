# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LanceDB table capability inspection for query routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pyarrow as pa

from nemo_retriever.common.vdb.lancedb_metadata import read_index_metadata

LanceRetrievalMode = Literal["dense", "hybrid", "sparse", "unknown"]

_RETRIEVAL_MODES: dict[str, LanceRetrievalMode] = {
    "dense": "dense",
    "hybrid": "hybrid",
    "sparse": "sparse",
}


@dataclass(frozen=True)
class LanceTableCapabilities:
    has_vector: bool
    has_fts: bool
    retrieval_mode: LanceRetrievalMode
    vector_column: str | None
    text_column: str | None
    index_format_version: str | None = None
    producer_version: str | None = None
    embedding_model_name: str | None = None


def _table_schema(table: Any) -> pa.Schema:
    schema = table.schema
    return schema() if callable(schema) else schema


def _metadata_retrieval_mode(schema: pa.Schema) -> LanceRetrievalMode | None:
    value = read_index_metadata(schema).retrieval_mode
    if value is not None:
        normalized = value.lower()
        if normalized in _RETRIEVAL_MODES:
            return _RETRIEVAL_MODES[normalized]
    return None


def _is_vector_type(data_type: pa.DataType) -> bool:
    is_list_type = (
        pa.types.is_list(data_type)
        or pa.types.is_large_list(data_type)
        or getattr(pa.types, "is_fixed_size_list", lambda _type: False)(data_type)
    )
    if not is_list_type:
        return False
    value_type = data_type.value_type
    return pa.types.is_floating(value_type) or pa.types.is_integer(value_type)


def _detect_vector_column(schema: pa.Schema) -> str | None:
    vector_fields = [field.name for field in schema if _is_vector_type(field.type)]
    if "vector" in vector_fields:
        return "vector"
    return vector_fields[0] if vector_fields else None


def _detect_text_column(schema: pa.Schema, fts_columns: list[str]) -> str | None:
    schema_names = set(schema.names)
    for column in fts_columns:
        if column in schema_names:
            return column
    if "text" in schema_names:
        return "text"
    for field in schema:
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
            return field.name
    return None


def _index_columns(index: Any) -> list[str]:
    columns = getattr(index, "columns", None)
    if columns is None:
        return []
    if isinstance(columns, str):
        return [columns]
    try:
        return [str(column) for column in columns]
    except TypeError:
        return []


def _detect_fts_columns(table: Any) -> list[str]:
    list_indices = getattr(table, "list_indices", None)
    if not callable(list_indices):
        return []

    fts_columns: list[str] = []
    for index in list_indices():
        index_type = str(getattr(index, "index_type", "") or "").strip().lower()
        index_repr = str(index).lower()
        if index_type == "fts" or "index(fts" in index_repr or " fts" in index_repr:
            fts_columns.extend(_index_columns(index))
    return list(dict.fromkeys(column for column in fts_columns if column))


def _mode_from_capabilities(has_vector: bool, has_fts: bool) -> LanceRetrievalMode:
    if has_vector and has_fts:
        return "hybrid"
    if has_vector:
        return "dense"
    if has_fts:
        return "sparse"
    return "unknown"


def inspect_lancedb_table(uri: str, table_name: str) -> LanceTableCapabilities:
    import lancedb  # type: ignore

    table = lancedb.connect(uri).open_table(table_name)
    return inspect_lancedb_table_object(table)


def inspect_lancedb_table_object(table: Any) -> LanceTableCapabilities:
    schema = _table_schema(table)
    index_metadata = read_index_metadata(schema)
    fts_columns = _detect_fts_columns(table)
    vector_column = _detect_vector_column(schema)
    text_column = _detect_text_column(schema, fts_columns)
    has_vector = vector_column is not None
    has_fts = bool(fts_columns)

    retrieval_mode = _metadata_retrieval_mode(schema) or _mode_from_capabilities(has_vector, has_fts)

    return LanceTableCapabilities(
        has_vector=has_vector,
        has_fts=has_fts,
        retrieval_mode=retrieval_mode,
        vector_column=vector_column,
        text_column=text_column,
        index_format_version=index_metadata.index_format_version,
        producer_version=index_metadata.producer_version,
        embedding_model_name=index_metadata.embedding_model_name,
    )
