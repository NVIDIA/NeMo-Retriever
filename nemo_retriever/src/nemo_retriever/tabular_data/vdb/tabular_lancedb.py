# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tabular-data-specific LanceDB operator.

Subclasses :class:`~nemo_retriever.vdb.lancedb.LanceDB` to expose
``label`` and ``database_name`` as **first-class top-level columns** on the
table — the same shape the Postgres tabular ingest already uses. This lets
the tabular retrieval layer filter with plain SQL column equality
(``WHERE label = 'Column' AND database_name = 'dor_prod'``) that runs
identically on LanceDB (DataFusion) and Postgres / pgvector, without
having to encode/decode JSON-nested metadata on either side.

This subclass lives under ``tabular_data/`` (not ``vdb/``) on purpose: it
is a vertical-specific extension and shouldn't pollute the reference
LanceDB operator that external customers consume.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from nemo_retriever.vdb.lancedb import LanceDB


class TabularLanceDB(LanceDB):
    """LanceDB variant for tabular ingest.

    Adds two nullable string columns to the schema — ``label`` (e.g.
    ``"Column"``, ``"Table"``, ``"CustomAnalysis"``) and ``database_name``
    — and populates them from each record's ``content_metadata`` so
    downstream callers can filter on them with plain SQL.

    Everything else (connection, index build, retrieval, ``where``
    predicate forwarding) is inherited unchanged from :class:`LanceDB`.
    """

    def _build_schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field("vector", pa.list_(pa.float32(), int(self.vector_dim))),
                pa.field("text", pa.string()),
                pa.field("metadata", pa.string()),
                pa.field("source", pa.string()),
                pa.field("label", pa.string(), nullable=True),
                pa.field("database_name", pa.string(), nullable=True),
            ]
        )

    def _make_row(self, element, *, embedding, text, content_meta, source_meta) -> dict:
        row = super()._make_row(
            element,
            embedding=embedding,
            text=text,
            content_meta=content_meta,
            source_meta=source_meta,
        )
        if isinstance(content_meta, dict):
            row["label"] = _coerce_str(content_meta.get("label"))
            row["database_name"] = _coerce_str(content_meta.get("database_name"))
        else:
            row["label"] = None
            row["database_name"] = None
        return row


def _coerce_str(value: Any) -> str | None:
    """Return ``value`` as a non-empty string, or ``None``."""
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    text = str(value)
    return text or None


__all__ = ["TabularLanceDB"]
