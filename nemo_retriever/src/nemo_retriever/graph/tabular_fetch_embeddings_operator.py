# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph operator: turn ``(tables_df, columns_df)`` into embedding-ready rows."""

from __future__ import annotations

import logging
from typing import Any, Iterable

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.tabular_data.ingestion.model.reserved_words import Labels

logger = logging.getLogger(__name__)


class TabularFetchEmbeddingsOp(AbstractOperator, CPUOperator):
    """Build an embedding-ready DataFrame from ``(tables_df, columns_df)``.

    Expected input: a 2-tuple ``(tables_df, columns_df)``. Both DataFrames
    carry the post-ingest UUIDs of the Table/Column nodes written to Neo4j;
    ``tables_df`` is keyed by ``id`` (table UUID) with at least
    ``table_name``, ``table_schema`` and ``description`` columns, and
    ``columns_df`` carries one row per column with ``id``, ``table_name``,
    ``column_name``, ``data_type``, ``description`` and ``sample_values``.
    Multiple schemas can be concatenated into the same pair — the
    ``table_schema`` column on each table row keeps them distinguishable.

    Output columns: ``text, _embed_modality, path, page_number, metadata``.
    Two row types are produced:

    * one ``Table`` row per table, whose ``text`` joins the table description
      with a compact list of its columns; and
    * one ``Column`` row per column.

    The text templates match the previous Neo4j-derived format, so the rest
    of the pipeline (``_BatchEmbedActor`` → ``IngestVdbOperator``) keeps
    working untouched.
    """

    def __init__(
        self,
        *,
        database_name: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(database_name=database_name, **kwargs)
        self._database_name = database_name

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> pd.DataFrame:
        if not (isinstance(data, tuple) and len(data) == 2):
            raise TypeError(
                f"TabularFetchEmbeddingsOp expected a (tables_df, columns_df) tuple for "
                f"database {self._database_name!r}, got {type(data).__name__}."
            )

        tables_df, columns_df = data
        rows = list(self._build_rows(tables_df, columns_df))
        if rows:
            return pd.DataFrame(rows)
        return pd.DataFrame(columns=["text", "_embed_modality", "path", "page_number", "metadata"])

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def _build_rows(self, tables_df: pd.DataFrame, columns_df: pd.DataFrame) -> Iterable[dict[str, Any]]:
        # Index columns by (schema, table_name) so duplicate table names across
        # different schemas (e.g. two "users" tables in two schemas) don't get
        # their columns merged into one bucket — which would both bloat each
        # table's text and double-emit every column row per duplicate.
        columns_by_table: dict[tuple[str, str], list[Any]] = {}
        for _, col in columns_df.iterrows():
            key = (
                str(col.get("table_schema", "")).lower(),
                str(col.get("table_name", "")).lower(),
            )
            columns_by_table.setdefault(key, []).append(col)

        rows: list[dict[str, Any]] = []
        for _, table in tables_df.iterrows():
            table_id = str(table.get("id", ""))
            table_name = str(table.get("table_name", ""))
            table_description = "" if pd.isna(v := table.get("description")) else str(v).strip()
            schema_name = str(table.get("table_schema", ""))
            columns = columns_by_table.get((schema_name.lower(), table_name.lower()), [])

            table_text = _create_table_text(
                table_name=table_name,
                table_description=table_description,
                columns=columns,
                schema_name=schema_name,
                database_name=self._database_name,
            )
            rows.append(
                _create_row(
                    text=table_text,
                    node_id=table_id,
                    label=Labels.TABLE,
                    name=table_name,
                    schema_name=schema_name,
                    database_name=self._database_name,
                )
            )

            for column in columns:
                column_id = str(column.get("id", ""))
                column_name = str(column.get("column_name", ""))
                data_type = "" if pd.isna(v := column.get("data_type")) else str(v).strip()
                column_description = "" if pd.isna(v := column.get("description")) else str(v).strip()
                sample_values = (column.get("sample_values") or [])[:5]
                column_text = _create_column_text(
                    column_name=column_name,
                    column_description=column_description,
                    data_type=data_type,
                    sample_values=sample_values,
                    schema_name=schema_name,
                    table_name=table_name,
                    database_name=self._database_name,
                )
                rows.append(
                    _create_row(
                        text=column_text,
                        node_id=column_id,
                        label=Labels.COLUMN,
                        name=column_name,
                        schema_name=schema_name,
                        database_name=self._database_name,
                    )
                )
        return rows


# ── Helpers ──────────────────────────────────────────────────────────────────


def _create_table_text(
    *,
    table_name: str,
    table_description: str,
    columns: list[Any],
    schema_name: str,
    database_name: str,
) -> str:
    """Build the embedding text for a Table node.

    Returns just the text string; the caller is responsible for wrapping it
    in an embed-row dict via :func:`_create_row`.
    """
    text = f"database_name: {database_name}" f", schema_name: {schema_name}" f", table_name: {table_name}"
    if table_description:
        text += f", table_description: {table_description}"

    column_pieces: list[str] = []
    for column in columns:
        column_name = column.get("column_name", "")
        data_type = "" if pd.isna(v := column.get("data_type")) else str(v).strip()
        piece = f"{{name: {column_name}, data_type: {data_type}"

        column_description = "" if pd.isna(v := column.get("description")) else str(v).strip()
        if column_description:
            piece += f", description: {column_description}"
        piece += "}"
        column_pieces.append(piece)

    text += f", columns: {','.join(column_pieces)}"
    return text


def _create_column_text(
    *,
    column_name: str,
    column_description: str,
    data_type: str,
    sample_values: list[Any],
    table_name: str,
    schema_name: str,
    database_name: str,
) -> str:
    """Build the embedding text for a Column node.

    Returns just the text string; the caller is responsible for wrapping it
    in an embed-row dict via :func:`_create_row`.
    """
    text = (
        f"database_name: {database_name}"
        f", schema_name: {schema_name}"
        f", table_name: {table_name}"
        f", column_name: {column_name}"
        f", data_type: {data_type}"
    )
    if column_description:
        text += f", column_description: {column_description}"
    if len(sample_values) > 0:
        text += f", sample_values: {', '.join(str(x) for x in sample_values)}"
    return text


def _create_row(
    *,
    text: str,
    node_id: str | None,
    label: str,
    name: str,
    schema_name: str,
    database_name: str,
) -> dict[str, Any]:
    path = f"neo4j:{node_id}" if node_id else "neo4j:unknown"
    # Nest tabular identifiers under content_metadata so they survive the
    # IngestVdbOperator → LanceDB write path (which only persists
    # content_metadata + source_metadata into the table's metadata column).
    # Top-level copies are kept for any in-memory consumer of this DataFrame.
    tabular_fields = {
        "id": node_id,
        "label": label,
        "name": name,
        "source_path": path,
        "schema_name": schema_name,
        "database_name": database_name,
    }
    return {
        "text": text.strip(),
        "_embed_modality": "text",
        "path": path,
        "page_number": -1,
        "metadata": {
            **tabular_fields,
            "content_metadata": dict(tabular_fields),
        },
    }
