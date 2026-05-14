# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph operator: turn a ``{schema_name: Schema}`` dict into embedding-ready rows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.tabular_data.ingestion.model.reserved_words import Labels

if TYPE_CHECKING:
    from nemo_retriever.tabular_data.ingestion.model.schema import Schema


_EMBED_COLUMNS = ["text", "_embed_modality", "path", "page_number", "metadata"]


class TabularFetchEmbeddingsOp(AbstractOperator, CPUOperator):
    """Build an embedding-ready DataFrame from a ``{schema_name: Schema}`` dict.

    Expected input: the dict produced by :class:`TabularSchemaExtractOp`. Each
    :class:`Schema` exposes ``tables_df`` and ``columns_df`` carrying the
    UUIDs of the Table/Column nodes written to Neo4j.

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
        node_ids: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(database_name=database_name, node_ids=node_ids, **kwargs)
        self._database_name = database_name
        # Optional allow-list of Table/Column UUIDs to embed. ``None`` means
        # "embed everything"; a (possibly empty) list restricts output to rows
        self._node_ids: set[str] | None = set(node_ids) if node_ids is not None else None

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> pd.DataFrame:
        if not isinstance(data, dict) or not data:
            return pd.DataFrame(columns=_EMBED_COLUMNS)

        rows: list[dict[str, Any]] = []
        for schema in data.values():
            rows.extend(self._build_rows_for_schema(schema))

        if not rows:
            return pd.DataFrame(columns=_EMBED_COLUMNS)
        return pd.DataFrame(rows, columns=_EMBED_COLUMNS)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def _build_rows_for_schema(self, schema: "Schema") -> Iterable[dict[str, Any]]:
        tables_df = getattr(schema, "tables_df", None)
        columns_df = getattr(schema, "columns_df", None)
        if tables_df is None or columns_df is None or tables_df.empty or columns_df.empty:
            return []

        # Index columns by lowercase table name so each table can pick up its
        # column rows without rescanning the full columns_df.
        columns_by_table: dict[str, list[Any]] = {}
        for _, col in columns_df.iterrows():
            key = str(col.get("table_name", "")).lower()
            columns_by_table.setdefault(key, []).append(col)

        allowed_ids = self._node_ids
        rows: list[dict[str, Any]] = []
        for _, table in tables_df.iterrows():
            table_id = str(table.get("id", ""))
            table_name = str(table.get("table_name", ""))
            table_description = "" if pd.isna(v := table.get("description")) else str(v).strip()
            columns = columns_by_table.get(table_name.lower(), [])
            schema_name = str(table.get("table_schema", ""))

            # Table-row text always references the full column list, even when
            # filtering — otherwise the table embedding would silently change
            # shape based on which columns happen to be in the allow-list.
            if allowed_ids is None or table_id in allowed_ids:
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
                        database_name=self._database_name,
                    )
                )

            for column in columns:
                column_id = str(column.get("id", ""))
                if allowed_ids is not None and column_id not in allowed_ids:
                    continue
                column_name = str(column.get("column_name", ""))
                data_type = "" if pd.isna(v := column.get("data_type")) else str(v).strip()
                column_description = "" if pd.isna(v := column.get("description")) else str(v).strip()
                sample_values = column.get("sample_values") or []
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

    text = f"db_name: {database_name}" f", schema_name: {schema_name}" f", table_name: {table_name}"
    if table_description:
        text += f", table_description: {table_description}"
    text += f", columns: {' '.join(column_pieces)}"
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
        f"db_name: {database_name}"
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
