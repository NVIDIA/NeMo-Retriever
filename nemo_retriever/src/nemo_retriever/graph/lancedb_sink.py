# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LanceDB Writer — Designer component that writes Ray Data batches to LanceDB.

Reuses the existing schema and row-building logic from
``nemo_retriever.vector_store.lancedb_utils`` so the table layout is
consistent with all other ingest paths.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated, Any

from nemo_retriever.graph.designer import Param, designer_component


@designer_component(
    name="LanceDB Writer",
    category="Data Sinks",
    compute="cpu",
    description="Writes Ray Data batches to a LanceDB table using the standard schema, then builds a vector index",
    category_color="#ff9f43",
    component_type="pipeline_sink",
)
class LanceDBWriterActor:
    """Ray Data ``map_batches`` actor that streams DataFrame batches into LanceDB.

    On init the target directory is cleared, and an empty table is created with
    the canonical schema (from ``lancedb_utils.lancedb_schema``).  Each batch is
    converted to structured rows via ``build_lancedb_rows`` and appended.

    After all batches have been written, call :meth:`finalize` (or rely on the
    generated code) to build the vector index.
    """

    def __init__(
        self,
        uri: Annotated[str, Param(label="LanceDB URI", placeholder="/path/to/lancedb")] = "lancedb",
        table_name: Annotated[str, Param(label="Table Name")] = "nv-ingest",
        overwrite: Annotated[bool, Param(label="Overwrite Table")] = True,
        create_index: Annotated[bool, Param(label="Create Index After Write")] = True,
        hybrid: Annotated[bool, Param(label="Hybrid (dense + FTS)")] = False,
        embedding_column: Annotated[str, Param(label="Embedding Column")] = "text_embeddings_1b_v2",
        embedding_key: Annotated[str, Param(label="Embedding Key")] = "embedding",
        text_column: Annotated[str, Param(label="Text Column")] = "text",
        include_text: Annotated[bool, Param(label="Include Text")] = True,
    ) -> None:
        from nemo_retriever.vector_store.lancedb_utils import lancedb_schema
        import lancedb as _ldb

        self._uri = str(Path(uri).expanduser().resolve())
        self._table_name = table_name
        self._overwrite = overwrite
        self._create_index = create_index
        self._hybrid = hybrid
        self._embedding_column = embedding_column
        self._embedding_key = embedding_key
        self._text_column = text_column
        self._include_text = include_text
        self._total_rows = 0

        ldb_path = Path(self._uri)
        if self._overwrite and ldb_path.is_dir():
            shutil.rmtree(ldb_path)
        ldb_path.mkdir(parents=True, exist_ok=True)

        self._db = _ldb.connect(uri=self._uri)
        self._schema = lancedb_schema(2048)
        mode = "overwrite" if self._overwrite else "create"
        self._table = self._db.create_table(
            self._table_name,
            schema=self._schema,
            mode=mode,
        )

    def __call__(self, batch_df: Any) -> Any:
        from nemo_retriever.vector_store.lancedb_utils import build_lancedb_rows

        rows = build_lancedb_rows(
            batch_df,
            embedding_column=self._embedding_column,
            embedding_key=self._embedding_key,
            text_column=self._text_column,
            include_text=self._include_text,
        )
        if rows:
            if self._table is None:
                self._table = self._db.open_table(self._table_name)
            self._table.add(rows)
            self._total_rows += len(rows)

        return batch_df

    def finalize(self) -> None:
        """Build the vector (and optionally FTS) index after all batches."""
        if not self._create_index or self._table is None:
            return
        from nemo_retriever.vector_store.lancedb_store import LanceDBConfig, create_lancedb_index

        cfg = LanceDBConfig(
            uri=self._uri,
            table_name=self._table_name,
            hybrid=self._hybrid,
        )
        create_lancedb_index(self._table, cfg=cfg)
