# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project embedded graph rows into a stable, non-duplicated Parquet schema."""

from __future__ import annotations

import copy
import json
from typing import Any

import pandas as pd
import pyarrow as pa

from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.cpu_operator import CPUOperator


class EmbeddingParquetActor(AbstractOperator, CPUOperator):
    """Create Arrow rows ready for durable Parquet staging and later VDB import."""

    def __init__(
        self,
        *,
        vector_dim: int,
        embedding_column: str = "text_embeddings_1b_v2",
        embedding_model: str,
        embedding_revision: str | None = None,
    ) -> None:
        if int(vector_dim) <= 0:
            raise ValueError("vector_dim must be positive")
        super().__init__(
            vector_dim=int(vector_dim),
            embedding_column=embedding_column,
            embedding_model=embedding_model,
            embedding_revision=embedding_revision,
        )

    @property
    def schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field("id", pa.string(), nullable=False),
                pa.field("document_id", pa.string(), nullable=False),
                pa.field("text", pa.string(), nullable=False),
                pa.field("url", pa.string(), nullable=False),
                pa.field("file_path", pa.string(), nullable=False),
                pa.field("dump", pa.string(), nullable=False),
                pa.field("source_id", pa.string(), nullable=False),
                pa.field("parent_id", pa.string()),
                pa.field("chunk_index", pa.int32()),
                pa.field("chunk_count", pa.int32()),
                pa.field("start_token", pa.int32()),
                pa.field("end_token", pa.int32()),
                pa.field("metadata", pa.string(), nullable=False),
                pa.field("embedding_model", pa.string(), nullable=False),
                pa.field("embedding_revision", pa.string(), nullable=False),
                pa.field("vector", pa.list_(pa.float32(), self.vector_dim), nullable=False),
            ],
            metadata={
                b"nemo_retriever.embedding_model": self.embedding_model.encode("utf-8"),
                b"nemo_retriever.embedding_revision": (self.embedding_revision or "").encode("utf-8"),
                b"nemo_retriever.vector_dim": str(self.vector_dim).encode("ascii"),
            },
        )

    def preprocess(self, data: Any, **kwargs: Any) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, (pa.Table, pa.RecordBatch)):
            return data.to_pandas()
        return pd.DataFrame(data)

    @staticmethod
    def _mapping(value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    def process(self, data: pd.DataFrame, **kwargs: Any) -> pa.Table:
        rows: list[dict[str, Any]] = []
        for row_number, source in enumerate(data.to_dict(orient="records")):
            metadata = self._mapping(source.get("metadata"))
            payload = self._mapping(source.get(self.embedding_column))
            vector = payload.get("embedding")
            if not isinstance(vector, list):
                vector = metadata.get("embedding")
            if not isinstance(vector, list) or len(vector) != self.vector_dim:
                actual_dim = len(vector) if isinstance(vector, list) else 0
                raise ValueError(f"Embedding row {row_number} has dimension {actual_dim}; expected {self.vector_dim}.")

            content_metadata = self._mapping(metadata.get("content_metadata"))
            source_metadata = self._mapping(metadata.get("source_metadata"))
            split = self._mapping(metadata.get("embedding_split"))
            base_id = str(
                source_metadata.get("document_id")
                or content_metadata.get("id")
                or source_metadata.get("source_id")
                or source_metadata.get("url")
                or source_metadata.get("file_path")
                or ""
            )
            row_id = str(split.get("chunk_id") or base_id)
            if not row_id:
                raise ValueError(f"Embedding row {row_number} has no stable identifier.")

            compact_metadata = copy.deepcopy(metadata)
            compact_metadata.pop("embedding", None)
            rows.append(
                {
                    "id": row_id,
                    "document_id": base_id,
                    "text": str(source.get("text") or ""),
                    "url": str(source_metadata.get("url") or ""),
                    "file_path": str(source_metadata.get("file_path") or ""),
                    "dump": str(source_metadata.get("dump") or ""),
                    "source_id": str(source_metadata.get("source_id") or base_id),
                    "parent_id": str(split["parent_id"]) if split.get("parent_id") is not None else None,
                    "chunk_index": int(split["chunk_index"]) if split.get("chunk_index") is not None else None,
                    "chunk_count": int(split["chunk_count"]) if split.get("chunk_count") is not None else None,
                    "start_token": int(split["start_token"]) if split.get("start_token") is not None else None,
                    "end_token": int(split["end_token"]) if split.get("end_token") is not None else None,
                    "metadata": json.dumps(
                        compact_metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
                    ),
                    "embedding_model": self.embedding_model,
                    "embedding_revision": self.embedding_revision or "",
                    "vector": vector,
                }
            )
        return pa.Table.from_pylist(rows, schema=self.schema)

    def postprocess(self, data: pa.Table, **kwargs: Any) -> pa.Table:
        return data
