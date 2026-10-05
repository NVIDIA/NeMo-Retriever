# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pandas as pd
import pyarrow as pa
import pytest

from nemo_retriever.operators.graph_ops.embedding_parquet_operator import EmbeddingParquetActor


def _row(*, split: bool = False):
    metadata = {
        "content_metadata": {"type": "text", "id": "doc-1"},
        "source_metadata": {
            "source_id": "https://example.test/1",
            "url": "https://example.test/1",
            "file_path": "s3://bucket/file",
            "document_id": "doc-1",
            "dump": "CC-MAIN-2026-30",
        },
        "embedding": [1.0, 2.0],
    }
    if split:
        metadata["embedding_split"] = {
            "content": "child",
            "parent_id": "parent-1",
            "chunk_id": "chunk-1",
            "chunk_index": 1,
            "chunk_count": 3,
            "start_token": 8,
            "end_token": 16,
        }
    return {
        "text": "child" if split else "document",
        "metadata": metadata,
        "text_embeddings_1b_v2": {"embedding": [1.0, 2.0], "info_msg": None},
    }


def test_embedding_parquet_actor_emits_fixed_vector_without_metadata_duplicate():
    actor = EmbeddingParquetActor(vector_dim=2, embedding_model="model", embedding_revision="revision")
    result = actor.run(pd.DataFrame([_row()]))

    assert isinstance(result, pa.Table)
    assert result.schema.field("vector").type == pa.list_(pa.float32(), 2)
    assert result.column("id").to_pylist() == ["doc-1"]
    assert result.column("url").to_pylist() == ["https://example.test/1"]
    assert result.column("vector").to_pylist() == [[1.0, 2.0]]
    assert "embedding" not in json.loads(result.column("metadata")[0].as_py())


def test_embedding_parquet_actor_uses_split_chunk_identity_and_lineage():
    actor = EmbeddingParquetActor(vector_dim=2, embedding_model="model")
    result = actor.run(pd.DataFrame([_row(split=True)]))

    assert result.column("id").to_pylist() == ["chunk-1"]
    assert result.column("document_id").to_pylist() == ["doc-1"]
    assert result.column("parent_id").to_pylist() == ["parent-1"]
    assert result.column("chunk_index").to_pylist() == [1]


def test_embedding_parquet_actor_rejects_wrong_vector_dimension():
    actor = EmbeddingParquetActor(vector_dim=3, embedding_model="model")
    with pytest.raises(ValueError, match="dimension 2; expected 3"):
        actor.run(pd.DataFrame([_row()]))
