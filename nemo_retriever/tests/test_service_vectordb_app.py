# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from fastapi.testclient import TestClient

from nemo_retriever.service.vectordb_app import create_vectordb_app


def test_query_empty_index_returns_422(tmp_path) -> None:
    app = create_vectordb_app(
        lancedb_uri=str(tmp_path),
        table_name="test_table",
        embed_endpoint="http://embed.example/v1/embeddings",
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
    )
    with TestClient(app) as client:
        resp = client.post("/v1/query", json={"query": "hello", "top_k": 3})

    assert resp.status_code == 422
    assert "No data has been ingested yet" in resp.json()["detail"]
