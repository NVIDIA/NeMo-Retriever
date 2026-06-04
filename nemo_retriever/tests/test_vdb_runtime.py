# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from nemo_retriever.vdb.runtime import describe_vdb_runtime


def test_describe_vdb_runtime_reports_dense_lancedb_defaults() -> None:
    summary = describe_vdb_runtime("lancedb", {"uri": Path("/tmp/kb")})

    assert summary["op"] == "lancedb"
    assert summary["target"] == {"uri": "/tmp/kb", "table_name": "nv-ingest"}
    assert summary["retrieval"] == {
        "mode": "dense",
        "signals": ["dense_vector"],
        "uses_query_texts": False,
    }
    assert summary["config"] == {"uri": "/tmp/kb"}


def test_describe_vdb_runtime_preserves_backend_collection_target() -> None:
    summary = describe_vdb_runtime(
        "custom",
        {"uri": "vdb://cluster", "table_name": "docs", "index_name": "semantic"},
    )

    assert summary["target"] == {
        "uri": "vdb://cluster",
        "table_name": "docs",
        "collection_name": "semantic",
    }


def test_describe_vdb_runtime_reports_hybrid_mode_and_sanitizes_config() -> None:
    summary = describe_vdb_runtime(
        "lancedb",
        {
            "uri": "./kb",
            "table_name": "docs",
            "hybrid": True,
            "refine_factor": 50,
            "n_probe": 64,
            "api_key": "secret-value",
            "query_texts": ["do not persist"],
            "search_kwargs": {"query_type": "hybrid", "token": "also-secret"},
        },
    )

    assert summary["target"] == {"uri": "./kb", "table_name": "docs"}
    assert summary["retrieval"]["mode"] == "hybrid"
    assert summary["retrieval"]["signals"] == ["dense_vector", "lexical_text"]
    assert summary["retrieval"]["uses_query_texts"] is True
    assert summary["retrieval"]["refine_factor"] == 50
    assert summary["retrieval"]["nprobes"] == 64
    assert summary["retrieval"]["search_kwargs"] == {
        "query_type": "hybrid",
        "token": "<redacted>",
    }
    assert summary["config"]["api_key"] == "<redacted>"
    assert "query_texts" not in summary["config"]
