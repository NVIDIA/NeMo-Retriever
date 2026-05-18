# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from nemo_retriever.vdb.records import _normalize_hit, normalize_retrieval_results


def test_normalize_hit_decodes_json_metadata_string() -> None:
    """LanceDB stores metadata as a JSON string; the hit must surface it as a dict."""
    content_meta = {"type": "text", "page_number": 3}
    hit = {
        "entity": {
            "text": "hello",
            "metadata": json.dumps(content_meta),
            "source": json.dumps({"source_id": "/tmp/doc.pdf"}),
        },
        "_distance": 0.42,
    }

    result = _normalize_hit(hit)

    assert isinstance(result["metadata"], dict)
    assert result["metadata"] == content_meta
    assert result["source_id"] == "/tmp/doc.pdf"
    assert result["page_number"] == 3
    assert result["pdf_basename"] == "doc"
    assert result["pdf_page"] == "doc_3"
    assert result["_distance"] == 0.42


def test_normalize_hit_passes_through_dict_metadata() -> None:
    """Non-LanceDB backends may hand back metadata already as a dict — don't mangle it."""
    content_meta = {"type": "table", "page_number": 1}
    hit = {
        "text": "hi",
        "content_metadata": content_meta,
        "source_id": "/tmp/a.pdf",
    }

    result = _normalize_hit(hit)

    assert result["metadata"] is content_meta


def test_normalize_hit_missing_metadata_yields_empty_dict() -> None:
    result = _normalize_hit({"text": "x", "source_id": "/tmp/b.pdf"})

    assert result["metadata"] == {}
    assert isinstance(result["metadata"], dict)


def test_normalize_retrieval_results_preserves_per_query_shape() -> None:
    raw = [
        [
            {"entity": {"text": "q1-h1", "metadata": json.dumps({"type": "text"})}},
            {"entity": {"text": "q1-h2", "metadata": json.dumps({"type": "image"})}},
        ],
        [
            {"entity": {"text": "q2-h1", "metadata": json.dumps({"type": "chart"})}},
        ],
    ]

    out = normalize_retrieval_results(raw)

    assert [len(hits) for hits in out] == [2, 1]
    assert out[0][0]["metadata"] == {"type": "text"}
    assert out[0][1]["metadata"] == {"type": "image"}
    assert out[1][0]["metadata"] == {"type": "chart"}


def test_normalize_retrieval_results_handles_none_and_dict_inputs() -> None:
    assert normalize_retrieval_results(None) == []

    single = normalize_retrieval_results({"entity": {"text": "t", "metadata": json.dumps({"type": "text"})}})
    assert len(single) == 1 and len(single[0]) == 1
    assert single[0][0]["metadata"] == {"type": "text"}
