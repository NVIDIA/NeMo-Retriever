# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end-ish check of ``/v1/query?format=evidence`` over a real LanceDB table.

Unlike ``test_service_vectordb_app.py`` (which mocks ``VectorDBState.search``), this
drives the full server path: real ``/internal/vectordb/write`` → real LanceDB vector
search → real ``normalize_retrieval_results`` → real ``build_evidence_result``. Only
the embedding *model* is stubbed (it needs a NIM/GPU otherwise); everything that the
evidence projection depends on — schema columns, the JSON-string ``metadata``
round-trip, score/locator/fidelity derivation — runs for real.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from nemo_retriever.service.vectordb_app import VectorDBState, create_vectordb_app

_DIM = 4
# One stored chunk, in the real LanceDB schema (metadata is a JSON *string* column).
_ROW = {
    "vector": [1.0, 0.0, 0.0, 0.0],
    "pdf_page": "10k_2023_12",
    "filename": "10k_2023.pdf",
    "pdf_basename": "10k_2023.pdf",
    "page_number": 12,
    "source": "10k_2023.pdf",
    "source_id": "10k_2023.pdf",
    "path": "/data/10k_2023.pdf",
    "text": "Revenue grew 12% year over year.",
    "metadata": json.dumps({"page_number": 12, "type": "text"}),
    "stored_image_uri": "",
    "content_type": "text",
    "bbox_xyxy_norm": "",
}


def test_query_evidence_format_end_to_end_over_real_lancedb(tmp_path) -> None:
    app = create_vectordb_app(
        lancedb_uri=str(tmp_path),
        table_name="nemo_retriever",
        embed_endpoint="http://embed.example/v1/embeddings",  # -> embed_mode="remote"
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
    )

    # Stub ONLY the embedding model; real LanceDB does the rest.
    with patch.object(VectorDBState, "embed_queries", return_value=[[1.0, 0.0, 0.0, 0.0]]):
        with TestClient(app) as client:
            write = client.post("/internal/vectordb/write", json={"rows": [_ROW]})
            assert write.status_code == 200, write.text
            assert write.json()["total_rows"] == 1

            resp = client.post(
                "/v1/query",
                json={"query": "revenue", "top_k": 5, "format": "evidence"},
            )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert list(body) == ["results"]
    assert len(body["results"]) == 1

    item = body["results"][0]
    assert set(item) == {"evidence", "coverage"}

    ev = item["evidence"][0]
    assert ev["text"] == "Revenue grew 12% year over year."
    assert ev["source"] == "10k_2023"
    assert ev["citation"] == "10k_2023 p.12"
    assert ev["locator"] == {"kind": "page", "value": 12}
    assert ev["modality"] == "text"
    assert ev["fidelity"] == "verbatim"
    # Score is the real LanceDB distance/relevance — present and numeric, value not asserted.
    assert isinstance(ev["score"], (int, float))

    coverage = item["coverage"]
    assert coverage["strategies_used"] == ["dense"]
    assert coverage["n_docs_seen"] == 1
    assert coverage["thin_spots"] == ["single source"]
