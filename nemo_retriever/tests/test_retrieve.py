# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os

import nemo_retriever.adapters.cli.sdk_workflow as sw

_CONTRACT = os.path.join(
    os.path.dirname(__file__), "..", "..", "docs", "superpowers", "contracts", "retriever", "contract.schema.json"
)


def _hit(text, *, source="doc.pdf", page=1, content_type="text", fidelity=None, score=0.3, meta_extra=None):
    meta = {"type": content_type}
    if fidelity is not None:
        meta["fidelity"] = fidelity
    if meta_extra:
        meta.update(meta_extra)
    return {
        "text": text,
        "pdf_basename": source[:-4] if source.endswith(".pdf") else source,
        "source": source,
        "page_number": page,
        "content_type": content_type,
        "metadata": meta,
        "_score": score,
    }


def _assert_conforms(result):
    schema = json.load(open(_CONTRACT))["$defs"]
    mod_enum = set(schema["evidence_item"]["properties"]["modality"]["enum"])
    fid_enum = set(schema["evidence_item"]["properties"]["fidelity"]["enum"])
    loc_enum = set(schema["locator"]["properties"]["kind"]["enum"])
    for e in result["evidence"]:
        assert {"text", "source", "locator", "modality", "fidelity", "score", "citation"} <= set(e)
        assert e["modality"] in mod_enum, e["modality"]
        assert e["fidelity"] in fid_enum, e["fidelity"]
        assert e["locator"]["kind"] in loc_enum
        assert isinstance(e["score"], (int, float))
    assert {"strategies_used", "n_docs_seen", "thin_spots"} <= set(result["coverage"])


def test_retrieve_shapes_evidence_and_coverage(monkeypatch):
    hits = [
        _hit("prose", source="a.pdf", page=3, content_type="text", fidelity="verbatim"),
        _hit("chart cap", source="a.pdf", page=3, content_type="chart", fidelity="vlm_caption"),
    ]
    monkeypatch.setattr(sw, "query_documents", lambda *a, **k: hits)
    r = sw.retrieve("q", lancedb_uri="x", table_name="t")
    _assert_conforms(r)
    assert r["evidence"][0]["citation"] == "a p.3"
    assert r["evidence"][0]["locator"] == {"kind": "page", "value": 3}
    assert r["coverage"]["strategies_used"] == ["semantic", "lexical"]
    assert r["coverage"]["n_docs_seen"] == 1
    assert "single source" in r["coverage"]["thin_spots"]


def test_retrieve_fidelity_fallback_when_absent(monkeypatch):
    # no metadata.fidelity -> derived from modality. standalone image -> vlm_caption;
    # chart is region-OCR'd per SP-A -> ocr.
    img = _hit("i", content_type="image", fidelity=None)
    chart = _hit("c", content_type="chart", fidelity=None)
    monkeypatch.setattr(sw, "query_documents", lambda *a, **k: [img, chart])
    r = sw.retrieve("q", lancedb_uri="x", table_name="t")
    assert r["evidence"][0]["fidelity"] == "vlm_caption"
    assert r["evidence"][1]["fidelity"] == "ocr"


def test_retrieve_empty_thin_spot(monkeypatch):
    monkeypatch.setattr(sw, "query_documents", lambda *a, **k: [])
    r = sw.retrieve("q", lancedb_uri="x", table_name="t")
    assert r["evidence"] == []
    assert "no matches — likely out of corpus" in r["coverage"]["thin_spots"]


def test_retrieve_graceful_vector_fallback(monkeypatch):
    calls = []

    def fake_qd(question, **k):
        calls.append(k.get("hybrid"))
        if k.get("hybrid"):
            raise RuntimeError("no FTS index")
        return [_hit("p")]

    monkeypatch.setattr(sw, "query_documents", fake_qd)
    r = sw.retrieve("q", lancedb_uri="x", table_name="t")
    assert calls == [True, False]  # tried hybrid, fell back to vector
    assert r["coverage"]["strategies_used"] == ["semantic"]
    assert len(r["evidence"]) == 1


def test_retrieve_cli_prints_json(monkeypatch) -> None:
    import importlib

    from typer.testing import CliRunner

    cli_main = importlib.import_module("nemo_retriever.adapters.cli.main")

    monkeypatch.setattr(
        sw,
        "retrieve",
        lambda question, **k: {
            "evidence": [],
            "coverage": {
                "strategies_used": ["semantic"],
                "n_docs_seen": 0,
                "thin_spots": ["no matches — likely out of corpus"],
            },
        },
    )
    result = CliRunner().invoke(cli_main.app, ["retrieve", "q", "--no-hybrid", "--table-name", "t"])
    assert result.exit_code == 0
    out = json.loads(result.output)
    assert out["coverage"]["strategies_used"] == ["semantic"]
    assert out["evidence"] == []
