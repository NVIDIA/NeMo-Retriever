# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evidence-output path for the skill: `build_evidence_result` shaping +
`retriever query --format evidence [--hybrid]`. (The standalone `retrieve`
subcommand was folded into `query`; `--format evidence`/`--hybrid` are opt-in
flags — `query`'s defaults stay legacy `hits`/vector-only.)"""

from __future__ import annotations

import importlib
import json
import os

from typer.testing import CliRunner

import nemo_retriever.cli.evidence as sw

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


# --- build_evidence_result: hits -> {evidence, coverage} shaping ------------- #
def test_build_evidence_result_shapes_evidence_and_coverage():
    hits = [
        _hit("prose", source="a.pdf", page=3, content_type="text", fidelity="verbatim"),
        _hit("chart cap", source="a.pdf", page=3, content_type="chart", fidelity="vlm_caption"),
    ]
    r = sw.build_evidence_result(hits, ["semantic", "lexical"])
    _assert_conforms(r)
    assert r["evidence"][0]["citation"] == "a p.3"
    assert r["evidence"][0]["locator"] == {"kind": "page", "value": 3}
    assert r["coverage"]["strategies_used"] == ["semantic", "lexical"]
    assert r["coverage"]["n_docs_seen"] == 1
    assert "single source" in r["coverage"]["thin_spots"]


def test_build_evidence_result_fidelity_fallback_when_absent():
    # no metadata.fidelity -> derived from modality. standalone image -> vlm_caption;
    # chart is region-OCR'd per SP-A -> ocr.
    img = _hit("i", content_type="image", fidelity=None)
    chart = _hit("c", content_type="chart", fidelity=None)
    r = sw.build_evidence_result([img, chart], ["semantic"])
    assert r["evidence"][0]["fidelity"] == "vlm_caption"
    assert r["evidence"][1]["fidelity"] == "ocr"


def test_build_evidence_result_empty_thin_spot():
    r = sw.build_evidence_result([], ["semantic"])
    assert r["evidence"] == []
    assert "no matches — likely out of corpus" in r["coverage"]["thin_spots"]


# --- `query --format evidence`: CLI integration ----------------------------- #
def test_query_evidence_graceful_vector_fallback(monkeypatch):
    """--hybrid tries hybrid, falls back to vector-only when the table has no FTS index."""
    cli_main = importlib.import_module("nemo_retriever.cli.main")
    calls = []

    def fake_qd(request):
        calls.append(request.retrieval.hybrid)
        if request.retrieval.hybrid:
            raise RuntimeError("no FTS index")
        return [_hit("p")]

    monkeypatch.setattr(cli_main, "query_documents", fake_qd)
    result = CliRunner().invoke(
        cli_main.app, ["query", "q", "--format", "evidence", "--hybrid", "--table-name", "t"]
    )
    assert result.exit_code == 0, result.output
    assert calls == [True, False]  # tried hybrid, fell back to vector
    out = json.loads(result.output)
    assert out["coverage"]["strategies_used"] == ["semantic"]
    assert len(out["evidence"]) == 1


def test_query_evidence_cli_prints_json(monkeypatch) -> None:
    cli_main = importlib.import_module("nemo_retriever.cli.main")
    monkeypatch.setattr(cli_main, "query_documents", lambda request: [])
    result = CliRunner().invoke(
        cli_main.app, ["query", "q", "--format", "evidence", "--no-hybrid", "--table-name", "t"]
    )
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    assert out["coverage"]["strategies_used"] == ["semantic"]
    assert out["evidence"] == []
