# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `pdf/stage.py` JSON-sidecar helpers — specifically the
recursive strip used by ``--compact-json`` to remove placeholder fields
that bloat sidecars for programmatic consumers."""

from __future__ import annotations

import pandas as pd
from typer.testing import CliRunner

import nemo_retriever.cli.pdf.stage as pdf_stage
from nemo_retriever.cli.pdf.stage import _strip_null_empty
from nemo_retriever.common.api.internal.schemas.extract.extract_pdf_schema import (
    NemotronParseConfigSchema,
    PDFExtractorSchema,
)


def test_strip_drops_nulls_and_empty_collections():
    assert _strip_null_empty({"a": None, "b": "", "c": [], "d": {}, "e": "keep"}) == {"e": "keep"}


def test_strip_drops_negative_one_placeholders():
    # `-1` is the conventional "not applicable" sentinel across
    # content_metadata.hierarchy, start_time/end_time, partition_id, etc.
    assert _strip_null_empty({"partition_id": -1, "page": 3}) == {"page": 3}


def test_strip_keeps_zero_and_false():
    # 0 and False are real values, not placeholders.
    assert _strip_null_empty({"count": 0, "flag": False, "x": -1}) == {"count": 0, "flag": False}


def test_strip_recurses_into_nested_dicts():
    obj = {
        "outer": {"inner_keep": "v", "inner_drop": None},
        "all_empty": {"a": None, "b": ""},
    }
    # Inner dict that becomes empty after recursion gets dropped from parent.
    assert _strip_null_empty(obj) == {"outer": {"inner_keep": "v"}}


def test_strip_filters_lists_and_drops_emptied_lists():
    # The classic text_location placeholder [-1, -1, -1, -1] collapses to [].
    assert _strip_null_empty({"text_location": [-1, -1, -1, -1], "bbox": [10, 20, 30, 40]}) == {
        "bbox": [10, 20, 30, 40]
    }


def test_strip_preserves_scalar_strings_and_numbers():
    assert _strip_null_empty({"text": "hello", "n": 42}) == {"text": "hello", "n": 42}


def _invoke_pdf_stage_and_capture_config(monkeypatch, tmp_path, args: list[str]) -> PDFExtractorSchema:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    captured: list[PDFExtractorSchema] = []

    monkeypatch.setattr(pdf_stage, "_safe_pdf_page_count", lambda _path: 1)
    monkeypatch.setattr(pdf_stage, "pdf_files_to_ledger_df", lambda *_args, **_kwargs: pd.DataFrame())

    def fake_extract(_df_ledger, *, extractor_config, **_kwargs):
        captured.append(extractor_config)
        return pd.DataFrame(), {}

    monkeypatch.setattr(pdf_stage, "extract_pdf_primitives_from_ledger_df", fake_extract)

    result = CliRunner().invoke(pdf_stage.app, [str(pdf), *args])
    assert result.exit_code == 0, result.output
    return captured[0]


def test_pdf_stage_default_pdfium_does_not_configure_nemotron_parse(monkeypatch, tmp_path):
    extractor_config = _invoke_pdf_stage_and_capture_config(monkeypatch, tmp_path, [])

    assert extractor_config.nemotron_parse_config is None


def test_pdf_stage_nemotron_parse_omits_model_to_use_schema_default(monkeypatch, tmp_path):
    extractor_config = _invoke_pdf_stage_and_capture_config(
        monkeypatch,
        tmp_path,
        [
            "--method",
            "nemotron_parse",
            "--nemotron-parse-http-endpoint",
            "http://parse:8000/v1/chat/completions",
        ],
    )

    assert extractor_config.nemotron_parse_config.nemotron_parse_model_name == (
        NemotronParseConfigSchema.model_fields["nemotron_parse_model_name"].default
    )


def test_pdf_stage_nemotron_parse_uses_explicit_model_name(monkeypatch, tmp_path):
    extractor_config = _invoke_pdf_stage_and_capture_config(
        monkeypatch,
        tmp_path,
        [
            "--method",
            "nemotron_parse",
            "--nemotron-parse-http-endpoint",
            "http://parse:8000/v1/chat/completions",
            "--nemotron-parse-model-name",
            "my-custom-model",
        ],
    )

    assert extractor_config.nemotron_parse_config.nemotron_parse_model_name == "my-custom-model"
