# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for :class:`GraphIngestionError` diagnostic enrichment.

When the strict ``error_policy="raise"`` path detects row-level errors
from explicitly configured remote NIM endpoints, the rendered exception
message used to read::

    Graph ingestion detected row-level errors from an explicitly
    configured remote NIM endpoint. row 2, column table_structure_ocr_v1
    , path error: remote_inference: ConnectionError: connection refused

That was easy to truncate in container log viewers and hard to attribute
to a specific NIM. The fix adds per-row ``[stage=… url=… http=…]`` tags,
extracts HTTP status codes from common payload shapes, and appends a
``Troubleshooting:`` footer with concrete next steps. These tests pin
the new shape and verify it remains backwards compatible with callers
that build :class:`GraphIngestionError` directly.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nemo_retriever.ingestor.graph_ingestor import (
    GraphIngestionError,
    GraphIngestor,
    _StageDiagnostic,
)
from nemo_retriever.common.params import EmbedParams, ExtractParams


# ---------------------------------------------------------------------------
# Stage diagnostics resolution
# ---------------------------------------------------------------------------


def test_remote_stage_diagnostics_populated_from_extract_params() -> None:
    ingestor = GraphIngestor(run_mode="inprocess").extract(
        ExtractParams(
            page_elements_invoke_url="http://page-elements.svc/v1/infer",
            ocr_invoke_url="http://ocr.svc/v1/infer",
            table_structure_invoke_url="http://table-structure.svc/v1/infer",
            graphic_elements_invoke_url="http://graphic-elements.svc/v1/infer",
        ),
    )
    diagnostics = ingestor._remote_stage_diagnostics()

    assert set(diagnostics.keys()) >= {"ocr", "table_structure_ocr_v1", "graphic_elements_ocr_v1"}

    page_elements = diagnostics.get("page_elements_v3") or diagnostics.get(
        ingestor._param_value(ingestor._extract_params, "output_column")
    )
    assert page_elements is not None
    assert page_elements.display_name == "Page Elements NIM"
    assert page_elements.invoke_url == "http://page-elements.svc/v1/infer"

    assert diagnostics["ocr"].invoke_url == "http://ocr.svc/v1/infer"
    assert diagnostics["table_structure_ocr_v1"].invoke_url == "http://table-structure.svc/v1/infer"
    assert diagnostics["graphic_elements_ocr_v1"].invoke_url == "http://graphic-elements.svc/v1/infer"


def test_remote_stage_diagnostics_populated_from_embed_params() -> None:
    ingestor = (
        GraphIngestor(run_mode="inprocess")
        .extract(ExtractParams())
        .embed(
            EmbedParams(
                embed_invoke_url="http://embed.svc/v1/embeddings",
                model_name="nvidia/llama-nemotron-embed-1b-v2",
            )
        )
    )
    diagnostics = ingestor._remote_stage_diagnostics()
    embed_diag = next(d for d in diagnostics.values() if d.role == "embed")
    assert embed_diag.display_name == "Embedding NIM"
    assert embed_diag.invoke_url == "http://embed.svc/v1/embeddings"
    assert embed_diag.model_name == "nvidia/llama-nemotron-embed-1b-v2"


def test_remote_stage_error_columns_preserves_legacy_shape() -> None:
    """``_remote_stage_error_columns`` still returns a plain ``set[str]``."""
    ingestor = GraphIngestor(run_mode="inprocess").extract(
        ExtractParams(table_structure_invoke_url="http://x/v1/infer"),
    )
    cols = ingestor._remote_stage_error_columns()
    assert isinstance(cols, set)
    assert "table_structure_ocr_v1" in cols


# ---------------------------------------------------------------------------
# Rendered error message
# ---------------------------------------------------------------------------


def test_error_message_includes_stage_name_and_invoke_url() -> None:
    diag = _StageDiagnostic(
        column="table_structure_ocr_v1",
        display_name="Table Structure NIM",
        invoke_url="http://table-structure.svc/v1/infer",
        role="table_structure",
    )
    err = GraphIngestionError(
        [
            {
                "row_index": 2,
                "column": "table_structure_ocr_v1",
                "path": "error",
                "error": {
                    "stage": "remote_inference",
                    "type": "ConnectionError",
                    "message": "connection refused",
                },
            }
        ],
        stage_diagnostics={"table_structure_ocr_v1": diag},
    )
    rendered = str(err)
    assert "stage=Table Structure NIM" in rendered
    assert "url=http://table-structure.svc/v1/infer" in rendered
    assert "row 2" in rendered
    assert "connection refused" in rendered


def test_error_message_extracts_http_status_code_from_status_code_field() -> None:
    diag = _StageDiagnostic(
        column="page_elements_v3",
        display_name="Page Elements NIM",
        invoke_url="http://pe.svc/v1/infer",
        role="page_elements",
    )
    err = GraphIngestionError(
        [
            {
                "row_index": 0,
                "column": "page_elements_v3",
                "path": "error",
                "error": {
                    "stage": "remote_inference",
                    "type": "HTTPError",
                    "status_code": 503,
                    "message": "service unavailable",
                },
            }
        ],
        stage_diagnostics={"page_elements_v3": diag},
    )
    rendered = str(err)
    assert "http=503" in rendered
    assert "5xx server error" in rendered  # troubleshooting hint
    assert "inspect the NIM pod logs" in rendered


@pytest.mark.parametrize(
    "field_name",
    ["status_code", "http_status", "status", "code"],
)
def test_error_message_extracts_http_status_from_alternative_fields(field_name: str) -> None:
    diag = _StageDiagnostic(
        column="ocr",
        display_name="OCR NIM",
        invoke_url="http://ocr.svc/v1/infer",
        role="ocr",
    )
    err = GraphIngestionError(
        [
            {
                "row_index": 0,
                "column": "ocr",
                "path": "error",
                "error": {"type": "HTTPError", field_name: 401, "message": "Unauthorized"},
            }
        ],
        stage_diagnostics={"ocr": diag},
    )
    rendered = str(err)
    assert "http=401" in rendered


def test_error_message_appends_auth_hint_for_401_or_403() -> None:
    diag = _StageDiagnostic(
        column="page_elements_v3",
        display_name="Page Elements NIM",
        invoke_url="http://pe.svc/v1/infer",
        role="page_elements",
    )
    err = GraphIngestionError(
        [
            {
                "row_index": 0,
                "column": "page_elements_v3",
                "path": "error",
                "error": {"type": "HTTPError", "status_code": 401, "message": "no token"},
            }
        ],
        stage_diagnostics={"page_elements_v3": diag},
    )
    rendered = str(err)
    assert "Troubleshooting:" in rendered
    assert "auth error" in rendered
    assert "NGC_API_KEY" in rendered


def test_error_message_appends_client_hint_for_4xx() -> None:
    diag = _StageDiagnostic(
        column="page_elements_v3",
        display_name="Page Elements NIM",
        invoke_url="http://pe.svc/v1/infer",
        role="page_elements",
    )
    err = GraphIngestionError(
        [
            {
                "row_index": 0,
                "column": "page_elements_v3",
                "path": "error",
                "error": {"type": "HTTPError", "status_code": 422, "message": "bad payload"},
            }
        ],
        stage_diagnostics={"page_elements_v3": diag},
    )
    rendered = str(err)
    assert "4xx client error" in rendered
    assert "expected input schema" in rendered


def test_error_message_falls_back_to_generic_hint_without_status_code() -> None:
    diag = _StageDiagnostic(
        column="ocr",
        display_name="OCR NIM",
        invoke_url="http://ocr.svc/v1/infer",
        role="ocr",
    )
    err = GraphIngestionError(
        [
            {
                "row_index": 0,
                "column": "ocr",
                "path": "error",
                "error": {"type": "TimeoutError", "message": "request timed out"},
            }
        ],
        stage_diagnostics={"ocr": diag},
    )
    rendered = str(err)
    assert "Troubleshooting:" in rendered
    assert "kubectl exec" in rendered
    assert "http://ocr.svc/v1/infer" in rendered


def test_error_message_groups_hints_one_per_distinct_stage() -> None:
    diagnostics = {
        "page_elements_v3": _StageDiagnostic(
            column="page_elements_v3",
            display_name="Page Elements NIM",
            invoke_url="http://pe.svc/v1/infer",
            role="page_elements",
        ),
        "ocr": _StageDiagnostic(
            column="ocr",
            display_name="OCR NIM",
            invoke_url="http://ocr.svc/v1/infer",
            role="ocr",
        ),
    }
    records = [
        {
            "row_index": 0,
            "column": "page_elements_v3",
            "path": "error",
            "error": {"type": "HTTPError", "status_code": 503, "message": "boom"},
        },
        {
            "row_index": 1,
            "column": "page_elements_v3",
            "path": "error",
            "error": {"type": "HTTPError", "status_code": 503, "message": "boom"},
        },
        {
            "row_index": 2,
            "column": "ocr",
            "path": "error",
            "error": {"type": "HTTPError", "status_code": 401, "message": "no token"},
        },
    ]
    err = GraphIngestionError(records, stage_diagnostics=diagnostics)
    rendered = str(err)
    assert rendered.count("5xx server error") == 1
    assert rendered.count("auth error") == 1


def test_error_message_remains_backwards_compatible_without_diagnostics() -> None:
    """Callers that pass only records keep the legacy single-line shape."""
    err = GraphIngestionError(
        [
            {
                "row_index": 0,
                "column": "table_structure_ocr_v1",
                "path": "error",
                "error": {"type": "ConnectionError", "message": "boom"},
            }
        ]
    )
    rendered = str(err)
    assert "row 0, column table_structure_ocr_v1" in rendered
    assert "stage=" not in rendered
    assert "Troubleshooting:" not in rendered
    assert err.stage_diagnostics == {}


def test_error_message_truncation_marker_renders_only_when_more_than_five() -> None:
    """The ``(N more)`` clause appears only when records exceed the limit."""
    diag = _StageDiagnostic(
        column="page_elements_v3",
        display_name="Page Elements NIM",
        invoke_url="http://pe.svc/v1/infer",
        role="page_elements",
    )
    records = [
        {
            "row_index": i,
            "column": "page_elements_v3",
            "path": "error",
            "error": {"type": "HTTPError", "status_code": 500},
        }
        for i in range(6)
    ]
    err = GraphIngestionError(records, stage_diagnostics={"page_elements_v3": diag})
    assert "(1 more)" in str(err)


# ---------------------------------------------------------------------------
# End-to-end via GraphIngestor._raise_for_stage_errors
# ---------------------------------------------------------------------------


def test_raise_for_stage_errors_attaches_diagnostics_from_extract_params() -> None:
    ingestor = GraphIngestor(run_mode="inprocess").extract(
        ExtractParams(
            table_structure_invoke_url="http://ts.svc/v1/infer",
            extract_text=False,
            extract_images=False,
            extract_tables=True,
            extract_charts=False,
            extract_infographics=False,
        ),
    )
    result = pd.DataFrame(
        {
            "table_structure_ocr_v1": [
                {
                    "error": {
                        "stage": "remote_inference",
                        "type": "HTTPError",
                        "status_code": 503,
                        "message": "table-structure NIM unavailable",
                    }
                }
            ],
            "metadata": [{"source": "doc.pdf"}],
        }
    )

    with pytest.raises(GraphIngestionError) as exc_info:
        ingestor._raise_for_stage_errors(result)

    rendered = str(exc_info.value)
    assert "stage=Table Structure NIM" in rendered
    assert "url=http://ts.svc/v1/infer" in rendered
    assert "http=503" in rendered
    assert "5xx server error" in rendered
    # The diagnostics mapping is stable, not just a side-effect of formatting:
    assert "table_structure_ocr_v1" in exc_info.value.stage_diagnostics
    assert exc_info.value.stage_diagnostics["table_structure_ocr_v1"].invoke_url == "http://ts.svc/v1/infer"


def test_raise_for_stage_errors_noop_under_collect_policy() -> None:
    ingestor = GraphIngestor(run_mode="inprocess", error_policy="collect").extract(
        ExtractParams(page_elements_invoke_url="http://x/v1/infer"),
    )
    result = pd.DataFrame(
        {
            "page_elements_v3": [{"error": {"type": "HTTPError", "status_code": 500}}],
        }
    )
    # No exception even with row-level errors when policy is collect.
    ingestor._raise_for_stage_errors(result)
