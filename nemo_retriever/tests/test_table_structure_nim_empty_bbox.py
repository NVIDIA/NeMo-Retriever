# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the table-structure NIM "no detections" path.

The ``nemotron-table-structure-v1`` NIM returns an empty
``bounding_boxes`` object when an input crop contains no table cells /
rows / columns — the canonical, expected response for non-table pages
in a mixed-content document::

    {"index": 0, "bounding_boxes": {}}

Until this regression was fixed, ``table_structure_ocr_page_elements``
treated *any* empty parse output (including this valid "zero
detections" reply) as a parse failure and fell through to
:func:`_prediction_to_detections`. That fallback path requires
``torch`` for tensor normalisation, which the slim ``retriever-service``
container image deliberately omits — the result was a flood of
row-level ``ImportError: torch required for prediction parsing.``
errors on every page that didn't contain a table, surfaced to the
client as ``GraphIngestionError`` with ``status=failed`` for
otherwise-healthy documents.

The fix has two layers, and these tests pin both down:

1. **Call-site discrimination** — ``table_structure_ocr_page_elements``
   only falls through to the legacy parser when the NIM response is
   NOT in ``bounding_boxes`` shape at all (the predicate is exposed as
   :func:`_is_nim_bounding_boxes_response`). An empty
   ``bounding_boxes: {}`` is trusted as "zero detections, done".

2. **Defence-in-depth** — :func:`_prediction_to_detections` in the
   table / chart / infographic modules now extracts candidate
   boxes / labels BEFORE checking ``torch``, so a dict input with no
   detection fields returns ``[]`` instead of raising ``ImportError``
   in torch-free environments. This means a future caller that
   accidentally hands a NIM-shaped response to the legacy parser
   degrades gracefully rather than crashing the whole graph.
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _can_import(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


_needs_pil = pytest.mark.skipif(not _can_import("PIL"), reason="PIL (Pillow) not installed")


# ----------------------------------------------------------------------
# Direct contract tests for the helper predicates
# ----------------------------------------------------------------------


class TestIsNimBoundingBoxesResponse:
    """Pin the response-shape predicate that disambiguates "empty NIM
    bounding_boxes" from "response is not NIM-shaped"."""

    def test_returns_true_for_empty_bounding_boxes_dict(self) -> None:
        from nemo_retriever.common.modality.table.shared import _is_nim_bounding_boxes_response

        # The NIM's canonical "no detections" response.
        assert _is_nim_bounding_boxes_response({"index": 0, "bounding_boxes": {}})

    def test_returns_true_for_populated_bounding_boxes(self) -> None:
        from nemo_retriever.common.modality.table.shared import _is_nim_bounding_boxes_response

        payload = {
            "index": 0,
            "bounding_boxes": {
                "cell": [{"x_min": 0.0, "y_min": 0.0, "x_max": 0.5, "y_max": 0.5, "confidence": 0.9}],
            },
        }
        assert _is_nim_bounding_boxes_response(payload)

    def test_returns_false_for_non_dict(self) -> None:
        from nemo_retriever.common.modality.table.shared import _is_nim_bounding_boxes_response

        assert not _is_nim_bounding_boxes_response(None)
        assert not _is_nim_bounding_boxes_response([])
        assert not _is_nim_bounding_boxes_response("not a dict")
        assert not _is_nim_bounding_boxes_response(0)

    def test_returns_false_for_dict_without_bounding_boxes(self) -> None:
        from nemo_retriever.common.modality.table.shared import _is_nim_bounding_boxes_response

        # Legacy in-process model output — no ``bounding_boxes`` key.
        assert not _is_nim_bounding_boxes_response({"boxes": [], "labels": [], "scores": []})

    def test_returns_false_when_bounding_boxes_is_not_a_dict(self) -> None:
        from nemo_retriever.common.modality.table.shared import _is_nim_bounding_boxes_response

        # Malformed shape — treat as "not a NIM response" so the caller
        # can fall through to the legacy parser.
        assert not _is_nim_bounding_boxes_response({"bounding_boxes": []})
        assert not _is_nim_bounding_boxes_response({"bounding_boxes": "broken"})


class TestParseNimBoundingBoxesEmpty:
    """``_parse_nim_bounding_boxes`` returns ``[]`` for both the NIM's
    empty-detections response and a non-bbox response — the new
    ``_is_nim_bounding_boxes_response`` predicate is the only signal
    that lets callers distinguish them."""

    def test_empty_bounding_boxes_parses_to_empty_list(self) -> None:
        from nemo_retriever.common.modality.table.shared import _parse_nim_bounding_boxes

        assert _parse_nim_bounding_boxes({"index": 0, "bounding_boxes": {}}) == []

    def test_non_bbox_response_parses_to_empty_list(self) -> None:
        from nemo_retriever.common.modality.table.shared import _parse_nim_bounding_boxes

        assert _parse_nim_bounding_boxes({"prediction": [1, 2, 3]}) == []

    def test_populated_bbox_response_parses_to_detections(self) -> None:
        from nemo_retriever.common.modality.table.shared import _parse_nim_bounding_boxes

        payload = {
            "index": 0,
            "bounding_boxes": {
                "cell": [
                    {
                        "x_min": 0.1,
                        "y_min": 0.2,
                        "x_max": 0.3,
                        "y_max": 0.4,
                        "confidence": 0.95,
                    }
                ],
            },
        }
        parsed = _parse_nim_bounding_boxes(payload)
        assert len(parsed) == 1
        assert parsed[0]["label_name"] == "cell"
        assert parsed[0]["bbox_xyxy_norm"] == [0.1, 0.2, 0.3, 0.4]
        assert parsed[0]["score"] == pytest.approx(0.95)


# ----------------------------------------------------------------------
# Defence-in-depth: _prediction_to_detections without torch
# ----------------------------------------------------------------------


class TestPredictionToDetectionsTorchOptional:
    """When the input has no boxes / labels (e.g. a NIM-shaped response
    handed to the wrong parser by mistake), the function must return
    ``[]`` even in torch-free images instead of raising ``ImportError``.
    """

    @pytest.mark.parametrize(
        "module_path",
        [
            "nemo_retriever.common.modality.table.shared",
            "nemo_retriever.common.modality.chart.shared",
            "nemo_retriever.common.modality.infographic.infographic_detection",
        ],
    )
    def test_empty_nim_response_returns_empty_without_torch(self, module_path: str) -> None:
        module = importlib.import_module(module_path)

        # Simulate the retriever-service image: torch is not importable.
        with patch.object(module, "torch", None):
            result = module._prediction_to_detections(
                {"index": 0, "bounding_boxes": {}},
                label_names=["cell", "row", "column"],
            )
        assert result == []

    @pytest.mark.parametrize(
        "module_path",
        [
            "nemo_retriever.common.modality.table.shared",
            "nemo_retriever.common.modality.chart.shared",
            "nemo_retriever.common.modality.infographic.infographic_detection",
        ],
    )
    def test_dict_with_only_index_returns_empty_without_torch(self, module_path: str) -> None:
        """A dict that lacks every box/label/score key must not require torch."""
        module = importlib.import_module(module_path)

        with patch.object(module, "torch", None):
            result = module._prediction_to_detections(
                {"index": 0, "model_name": "nvidia/foo"},
                label_names=[],
            )
        assert result == []

    @pytest.mark.parametrize(
        "module_path",
        [
            "nemo_retriever.common.modality.table.shared",
            "nemo_retriever.common.modality.chart.shared",
            "nemo_retriever.common.modality.infographic.infographic_detection",
        ],
    )
    def test_none_input_returns_empty_without_torch(self, module_path: str) -> None:
        module = importlib.import_module(module_path)

        with patch.object(module, "torch", None):
            result = module._prediction_to_detections(None, label_names=[])
        assert result == []

    @pytest.mark.parametrize(
        "module_path",
        [
            "nemo_retriever.common.modality.table.shared",
            "nemo_retriever.common.modality.chart.shared",
            "nemo_retriever.common.modality.infographic.infographic_detection",
        ],
    )
    def test_payload_with_boxes_still_requires_torch(self, module_path: str) -> None:
        """If the caller really did hand us a torch-shaped payload, the
        function must raise ``ImportError`` so the operator notices the
        torch dependency is missing — silent return-empty would mask
        a misconfigured image."""
        module = importlib.import_module(module_path)

        with patch.object(module, "torch", None):
            with pytest.raises(ImportError, match="torch required for prediction parsing"):
                module._prediction_to_detections(
                    {"boxes": [[0.0, 0.0, 1.0, 1.0]], "labels": [0], "scores": [0.9]},
                    label_names=["cell"],
                )


# ----------------------------------------------------------------------
# End-to-end: empty bbox NIM response in `table_structure_ocr_page_elements`
# ----------------------------------------------------------------------


def _make_page_df_with_table() -> pd.DataFrame:
    """Single-row page DF where ``page_elements_v3`` says there *is* a
    table (so the table-structure stage runs) and the NIM is mocked.
    """
    import base64
    import io

    from PIL import Image

    img = Image.new("RGB", (320, 240), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return pd.DataFrame(
        [
            {
                "page_image": {"image_b64": image_b64},
                "page_elements_v3": {
                    "detections": [
                        {
                            "label_name": "table",
                            "bbox_xyxy_norm": [0.0, 0.0, 1.0, 1.0],
                            "score": 0.95,
                        }
                    ]
                },
                "page_elements_v3_counts_by_label": {"table": 1},
            }
        ]
    )


@_needs_pil
class TestTableStructureNimEmptyBboxEndToEnd:
    """End-to-end: when the table-structure NIM returns an empty
    ``bounding_boxes`` payload, the stage must succeed (no torch use,
    no row-level error) and produce zero structure detections for the
    crop."""

    def test_empty_nim_response_does_not_raise_in_torchless_image(self) -> None:
        from nemo_retriever.models.nim import nim as nim_module
        from nemo_retriever.common.modality.table import shared as table_shared
        from nemo_retriever.operators.extract.table.table_detection import table_structure_ocr_page_elements

        df = _make_page_df_with_table()

        # Simulate the deployed environment: no torch + remote NIM
        # invocation returns the canonical "no detections" payload.
        # ``invoke_image_inference_batches`` is imported locally inside
        # ``table_structure_ocr_page_elements`` so we must patch it on
        # its source module (``nemo_retriever.models.nim.nim``), not on the
        # caller.
        empty_payload = [{"index": 0, "bounding_boxes": {}}]

        with patch.object(table_shared, "torch", None), patch.object(
            nim_module,
            "invoke_image_inference_batches",
            return_value=empty_payload,
        ):
            result = table_structure_ocr_page_elements(
                df,
                table_structure_invoke_url="http://nemotron-table-structure-v1:8000/v1/infer",
                page_elements_invoke_url="http://nemotron-page-elements-v3:8000/v1/infer",
                ocr_invoke_url="http://nemotron-ocr-v1:8000/v1/infer",
            )

        # Stage finished successfully, no row-level error recorded.
        meta = result.iloc[0]["table_structure_ocr_v1"]
        assert (
            meta.get("error") is None
        ), f"unexpected stage error for an empty-detections response: {meta.get('error')!r}"
        # Zero detections on the crop ⇒ a structure-only "table" entry
        # with empty structure_counts (or no entry at all, depending
        # on the downstream join). The crucial assertion is that we
        # did NOT raise an ImportError.
        assert "table" in result.columns

    def test_empty_nim_response_does_not_call_prediction_to_detections(self) -> None:
        from nemo_retriever.models.nim import nim as nim_module
        from nemo_retriever.common.modality.table import shared as table_shared
        from nemo_retriever.operators.extract.table.table_detection import table_structure_ocr_page_elements

        df = _make_page_df_with_table()
        empty_payload = [{"index": 0, "bounding_boxes": {}}]

        with patch.object(table_shared, "torch", None), patch.object(
            nim_module,
            "invoke_image_inference_batches",
            return_value=empty_payload,
        ), patch.object(
            table_shared,
            "_prediction_to_detections",
            side_effect=AssertionError("fallback parser must NOT be called for an empty-bbox NIM response"),
        ):
            # If the fallback is called, the AssertionError surfaces.
            table_structure_ocr_page_elements(
                df,
                table_structure_invoke_url="http://nemotron-table-structure-v1:8000/v1/infer",
                page_elements_invoke_url="http://nemotron-page-elements-v3:8000/v1/infer",
                ocr_invoke_url="http://nemotron-ocr-v1:8000/v1/infer",
            )

    def test_non_bbox_response_still_falls_through_to_legacy_parser(self) -> None:
        """When the response truly isn't NIM-shaped, the legacy parser
        is still invoked — so the fix only suppresses the fallback for
        responses that look like NIM bounding-box envelopes."""
        from nemo_retriever.models.nim import nim as nim_module
        from nemo_retriever.common.modality.table import shared as table_shared
        from nemo_retriever.operators.extract.table.table_detection import table_structure_ocr_page_elements

        df = _make_page_df_with_table()
        # Legacy in-process shape (a dict-of-tensors-style payload, but
        # without box/label keys ⇒ legacy parser returns ``[]`` cleanly
        # under the new ordering even without torch).
        legacy_payload = [{"prediction": {"foo": "bar"}}]

        called = MagicMock(return_value=[])

        with patch.object(table_shared, "torch", None), patch.object(
            nim_module,
            "invoke_image_inference_batches",
            return_value=legacy_payload,
        ), patch.object(table_shared, "_prediction_to_detections", side_effect=called):
            table_structure_ocr_page_elements(
                df,
                table_structure_invoke_url="http://nemotron-table-structure-v1:8000/v1/infer",
                page_elements_invoke_url="http://nemotron-page-elements-v3:8000/v1/infer",
                ocr_invoke_url="http://nemotron-ocr-v1:8000/v1/infer",
            )

        # The non-bbox legacy response triggers the fallback exactly once.
        assert called.called, "legacy parser must be invoked for non-bbox responses"
