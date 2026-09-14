# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd

from nemo_retriever.common.params import ExtractParams
from nemo_retriever.operators.extract.ocr.gpu_ocr import OCRActor as OCRGPUActor
from nemo_retriever.operators.extract.ocr.ocr import OCRActor
from nemo_retriever.operators.extract.page_elements.gpu_actor import PageElementDetectionActor as PageElementsGPUActor
from nemo_retriever.operators.extract.page_elements.page_elements import PageElementDetectionActor
from nemo_retriever.operators.extract.parse.nemotron_parse import NemotronParseActor
from nemo_retriever.operators.extract.table.table_detection import TableStructureActor
from nemo_retriever.operators.graph_ops.multi_type_extract_operator import (
    MultiTypeExtractCPUActor,
    MultiTypeExtractOperator,
)


def test_ocr_blank_canonical_does_not_shadow_alias() -> None:
    kwargs = {"ocr_invoke_url": "   ", "invoke_url": "http://localhost:8000"}
    assert OCRActor.prefers_cpu_variant(kwargs) is True
    assert OCRActor.prefers_cpu_variant({"ocr_invoke_url": "   "}) is False


def test_ocr_canonical_endpoint_replaces_blank_alias_for_delegate() -> None:
    actor = OCRGPUActor(ocr_invoke_url="http://canonical", invoke_url="   ")
    assert actor.ocr_kwargs["invoke_url"] == "http://canonical"
    assert actor._model is None


def test_page_elements_blank_canonical_does_not_shadow_alias() -> None:
    kwargs = {"page_elements_invoke_url": "   ", "invoke_url": "http://localhost:8000"}
    assert PageElementDetectionActor.prefers_cpu_variant(kwargs) is True


def test_page_elements_canonical_endpoint_replaces_conflicting_alias_for_delegate() -> None:
    actor = PageElementsGPUActor(page_elements_invoke_url="http://canonical", invoke_url="http://alias")
    assert actor.detect_kwargs["invoke_url"] == "http://canonical"
    assert actor._model is None


def test_table_structure_blank_canonical_does_not_shadow_alias() -> None:
    kwargs = {"table_structure_invoke_url": "   ", "invoke_url": "http://localhost:8000"}
    assert TableStructureActor.prefers_cpu_variant(kwargs) is True


def test_nemotron_parse_blank_canonical_does_not_shadow_alias() -> None:
    kwargs = {"nemotron_parse_invoke_url": "   ", "invoke_url": "http://localhost:8000"}
    assert NemotronParseActor.prefers_cpu_variant(kwargs) is True


def test_multi_type_extract_remote_ocr_without_page_elements_prefers_cpu() -> None:
    params = ExtractParams(
        method="pdfium_hybrid",
        use_page_elements=False,
        ocr_invoke_url="http://ocr-nim/v1/infer",
    )

    assert MultiTypeExtractOperator.prefers_cpu_variant({"extract_params": params}) is True


def test_multi_type_extract_remote_ocr_without_page_elements_skips_detection() -> None:
    params = ExtractParams(
        method="pdfium_hybrid",
        use_page_elements=False,
        extract_tables=False,
        extract_charts=False,
        ocr_invoke_url="http://ocr-nim/v1/infer",
    )
    actor = object.__new__(MultiTypeExtractCPUActor)
    actor.extract_params = params
    resolved: list[type] = []

    class _PassthroughActor:
        @staticmethod
        def run(batch_df: pd.DataFrame) -> pd.DataFrame:
            return batch_df

    def resolve(archetype: type, **_kwargs: object) -> _PassthroughActor:
        resolved.append(archetype)
        return _PassthroughActor()

    actor._instantiate_resolved = resolve  # type: ignore[method-assign]
    batch_df = pd.DataFrame({"page_image": [{"image_b64": "page"}], "metadata": [{"needs_ocr_for_text": True}]})

    result = actor._run_detection_pipeline(batch_df)

    assert result.equals(batch_df)
    assert PageElementDetectionActor not in resolved
    assert resolved == [OCRActor]
