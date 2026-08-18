# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_retriever.operators.extract.ocr.gpu_ocr import OCRActor as OCRGPUActor
from nemo_retriever.operators.extract.ocr.ocr import OCRActor
from nemo_retriever.operators.extract.page_elements.gpu_actor import PageElementDetectionActor as PageElementsGPUActor
from nemo_retriever.operators.extract.page_elements.page_elements import PageElementDetectionActor
from nemo_retriever.operators.extract.parse.nemotron_parse import NemotronParseActor
from nemo_retriever.operators.extract.table.table_detection import TableStructureActor


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
