# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The in-process PDF pipeline must render pages before Nemotron Parse.

``nemotron_parse_pages`` reads ``page_image.image_b64`` off every row and skips
the rows that do not have one. ``PDFSplitActor`` does not produce that column,
so the parse branch has to run ``PDFExtractionActor`` with
``extract_page_as_image`` first, the way ``build_graph`` does.
"""

from __future__ import annotations

import pandas as pd

from nemo_retriever.common.params import ExtractParams
from nemo_retriever.operators.graph_ops import multi_type_extract_operator as multi_type_module
from nemo_retriever.operators.graph_ops.multi_type_extract_operator import MultiTypeExtractCPUActor


def test_parse_branch_renders_pages_before_the_parse_actor(monkeypatch) -> None:
    ran: list[str] = []

    class _RecordingStage:
        def __init__(self, name: str):
            self._name = name

        def run(self, data):
            ran.append(self._name)
            return data

    class _FakeConversion:
        def __init__(self, **_kwargs):
            pass

        def run(self, data):
            ran.append("DocToPdfConversionActor")
            return data

    class _FakeSplit:
        def __init__(self, **_kwargs):
            pass

        def run(self, data):
            ran.append("PDFSplitActor")
            return data

    def _fake_pdf_extraction(**kwargs):
        ran.append("PDFExtractionActor")
        # the page images are the whole reason this stage is in the parse branch
        assert kwargs.get("extract_page_as_image") is True
        return _RecordingStage("PDFExtractionActor.run")

    def _fake_instantiate_resolved(self, operator_class, **_operator_kwargs):
        return _RecordingStage(operator_class.__name__)

    monkeypatch.setattr(multi_type_module, "DocToPdfConversionActor", _FakeConversion)
    monkeypatch.setattr(multi_type_module, "PDFSplitActor", _FakeSplit)
    monkeypatch.setattr(multi_type_module, "PDFExtractionActor", _fake_pdf_extraction)
    monkeypatch.setattr(MultiTypeExtractCPUActor, "_instantiate_resolved", _fake_instantiate_resolved)

    op = MultiTypeExtractCPUActor(
        extraction_mode="auto",
        extract_params=ExtractParams(method="nemotron_parse"),
        split_config={},
    )
    op._run_pdf_pipeline(pd.DataFrame({"path": ["/tmp/doc.pdf"]}))

    assert "PDFExtractionActor" in ran
    assert ran.index("PDFExtractionActor") < ran.index("NemotronParseActor")
