# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from nemo_retriever.common.params import ExtractParams
from nemo_retriever.ingestor.graph_ingestor import GraphIngestor


@pytest.fixture
def pdf_ingestor(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    return GraphIngestor(run_mode="inprocess", show_progress=False).files([str(pdf)])


def _extract_images(ingestor):
    return ingestor._resolve_effective_extraction_inputs().extract_params.extract_images


def test_auto_without_consumers_resolves_false_and_logs_once(pdf_ingestor, caplog):
    pdf_ingestor.extract()
    with caplog.at_level(logging.INFO, logger="nemo_retriever.ingestor.graph_ingestor"):
        assert _extract_images(pdf_ingestor) is False
        assert _extract_images(pdf_ingestor) is False
    messages = [r.getMessage() for r in caplog.records if "extract_images=auto" in r.getMessage()]
    assert messages == ["extract_images=auto -> False (no caption/store/dedup consumer)"]
    assert pdf_ingestor._extract_params.extract_images is None


def test_extract_kwargs_keep_auto(pdf_ingestor):
    pdf_ingestor.extract(ExtractParams(), dpi=300)
    assert pdf_ingestor._extract_params.extract_images is None
    assert _extract_images(pdf_ingestor) is False


@pytest.mark.parametrize("stage", ["caption", "dedup", "store"])
def test_image_consumer_stages_resolve_auto_to_true(pdf_ingestor, stage, caplog):
    getattr(pdf_ingestor.extract(), stage)()
    with caplog.at_level(logging.INFO, logger="nemo_retriever.ingestor.graph_ingestor"):
        assert _extract_images(pdf_ingestor) is True
    assert f"extract_images=auto -> True (consumers: {stage})" in caplog.text


@pytest.mark.parametrize("value", [True, False])
def test_explicit_extract_images_is_kept(pdf_ingestor, value):
    pdf_ingestor.extract(extract_images=value)
    assert _extract_images(pdf_ingestor) is value
