# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_retriever.common.params import ExtractParams
from nemo_retriever.operators.extract.pdf.extract import build_pdf_extraction_kwargs


@pytest.mark.parametrize(("value", "expected"), [(None, True), (True, True), (False, False)])
def test_unresolved_auto_keeps_image_crops_outside_graph_ingestor(value, expected):
    assert build_pdf_extraction_kwargs(ExtractParams(extract_images=value))["extract_images"] is expected
