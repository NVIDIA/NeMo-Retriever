# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys


def _clear_modules(*names: str) -> None:
    for name in names:
        sys.modules.pop(name, None)


def test_cpu_actor_modules_do_not_import_local_models():
    _clear_modules(
        "nemo_retriever.models.local",
        "nemo_retriever.operators.extract.page_elements.cpu_actor",
        "nemo_retriever.common.modality.page_elements.shared",
        "nemo_retriever.common.modality.page_elements.local",
        "nemo_retriever.operators.extract.ocr.cpu_ocr",
        "nemo_retriever.operators.extract.ocr.cpu_parse",
        "nemo_retriever.common.modality.ocr.shared",
        "nemo_retriever.operators.extract.chart.cpu_actor",
        "nemo_retriever.common.modality.chart.shared",
        "nemo_retriever.operators.extract.table.cpu_actor",
        "nemo_retriever.common.modality.table.shared",
        "nemo_retriever.operators.embed.cpu_operator",
        "nemo_retriever.models.inference.shared",
    )

    importlib.import_module("nemo_retriever.operators.extract.page_elements.cpu_actor")
    importlib.import_module("nemo_retriever.operators.extract.ocr.cpu_ocr")
    importlib.import_module("nemo_retriever.operators.extract.ocr.cpu_parse")
    importlib.import_module("nemo_retriever.operators.extract.chart.cpu_actor")
    importlib.import_module("nemo_retriever.operators.extract.table.cpu_actor")
    importlib.import_module("nemo_retriever.operators.embed.cpu_operator")

    assert "nemo_retriever.models.local" not in sys.modules


def test_legacy_cpu_safe_shims_do_not_import_local_models():
    _clear_modules(
        "nemo_retriever.models.local",
        "nemo_retriever.operators.extract.page_elements.page_elements",
        "nemo_retriever.operators.extract.ocr.ocr",
        "nemo_retriever.operators.extract.chart.chart_detection",
        "nemo_retriever.operators.extract.table.table_detection",
        "nemo_retriever.operators.embed.operators",
    )

    importlib.import_module("nemo_retriever.operators.extract.page_elements.page_elements")
    importlib.import_module("nemo_retriever.operators.extract.ocr.ocr")
    importlib.import_module("nemo_retriever.operators.extract.chart.chart_detection")
    importlib.import_module("nemo_retriever.operators.extract.table.table_detection")
    importlib.import_module("nemo_retriever.operators.embed.operators")

    assert "nemo_retriever.models.local" not in sys.modules
