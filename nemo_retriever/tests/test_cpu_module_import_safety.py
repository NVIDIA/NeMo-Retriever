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
        "nemo_retriever.model.local",
        "nemo_retriever.page_elements.cpu_actor",
        "nemo_retriever.page_elements.shared",
        "nemo_retriever.page_elements.local",
        "nemo_retriever.ocr.cpu_ocr",
        "nemo_retriever.ocr.cpu_parse",
        "nemo_retriever.ocr.shared",
        "nemo_retriever.chart.cpu_actor",
        "nemo_retriever.chart.shared",
        "nemo_retriever.table.cpu_actor",
        "nemo_retriever.table.shared",
        "nemo_retriever.text_embed.cpu_operator",
        "nemo_retriever.text_embed.shared",
    )

    importlib.import_module("nemo_retriever.page_elements.cpu_actor")
    importlib.import_module("nemo_retriever.ocr.cpu_ocr")
    importlib.import_module("nemo_retriever.ocr.cpu_parse")
    importlib.import_module("nemo_retriever.chart.cpu_actor")
    importlib.import_module("nemo_retriever.table.cpu_actor")
    importlib.import_module("nemo_retriever.text_embed.cpu_operator")

    assert "nemo_retriever.model.local" not in sys.modules


def test_legacy_cpu_safe_shims_do_not_import_local_models():
    _clear_modules(
        "nemo_retriever.model.local",
        "nemo_retriever.page_elements.page_elements",
        "nemo_retriever.ocr.ocr",
        "nemo_retriever.chart.chart_detection",
        "nemo_retriever.table.table_detection",
        "nemo_retriever.text_embed.operators",
    )

    importlib.import_module("nemo_retriever.page_elements.page_elements")
    importlib.import_module("nemo_retriever.ocr.ocr")
    importlib.import_module("nemo_retriever.chart.chart_detection")
    importlib.import_module("nemo_retriever.table.table_detection")
    importlib.import_module("nemo_retriever.text_embed.operators")

    assert "nemo_retriever.model.local" not in sys.modules
