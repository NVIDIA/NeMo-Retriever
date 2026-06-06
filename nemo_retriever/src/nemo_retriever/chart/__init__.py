# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Chart extraction stage (pure Python + Ray Data adapters).

This stage enriches existing STRUCTURED/chart primitives by populating
`metadata.table_metadata.table_content` using YOLOX + OCR fusion.
"""

from nemo_retriever.chart.config import ChartExtractionStageConfig, load_chart_extractor_schema_from_dict
from nemo_retriever.chart.commands import app
from nemo_retriever.chart.processor import extract_chart_data_from_primitives_df

__all__ = [
    "app",
    "ChartExtractionStageConfig",
    "extract_chart_data_from_primitives_df",
    "load_chart_extractor_schema_from_dict",
]
