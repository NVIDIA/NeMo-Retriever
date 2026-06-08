# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PDF extraction stage (pure Python + Ray Data adapters).

This package intentionally reuses the core extraction logic from `nemo_retriever.api`
and only provides thin orchestration wrappers.
"""

from nemo_retriever.cli.pdf.__main__ import app
from nemo_retriever.common.modality.pdf.config import PDFExtractionStageConfig, load_pdf_extractor_schema_from_dict
from nemo_retriever.common.modality.pdf.io import pdf_files_to_ledger_df
from nemo_retriever.cli.pdf.stage import extract_pdf_primitives_from_ledger_df, make_pdf_task_config

__all__ = [
    "app",
    "PDFExtractionStageConfig",
    "extract_pdf_primitives_from_ledger_df",
    "load_pdf_extractor_schema_from_dict",
    "make_pdf_task_config",
    "pdf_files_to_ledger_df",
]
