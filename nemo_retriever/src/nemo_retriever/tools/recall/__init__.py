# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Recall evaluation utilities and CLI.
"""

from nemo_retriever.tools.recall.__main__ import app
from nemo_retriever.tools.recall.core import RecallConfig, evaluate_recall

__all__ = [
    "app",
    "RecallConfig",
    "evaluate_recall",
]
