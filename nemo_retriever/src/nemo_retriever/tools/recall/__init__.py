# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Recall evaluation utilities.
"""

from nemo_retriever.tools.recall.core import RecallConfig, evaluate_recall

__all__ = [
    "RecallConfig",
    "evaluate_recall",
]
