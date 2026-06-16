# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility exports for the canonical graph operator base classes."""

from __future__ import annotations

from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.cpu_operator import CPUOperator
from nemo_retriever.operators.gpu_operator import GPUOperator

__all__ = ["AbstractOperator", "CPUOperator", "GPUOperator"]
