# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mixin flag for operators that require GPU resources."""


class GPUOperator:
    """Mixin flag indicating an operator requires GPU resources.

    Operators that load torch models or perform CUDA-based inference
    should inherit from both :class:`AbstractOperator` and this class::

        class MyGPUActor(AbstractOperator, GPUOperator):
            ...

    Executors can inspect ``isinstance(op, GPUOperator)`` to allocate
    GPU resources or route work to GPU-capable workers.
    """
