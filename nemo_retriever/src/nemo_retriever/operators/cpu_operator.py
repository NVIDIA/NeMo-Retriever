# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mixin flag for operators that run on CPU only."""


class CPUOperator:
    """Mixin flag indicating an operator runs on CPU only.

    Operators that perform no GPU work (file I/O, text splitting,
    DataFrame transforms) should inherit from both
    :class:`AbstractOperator` and this class::

        class MyCPUActor(AbstractOperator, CPUOperator):
            ...

    Executors can inspect ``isinstance(op, CPUOperator)`` to skip GPU
    resource allocation for these stages.
    """
