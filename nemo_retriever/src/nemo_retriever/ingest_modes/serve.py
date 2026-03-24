# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_retriever.adapters.service.app import app

__all__ = ["app"]

try:
    from nemo_retriever.adapters.service.app import RetrieverAPIDeployment  # noqa: F401

    __all__.append("RetrieverAPIDeployment")
except ImportError:
    pass
