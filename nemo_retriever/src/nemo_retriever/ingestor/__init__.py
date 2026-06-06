# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ingestor bucket: ingestion orchestration, planning, manifests and results.

The public ingestor API (``create_ingestor``, ``ingestor``/``Ingestor``) is
re-exported here from :mod:`nemo_retriever.ingestor.core` so that
``nemo_retriever.ingestor`` keeps working as before the reorganization.
"""
from nemo_retriever.ingestor.core import (  # noqa: F401
    Ingestor,
    _merge_params,
    create_ingestor,
    ingestor,
)

__all__ = ["create_ingestor", "ingestor", "Ingestor"]
