# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ingestor bucket: ingestion orchestration, planning, manifests and results.

The public ingestor API lives in :mod:`nemo_retriever.ingestor.core` and is
re-exported here so that ``nemo_retriever.ingestor`` keeps the exact module-level
surface it had before the reorganization (``create_ingestor``, ``ingestor`` /
``Ingestor``, ``_merge_params`` and the re-exported param models such as
``IngestorCreateParams``).
"""
from nemo_retriever.ingestor.core import *  # noqa: F401,F403
from nemo_retriever.ingestor.core import _merge_params  # noqa: F401

__all__ = ["create_ingestor", "ingestor", "Ingestor"]
