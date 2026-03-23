# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_retriever.params import EmbedParams
from nemo_retriever.params import IngestorCreateParams
from nemo_retriever.params import TabularExtractParams
from nemo_retriever.params import VdbUploadParams

from .executor import run_mode_ingest_tabular


def run_batch_tabular(
    *,
    create_params: IngestorCreateParams | None = None,
    tabular_params: TabularExtractParams,
    embed_params: EmbedParams | None = None,
    vdb_params: VdbUploadParams | None = None,
) -> object:
    return run_mode_ingest_tabular(
        run_mode="batch",
        create_params=create_params,
        tabular_params=tabular_params,
        embed_params=embed_params,
        vdb_params=vdb_params,
    )
