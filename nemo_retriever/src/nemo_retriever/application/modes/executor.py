# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_retriever.params import EmbedParams
from nemo_retriever.params import IngestExecuteParams
from nemo_retriever.params import IngestorCreateParams
from nemo_retriever.params import RunMode
from nemo_retriever.params import TabularExtractParams
from nemo_retriever.params import VdbUploadParams

from .factory import create_runmode_ingestor


def run_mode_ingest(
    *,
    run_mode: RunMode,
    create_params: IngestorCreateParams | None = None,
    ingest_params: IngestExecuteParams | None = None,
) -> object:
    ingestor = create_runmode_ingestor(run_mode=run_mode, params=create_params)
    return ingestor.ingest(params=ingest_params)


def run_mode_ingest_tabular(
    *,
    run_mode: RunMode,
    create_params: IngestorCreateParams | None = None,
    tabular_params: TabularExtractParams,
    embed_params: EmbedParams | None = None,
    vdb_params: VdbUploadParams | None = None,
) -> object:
    ingestor = create_runmode_ingestor(run_mode=run_mode, params=create_params)

    if embed_params is not None:
        ingestor = ingestor.embed(embed_params)
    if vdb_params is not None:
        ingestor = ingestor.vdb_upload(vdb_params)

    return ingestor.ingest_tabular(tabular_params)
