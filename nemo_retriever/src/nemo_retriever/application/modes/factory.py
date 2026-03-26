# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_retriever.params import IngestorCreateParams
from nemo_retriever.params import RunMode


def create_runmode_ingestor(*, run_mode: RunMode = "inprocess", params: IngestorCreateParams | None = None):
    p = params or IngestorCreateParams()
    if run_mode == "inprocess":
        from nemo_retriever.ingest_modes.inprocess import InProcessIngestor
        from nemo_retriever.inference_presets import resolve_inference_preset

        extract_defaults, embed_defaults, caption_defaults = resolve_inference_preset(p.inference)
        init_kwargs: dict = {"documents": p.documents}
        if extract_defaults:
            init_kwargs["default_extract_kwargs"] = extract_defaults
        if embed_defaults:
            init_kwargs["default_embed_kwargs"] = embed_defaults
        if caption_defaults:
            init_kwargs["default_caption_kwargs"] = caption_defaults
        return InProcessIngestor(**init_kwargs)
    if run_mode == "batch":
        from nemo_retriever.ingest_modes.batch import BatchIngestor

        return BatchIngestor(
            documents=p.documents,
            ray_address=p.ray_address,
            ray_log_to_driver=p.ray_log_to_driver,
            debug=p.debug,
            allow_no_gpu=p.allow_no_gpu,
        )
    if run_mode == "fused":
        from nemo_retriever.ingest_modes.fused import FusedIngestor

        return FusedIngestor(
            documents=p.documents,
            ray_address=p.ray_address,
            ray_log_to_driver=p.ray_log_to_driver,
        )
    if run_mode == "online":
        from nemo_retriever.ingest_modes.online import OnlineIngestor

        return OnlineIngestor(documents=p.documents, base_url=p.base_url)
    raise ValueError(f"Unknown run_mode: {run_mode!r}")
