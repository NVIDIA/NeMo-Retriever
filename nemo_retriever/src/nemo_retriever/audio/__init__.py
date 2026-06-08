# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Audio pipeline: media chunking (MediaChunkActor) and ASR (ASRActor).

Provides the same semantics as `nemo_retriever.api` dataloader + Parakeet for
batch and inprocess ingestion run modes.
"""

from __future__ import annotations

from nemo_retriever.operators.extract.audio.asr_actor import ASRActor
from nemo_retriever.operators.extract.audio.asr_actor import asr_params_from_env
from nemo_retriever.operators.extract.audio.cpu_actor import ASRCPUActor
from nemo_retriever.operators.extract.audio.gpu_actor import ASRGPUActor
from nemo_retriever.operators.extract.audio.chunk_actor import MediaChunkActor
from nemo_retriever.common.modality.audio.media_interface import MediaInterface
from nemo_retriever.common.params import ASRParams
from nemo_retriever.common.params import AudioChunkParams

from nemo_retriever.cli.audio.cli import app

__all__ = [
    "ASRActor",
    "ASRCPUActor",
    "ASRGPUActor",
    "ASRParams",
    "app",
    "asr_params_from_env",
    "AudioChunkParams",
    "MediaChunkActor",
    "MediaInterface",
]
