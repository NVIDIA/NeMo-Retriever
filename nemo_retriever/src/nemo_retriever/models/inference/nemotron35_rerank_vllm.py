# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-safe registration of the NRL bidirectional N3.5 reranking model."""


def register_nemotron35_rerank_vllm() -> None:
    """Register the reranker architecture without loading weights or CUDA.

    Returns:
        None. vLLM workers can import the model through the registered path.

    Raises:
        ImportError: If vLLM is not installed.
    """
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "NRLNemotron35RerankModel",
        "nemo_retriever.models.local.nemotron35_rerank_vllm:NRLNemotron35RerankModel",
    )
