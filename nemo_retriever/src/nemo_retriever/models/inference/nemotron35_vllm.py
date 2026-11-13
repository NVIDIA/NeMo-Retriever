# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-safe registration and checkpoint validation for N3.5 retrieval."""

from typing import Any

NEMOTRON35_VLLM_ARCHITECTURE = "NRLNemotron35EmbeddingModel"


def register_nemotron35_vllm() -> None:
    """Register only the NRL architecture, without importing CUDA model code.

    Parameters
    ----------
    None
        Registers the project's fixed architecture name and lazy worker import path.

    Returns
    -------
    None
        The worker can construct the architecture through vLLM's model registry.

    Raises
    ------
    ImportError
        If vLLM is not installed in the current environment.
    """
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        NEMOTRON35_VLLM_ARCHITECTURE,
        "nemo_retriever.models.local.nemotron35_vllm:NRLNemotron35EmbeddingModel",
    )
    from nemo_retriever.common import nvtx_layers

    nvtx_layers.install("vllm")


def validate_bidirectional_config(config: Any, text_config: Any) -> None:
    """Require the bidirectional Mistral3/Ministral3 retrieval configuration.

    Raises:
        ValueError: If the checkpoint architecture or causal flag is unsupported.
    """
    if (
        config.model_type != "mistral3"
        or text_config.model_type != "ministral3"
        or getattr(text_config, "is_causal", None) is not False
    ):
        raise ValueError("The N3.5 vLLM adapter requires a bidirectional Mistral3 retrieval checkpoint.")
