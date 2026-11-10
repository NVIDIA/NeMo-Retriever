# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scoped bidirectional attention and trained N3.5 classification-head loading."""

from collections.abc import Iterable

import torch
from vllm.model_executor.layers.attention import EncoderOnlyAttention
from vllm.model_executor.models.transformers import TransformersMultiModalForSequenceClassification

from nemo_retriever.models.inference.nemotron35_vllm import validate_bidirectional_config


class NRLNemotron35RerankModel(TransformersMultiModalForSequenceClassification):
    """Use mean pooling and the checkpoint's trained head without a sigmoid."""

    default_seq_pooling_type: str = "MEAN"

    def _get_attn_cls(self) -> type[EncoderOnlyAttention]:
        """Select bidirectional attention using the vLLM 0.32 construction hook.

        Raises:
            ValueError: If checkpoint attention or pooling configuration is invalid.
        """
        validate_bidirectional_config(self.config, self.text_config)
        if getattr(self.text_config, "pooling", None) != "avg":
            raise ValueError("The N3.5 reranker requires mean pooling.")
        return EncoderOnlyAttention

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load all weights and map the trained scoring head to vLLM's head.

        Args:
            weights: Checkpoint tensor names and values.

        Returns:
            Names of weights successfully loaded by vLLM.

        Raises:
            RuntimeError: If the trained classification head was not loaded.
            ValueError: If checkpoint tensors are incompatible.
        """
        mapped = (
            ("classifier.weight" if name == "language_model.score.weight" else name, value) for name, value in weights
        )
        loaded = super().load_weights(mapped)
        if "classifier.weight" not in loaded:
            raise RuntimeError("Nemotron 3.5 reranker trained classification head was not loaded.")
        return loaded
