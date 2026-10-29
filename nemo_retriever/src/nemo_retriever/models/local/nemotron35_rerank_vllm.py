# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scoped bidirectional attention and trained N3.5 classification-head loading."""

from collections.abc import Iterable

import torch
from vllm.model_executor.models.transformers import TransformersMultiModalForSequenceClassification
from vllm.model_executor.models.transformers.base import AttentionType, EncoderOnlyAttention, get_pp_indices

from nemo_retriever.models.inference.nemotron35_vllm import create_bidirectional_attention


class NRLNemotron35RerankModel(TransformersMultiModalForSequenceClassification):
    """Use mean pooling and the checkpoint's trained head without a sigmoid."""

    default_seq_pooling_type: str = "MEAN"

    def create_attention_instances(self) -> dict[int, EncoderOnlyAttention]:
        """Construct encoder-only attention for this bidirectional text tower.

        Returns:
            Attention layers indexed by global text-layer number.

        Raises:
            ValueError: If the architecture, pooling, or causal flags are invalid.
            RuntimeError: If the vLLM attention backend cannot initialize.
        """
        if getattr(self.text_config, "pooling", None) != "avg":
            raise ValueError("The N3.5 reranker requires mean pooling.")
        return create_bidirectional_attention(
            self,
            attention_cls=EncoderOnlyAttention,
            attention_type=AttentionType.ENCODER_ONLY,
            pp_indices=get_pp_indices,
        )

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
