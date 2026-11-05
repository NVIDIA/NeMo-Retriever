# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scoped bidirectional attention and trained N3.5 classification-head loading."""

from collections.abc import Iterable

import torch
from vllm.model_executor.models.transformers import TransformersMultiModalForSequenceClassification
from vllm.distributed.utils import get_pp_indices
from vllm.model_executor.layers.attention import EncoderOnlyAttention
from vllm.v1.attention.backend import AttentionType

from nemo_retriever.models.inference.nemotron35_vllm import create_bidirectional_attention


class NRLNemotron35RerankModel(TransformersMultiModalForSequenceClassification):
    """Use mean pooling and the checkpoint's trained head without a sigmoid."""

    default_seq_pooling_type: str = "MEAN"

    def _get_attn_cls(self) -> type[EncoderOnlyAttention]:
        """Select bidirectional attention using the vLLM 0.32 construction hook.

        Raises:
            ValueError: If checkpoint attention or pooling configuration is invalid.
        """
        text = self.text_config
        if (
            self.config.model_type != "mistral3"
            or text.model_type != "ministral3"
            or getattr(text, "is_causal", None) is not False
        ):
            raise ValueError("The N3.5 vLLM adapter requires a bidirectional Mistral3 retrieval checkpoint.")
        if getattr(text, "pooling", None) != "avg":
            raise ValueError("The N3.5 reranker requires mean pooling.")
        return EncoderOnlyAttention

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
