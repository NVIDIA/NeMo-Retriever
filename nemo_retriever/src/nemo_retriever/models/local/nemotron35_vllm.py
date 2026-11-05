# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scoped bidirectional multimodal pooling model for the pinned vLLM runtime."""

from vllm.distributed.utils import get_pp_indices
from vllm.model_executor.layers.attention import EncoderOnlyAttention
from vllm.model_executor.models.transformers import TransformersMultiModalEmbeddingModel
from vllm.v1.attention.backend import AttentionType

from nemo_retriever.models.inference.nemotron35_vllm import (
    create_bidirectional_attention,
)


class NRLNemotron35EmbeddingModel(TransformersMultiModalEmbeddingModel):
    """Mean-pool a bidirectional Mistral3 text backbone with native image processing."""

    default_seq_pooling_type = "MEAN"

    def extra_repr(self) -> str:
        """Return checkpoint causality and constructed attention classes for inspection.

        Parameters
        ----------
        None
            Uses the initialized model's text configuration and attention instances.

        Returns
        -------
        str
            Compact text-attention description for the PyTorch model representation.

        Raises
        ------
        AttributeError
            If called before text configuration or attention instances are initialized.
        """
        instances = getattr(self, "attention_instances", None)
        layers = instances.values() if instances is not None else self.modules()
        attention_classes = sorted(
            {type(layer).__name__ for layer in layers if isinstance(layer, EncoderOnlyAttention)}
        )
        return f"text_is_causal={self.text_config.is_causal}, attention={','.join(attention_classes)}"

    def _get_attn_cls(self) -> type[EncoderOnlyAttention]:
        """Select bidirectional text attention for the current vLLM construction hook.

        Returns:
            Encoder-only attention, with layer construction owned by vLLM.

        Raises:
            ValueError: If the checkpoint is not a bidirectional Mistral3 retrieval model.
        """
        text = self.text_config
        if (
            self.config.model_type != "mistral3"
            or text.model_type != "ministral3"
            or getattr(text, "is_causal", None) is not False
        ):
            raise ValueError("The N3.5 vLLM adapter requires a bidirectional Mistral3 retrieval checkpoint.")
        return EncoderOnlyAttention

    def create_attention_instances(self) -> dict[int, EncoderOnlyAttention]:
        """Create shard-local attention for vLLM's legacy construction hook.

        Parameters
        ----------
        None
            Uses this model's initialized backend, cache, and parallel settings.

        Returns
        -------
        dict[int, Any]
            Global text-layer indices mapped to EncoderOnlyAttention instances.

        Raises
        ------
        ValueError
            If the checkpoint does not request bidirectional Mistral3/Ministral3 attention.
        RuntimeError
            If backend attention construction fails.
        """
        return create_bidirectional_attention(
            self,
            attention_cls=EncoderOnlyAttention,
            attention_type=AttentionType.ENCODER_ONLY,
            pp_indices=get_pp_indices,
        )
