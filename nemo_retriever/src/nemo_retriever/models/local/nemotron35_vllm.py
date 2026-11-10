# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scoped bidirectional multimodal pooling model for the pinned vLLM runtime."""

import torch
from vllm.model_executor.layers.attention import EncoderOnlyAttention
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.transformers import TransformersMultiModalEmbeddingModel

from nemo_retriever.models.inference.nemotron35_vllm import (
    validate_bidirectional_config,
)


def _install_single_image_flash_mask() -> None:
    """Let Pixtral run one-image vision calls without a block mask so SDPA can select flash attention."""
    from transformers.models.pixtral import modeling_pixtral

    generate_block_attention_mask = modeling_pixtral.generate_block_attention_mask
    if getattr(generate_block_attention_mask, "nrl_single_image", False):
        return

    def generate(patch_embeds_list, tensor):
        # One image is one all-visible block, so the dense mask adds nothing.
        if len(patch_embeds_list) == 1 and tensor.shape[0] == 1:
            return None
        return generate_block_attention_mask(patch_embeds_list, tensor)

    generate.nrl_single_image = True
    modeling_pixtral.generate_block_attention_mask = generate


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
            If called before the text configuration is initialized.
        """
        attention_classes = sorted(
            {type(layer).__name__ for layer in self.modules() if isinstance(layer, EncoderOnlyAttention)}
        )
        return f"text_is_causal={self.text_config.is_causal}, attention={','.join(attention_classes)}"

    def _get_attn_cls(self) -> type[EncoderOnlyAttention]:
        """Select bidirectional text attention for the current vLLM construction hook.

        Returns:
            Encoder-only attention, with layer construction owned by vLLM.

        Raises:
            ValueError: If the checkpoint is not a bidirectional Mistral3 retrieval model.
        """
        validate_bidirectional_config(self.config, self.text_config)
        return EncoderOnlyAttention

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """Run the vision tower once per image and return embeddings in input order.

        Parameters
        ----------
        **kwargs
            vLLM multimodal inputs; ``pixel_values`` or ``image_patches`` and
            ``num_image_patches`` describe a batch of images.

        Returns
        -------
        MultiModalEmbeddings
            Per-image embeddings from the parent implementation, concatenated in input order.

        Raises
        ------
        RuntimeError
            If the vision tower or projector fails.
        """
        _install_single_image_flash_mask()
        pixel_values = kwargs.get("pixel_values", kwargs.get("image_patches"))
        num_image_patches = kwargs.get("num_image_patches")
        if kwargs.get("image_embeds") is not None or pixel_values is None or num_image_patches is None:
            return super().embed_multimodal(**kwargs)
        sizes = [int(n) for n in torch.as_tensor(num_image_patches).flatten().tolist()]
        if len(sizes) <= 1:
            return super().embed_multimodal(**kwargs)
        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size)

        def item(value: object, index: int) -> object:
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                if value.shape[0] == len(sizes):
                    return value[index : index + 1]
                if value.shape[0] == offsets[-1]:
                    return value[offsets[index] : offsets[index + 1]]
            elif isinstance(value, (list, tuple)) and len(value) == len(sizes):
                return type(value)([value[index]])
            return value

        outputs: list[torch.Tensor] = []
        for index in range(len(sizes)):
            embeddings = super().embed_multimodal(**{key: item(value, index) for key, value in kwargs.items()})
            outputs.extend(embeddings if isinstance(embeddings, (list, tuple)) else [embeddings])
        return outputs
