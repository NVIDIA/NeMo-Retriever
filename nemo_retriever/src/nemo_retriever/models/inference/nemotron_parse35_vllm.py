# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-safe Parse 3.5 plugin registration and scoped DFlash2 target support."""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor
    from torch.nn import Module
    from vllm.sequence import IntermediateTensors

from nemo_retriever.common.params.utils import NEMOTRON_PARSE35_MODEL
from nemo_retriever.models.hf_model_registry import get_hf_revision

PARSE35_SNAPSHOT_ENV = "NRL_NEMOTRON_PARSE35_SNAPSHOT"


def register_nemotron_parse35_vllm() -> None:
    """Register the vendor runtime and scoped DFlash2 target in the current process.

    The adapter sets ``NRL_NEMOTRON_PARSE35_SNAPSHOT`` to its downloaded immutable
    snapshot before engine startup. vLLM invokes this entry point in workers too.
    Other vLLM models leave it unset, so importing their plugins does no Parse work.
    No package is installed and no vendor code is copied into NRL.

    Args:
        None. The snapshot path is inherited through the process environment.

    Returns:
        None. Registers the official plugin only when a Parse snapshot is selected.

    Raises:
        RuntimeError: If the snapshot is not the pinned revision, a different
            plugin is already imported, or the vendor runtime guard rejects versions.
        ImportError: If the snapshot plugin or its runtime dependencies cannot load.
        OSError: If the plugin files cannot be read.
    """
    snapshot = os.environ.get(PARSE35_SNAPSHOT_ENV)
    if not snapshot:
        return
    root = Path(snapshot).resolve()
    if root.name != get_hf_revision(NEMOTRON_PARSE35_MODEL):
        raise RuntimeError("Parse 3.5 vLLM plugin must use the pinned checkpoint snapshot.")
    package_path = root / "vllm_tied_patch" / "__init__.py"
    existing = sys.modules.get("vllm_tied_patch")
    if existing is not None:
        if Path(existing.__file__).resolve() != package_path.resolve():
            raise RuntimeError("A different Parse vLLM plugin is already imported; use a fresh process.")
    else:
        spec = importlib.util.spec_from_file_location("vllm_tied_patch", package_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load the Parse 3.5 plugin at {package_path}")
        package = importlib.util.module_from_spec(spec)
        sys.modules["vllm_tied_patch"] = package
        spec.loader.exec_module(package)
    plugin = importlib.import_module("vllm_tied_patch.plugin")
    plugin.register()
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(
        "NRLNemotronParse35DFlashModel",
        "nemo_retriever.models.local.nemotron_parse35_dflash:NRLNemotronParse35DFlashModel",
    )


class Parse35DFlashMixin:
    """Expose pinned mBART target states and tied weights through vLLM's DFlash interface.

    The vendor model supplies ``decoder`` and ``lm_head``. This mixin retains
    its text embedding and normalization operations while exposing intermediate
    decoder states to the drafter.
    """

    decoder: Module
    lm_head: Module
    _aux_hidden_state_layers: tuple[int, ...] = ()

    @property
    def model(self) -> Module:
        """Expose the decoder for target/draft embedding-weight sharing.

        Returns:
            The vendor decoder module, including its token embeddings.

        Raises:
            AttributeError: If the vendor model has not initialized its decoder.
        """
        return self.decoder

    def get_language_model(self) -> Self:
        """Expose both the decoder and the tied output head to the drafter.

        Returns:
            This target model, whose ``model`` property exposes the decoder and
            whose ``lm_head`` attribute supplies the shared output head.
        """
        return self

    def _embed_text_input_ids(
        self,
        input_ids: Tensor,
        embed_input_ids: Callable[[Tensor], Tensor],
        *,
        is_multimodal: Tensor | None,
    ) -> Tensor:
        """Retain vLLM's multimodal masking while embedding text with the decoder.

        Args:
            input_ids: Token IDs to embed.
            embed_input_ids: Callback supplied by vLLM's multimodal interface.
                Replace it with the decoder callback to avoid target recursion.
            is_multimodal: Optional mask identifying multimodal placeholder tokens.

        Returns:
            Token embeddings after the inherited multimodal masking logic.

        Raises:
            AttributeError: If the decoder or inherited embedding interface is unavailable.
            IndexError: If an unmasked token ID is outside the embedding vocabulary.
            RuntimeError: If a backend embedding operation fails.
        """
        return super()._embed_text_input_ids(
            input_ids,
            self.decoder.embed_input_ids,
            is_multimodal=is_multimodal,
        )

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        """Select post-block states using vLLM's one-based layer convention.

        Args:
            layers: Nonempty, increasing tuple of unique decoder layer numbers,
                from one through the number of decoder layers.

        Returns:
            None. Updates the layers collected by subsequent forward calls.

        Raises:
            ValueError: If the tuple is empty, unordered, duplicated, or out of range.
            AttributeError: If the vendor decoder has not been initialized.
        """
        if (
            not layers
            or tuple(sorted(set(layers))) != layers
            or not all(1 <= index <= len(self.decoder.layers) for index in layers)
        ):
            raise ValueError("DFlash2 auxiliary layers must be ordered, unique decoder layer numbers.")
        self._aux_hidden_state_layers = layers

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        """Return the auxiliary layers expected by the pinned DFlash2 checkpoint.

        Returns:
            One-based post-block indices ``(2, 5, 9)``, corresponding to the
            drafter's zero-based target layer IDs ``(1, 4, 8)``.
        """
        return (2, 5, 9)

    def forward(
        self,
        input_ids: Tensor | None,
        positions: Tensor | None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: Tensor | None = None,
        **kwargs: object,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Compute target hidden states and the selected auxiliary decoder states.

        Args:
            input_ids: Token IDs, required when ``inputs_embeds`` is absent.
            positions: Position indices accepted by vLLM's interface; unused by
                the pinned mBART decoder.
            intermediate_tensors: Pipeline intermediates accepted for interface
                compatibility; unused by the pinned decoder.
            inputs_embeds: Optional input embeddings, including any vision prefix.
                When supplied, these replace token embedding lookup.
            **kwargs: Additional vLLM arguments forwarded to the vendor implementation
                when no auxiliary layers are selected, and otherwise unused.

        Returns:
            Normalized decoder hidden states when auxiliary output is disabled.
            Otherwise, a tuple of normalized hidden states and the requested
            post-block states, in layer order and before final normalization.
            Each tensor has shape ``(num_tokens, hidden_size)``.

        Raises:
            AttributeError: If the vendor decoder has not been initialized.
            TypeError: If neither token IDs nor input embeddings are supplied.
            IndexError: If a token ID is outside the embedding vocabulary.
            RuntimeError: If a backend embedding, attention, or normalization operation fails.
        """
        if not self._aux_hidden_state_layers:
            return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
        if inputs_embeds is None:
            inputs_embeds = self.decoder.embed_tokens(input_ids)
        hidden = self.decoder.layernorm_embedding(inputs_embeds)
        auxiliary: list[Tensor] = []
        for index, layer in enumerate(self.decoder.layers, start=1):
            hidden = layer(hidden)
            if index in self._aux_hidden_state_layers:
                auxiliary.append(hidden)
        return self.decoder.layer_norm(hidden), auxiliary
