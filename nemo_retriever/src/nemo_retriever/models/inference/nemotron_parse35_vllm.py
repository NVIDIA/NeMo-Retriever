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
    """Expose pinned mBART target states and tied weights through vLLM's DFlash interface."""

    _aux_hidden_state_layers: tuple[int, ...] = ()

    @property
    def model(self):
        """Expose the decoder embeddings for vLLM's target/draft weight sharing."""
        return self.decoder

    def get_language_model(self):
        """Keep both the decoder and the tied output head visible to the drafter."""
        return self

    def _embed_text_input_ids(self, input_ids, embed_input_ids, *, is_multimodal):
        """Retain vLLM's multimodal masking while delegating text to the decoder."""
        return super()._embed_text_input_ids(
            input_ids,
            self.decoder.embed_input_ids,
            is_multimodal=is_multimodal,
        )

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        """Select post-block states using vLLM's one-based layer convention."""
        if (
            not layers
            or tuple(sorted(set(layers))) != layers
            or not all(1 <= index <= len(self.decoder.layers) for index in layers)
        ):
            raise ValueError("DFlash2 auxiliary layers must be ordered, unique decoder layer numbers.")
        self._aux_hidden_state_layers = layers

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        """Match the pinned drafter's zero-based target layers 1, 4, and 8."""
        return (2, 5, 9)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        """Return normal target logits input together with the requested block states."""
        if not self._aux_hidden_state_layers:
            return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
        if inputs_embeds is None:
            inputs_embeds = self.decoder.embed_tokens(input_ids)
        hidden = self.decoder.layernorm_embedding(inputs_embeds)
        auxiliary = []
        for index, layer in enumerate(self.decoder.layers, start=1):
            hidden = layer(hidden)
            if index in self._aux_hidden_state_layers:
                auxiliary.append(hidden)
        return self.decoder.layer_norm(hidden), auxiliary
