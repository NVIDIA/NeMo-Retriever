# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-scoped DFlash2 target interface for the pinned Parse 3.5 plugin."""

from vllm.model_executor.models.interfaces import SupportsEagle3
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm_tied_patch.vllm_radio_mbart_prefix import (
    RadioMBartPrefixDummyInputsBuilder,
    RadioMBartPrefixForCausalLM,
    RadioMBartPrefixProcessingInfo,
    RadioMBartPrefixProcessor,
)


from nemo_retriever.models.inference.nemotron_parse35_vllm import Parse35DFlashMixin


@MULTIMODAL_REGISTRY.register_processor(
    RadioMBartPrefixProcessor,
    info=RadioMBartPrefixProcessingInfo,
    dummy_inputs=RadioMBartPrefixDummyInputsBuilder,
)
class NRLNemotronParse35DFlashModel(Parse35DFlashMixin, RadioMBartPrefixForCausalLM, SupportsEagle3):
    """Expose intermediate decoder states and tied weights to vLLM's DFlash2 drafter.

    Default Parse uses the vendor class. Only an explicit DFlash2 request
    selects this subclass; no vendor classes or global attention hooks change.
    """
