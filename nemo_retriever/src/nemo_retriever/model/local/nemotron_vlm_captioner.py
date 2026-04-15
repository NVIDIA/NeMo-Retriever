# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, List, Optional

from PIL import Image

from nemo_retriever.utils.hf_cache import configure_global_hf_cache_base
from nemo_retriever.utils.nvtx import gpu_inference_range
from ..model import BaseModel, RunMode


def _b64_to_pil(b64: str) -> Image.Image:
    """Decode a base64-encoded image string to a PIL Image."""
    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")


# Bidirectional alias maps: short NIM names ↔ full HuggingFace paths.
_NIM_TO_HF: dict[str, str] = {
    "nvidia/nemotron-nano-12b-v2-vl": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
    "nvidia/nemotron-nano-12b-v2-vl-bf16": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
    "nvidia/nemotron-nano-12b-v2-vl-fp8": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8",
    "nvidia/nemotron-nano-12b-v2-vl-nvfp4-qad": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD",
}
_HF_TO_NIM: dict[str, str] = {
    "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16": "nvidia/nemotron-nano-12b-v2-vl",
    "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8": "nvidia/nemotron-nano-12b-v2-vl-fp8",
    "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD": "nvidia/nemotron-nano-12b-v2-vl-nvfp4-qad",
}


def resolve_caption_model_name(name: str, *, target: str = "local") -> str:
    """Normalize a caption model name for the given target.

    Parameters
    ----------
    name : str
        Model name (NIM short form or full HuggingFace path).
    target : str
        ``"local"`` returns the full HF path, ``"remote"`` returns the
        short NIM endpoint name.
    """
    lower = name.lower()
    if target == "local":
        return _NIM_TO_HF.get(lower, name)
    # remote
    return _HF_TO_NIM.get(name, _HF_TO_NIM.get(_NIM_TO_HF.get(lower, name), name))


class NemotronVLMCaptioner(BaseModel):
    """
    Local VLM captioner wrapping Nemotron Nano 12B v2 VL variants.

    Supported models:

    * ``nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16`` (default, BFloat16)
    * ``nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8``  (FP8 quantised)
    * ``nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD`` (NVFP4 quantised,
      requires GPU compute capability >= 8.9, e.g. Ada Lovelace / Hopper)

    Uses vLLM for inference with batched scheduling.

    Usage::

        captioner = NemotronVLMCaptioner()
        captions = captioner.caption_batch(
            ["<base64-png>", "<base64-png>"],
            prompt="Caption the content of this image:",
        )
    """

    SUPPORTED_MODELS: dict[str, str] = {
        "BF16": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
        "FP8": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8",
        "NVFP4-QAD": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD",
    }

    MODEL_ALIASES: dict[str, str] = _NIM_TO_HF

    # Pinned HF revision (commit SHA) per model to ensure reproducibility.
    _MODEL_REVISIONS: dict[str, str] = {
        "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16": "5d250e2e111dc5e1434131bdf3d590c27a878ade",
        "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8": "7394488badb786e1decc0e00e308de1cab9560e6",
        "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD": "b8d3c170d9ee3a078917ef9bfd508eff988d6de7",
    }

    # Map model-name suffixes to vLLM engine kwargs.
    # The FP8 HF config ships with quant_method="modelopt" which triggers
    # vLLM's ModelOptFp8Config (SM89+).  Override to quant_method="fp8" in
    # the HF config so vLLM uses its plain FP8 handler (SM80+).
    _QUANTIZATION_PROFILES: dict[str, dict[str, Any]] = {
        "BF16": {"dtype": "bfloat16"},
        "FP8": {
            "dtype": "auto",
            "quantization": "fp8",
            "hf_overrides": {"quantization_config": {"quant_method": "fp8", "activation_scheme": "static"}},
        },
        "NVFP4-QAD": {"dtype": "auto", "quantization": "modelopt"},
    }

    def __init__(
        self,
        model_path: str = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
        device: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
        max_new_tokens: int = 1024,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.5,
    ) -> None:
        super().__init__()

        # Resolve short aliases to full model paths.
        model_path = self.MODEL_ALIASES.get(model_path.lower(), model_path)

        valid_models = list(self.SUPPORTED_MODELS.values())
        if model_path not in valid_models:
            raise ValueError(
                f"Unknown caption model: {model_path!r}\n"
                f"Supported models:\n" + "\n".join(f"  - {m}" for m in valid_models)
            )

        try:
            from vllm import LLM, SamplingParams  # noqa: F401
        except ImportError as e:
            raise ImportError(
                'Local VLM captioning requires vLLM. Install with: pip install "nemo-retriever[vlm-caption]"'
            ) from e

        self._model_path = model_path
        self._max_new_tokens = max_new_tokens

        if device is not None:
            # vLLM uses CUDA_VISIBLE_DEVICES rather than a torch device string.
            # Translate e.g. "cuda:1" → "1" so vLLM sees only the requested GPU.
            import os

            dev_id = device.split(":")[-1] if ":" in device else device
            os.environ["CUDA_VISIBLE_DEVICES"] = dev_id

        configure_global_hf_cache_base(hf_cache_dir)

        revision = self._MODEL_REVISIONS.get(model_path)

        # Pick vLLM engine kwargs based on the model variant.
        engine_kwargs: dict[str, Any] = {"dtype": "bfloat16"}  # fallback
        model_upper = model_path.upper()
        for suffix, profile in self._QUANTIZATION_PROFILES.items():
            if model_upper.endswith(suffix):
                engine_kwargs = profile
                break

        self._llm = LLM(
            model=model_path,
            revision=revision,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            **engine_kwargs,
        )

    def _build_messages(
        self,
        base64_image: str,
        *,
        prompt: str,
        system_prompt: Optional[str],
    ) -> list[dict[str, Any]]:
        """Build chat messages in OpenAI format for vLLM."""
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        )
        return messages

    def caption(
        self,
        base64_image: str,
        *,
        prompt: str = "Caption the content of this image:",
        system_prompt: Optional[str] = "/no_think",
        temperature: float = 1.0,
    ) -> str:
        """Generate a caption for a single base64-encoded image."""
        return self.caption_batch([base64_image], prompt=prompt, system_prompt=system_prompt, temperature=temperature)[
            0
        ]

    def caption_batch(
        self,
        base64_images: List[str],
        *,
        prompt: str = "Caption the content of this image:",
        system_prompt: Optional[str] = "/no_think",
        temperature: float = 1.0,
    ) -> List[str]:
        """Generate captions for a list of base64-encoded images.

        vLLM batches internally and handles scheduling across images.
        """
        from vllm import SamplingParams

        conversations = [self._build_messages(b64, prompt=prompt, system_prompt=system_prompt) for b64 in base64_images]
        sampling_params = SamplingParams(temperature=temperature, max_tokens=self._max_new_tokens)
        with gpu_inference_range("NemotronVLMCaptioner", batch_size=len(conversations)):
            outputs = self._llm.chat(conversations, sampling_params=sampling_params)
        return [out.outputs[0].text.strip() for out in outputs]

    # ---- BaseModel abstract interface ----

    @property
    def model_name(self) -> str:
        return self._model_path

    @property
    def model_type(self) -> str:
        return "vlm-captioner"

    @property
    def model_runmode(self) -> RunMode:
        return "local"

    @property
    def input(self) -> Any:
        return {
            "type": "image",
            "format": "base64",
            "description": "Base64-encoded image for captioning.",
        }

    @property
    def output(self) -> Any:
        return {
            "type": "text",
            "format": "string",
            "description": "Generated caption for the input image.",
        }

    @property
    def input_batch_size(self) -> int:
        return 1
