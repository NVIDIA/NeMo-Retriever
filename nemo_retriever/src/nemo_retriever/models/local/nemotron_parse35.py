# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned Transformers and vLLM implementations of local Parse 3.5 inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

import torch
from PIL import Image

from nemo_retriever.common.params.utils import NEMOTRON_PARSE35_MODEL
from nemo_retriever.models.hf_cache import configure_global_hf_cache_base
from nemo_retriever.models.hf_model_registry import NEMOTRON_PARSE35_DFLASH2_REVISION, get_hf_revision
from nemo_retriever.models.inference.parse_image import ImageInput, normalize_parse_image
from nemo_retriever.models.model import BaseModel, ModelRunMode

_DEFAULT_TASK_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"


class NemotronParse35(BaseModel):
    """Parse page images with the checkpoint's native Transformers or vLLM backend.

    Weights and custom code are loaded lazily from an immutable registry revision.
    Pages are converted and generated sequentially to bound additional image
    and GPU memory independently of batch size. Local vLLM uses the legacy
    pinned plugin by default. Explicitly enabling DFlash2 loads its drafter
    from the same pinned snapshot as the target in a compatible native runtime.
    """

    def __init__(
        self,
        device: str = "cuda",
        hf_cache_dir: str | None = None,
        task_prompt: str = _DEFAULT_TASK_PROMPT,
        max_tokens: int = 9000,
        backend: Literal["transformers", "vllm"] = "vllm",
        sampling_params: dict[str, Any] | None = None,
        use_dflash: bool = False,
        num_speculative_tokens: Literal[8, 16, 32] = 32,
    ) -> None:
        """Configure a lazy BF16 parser without loading weights.

        Args:
            device: PyTorch inference device, normally the worker's ``cuda`` device.
                Select ``cpu`` with ``backend="transformers"`` when CUDA is unavailable.
            hf_cache_dir: Optional Hugging Face snapshot cache directory.
            task_prompt: Native control tokens. Picture text is enabled by default.
            max_tokens: Positive maximum number of generated tokens per page.
            backend: In-process vLLM (the default) or explicit native Transformers.
                vLLM requires CUDA device zero
                in the worker's visible GPU set; Transformers also supports CPU.

            sampling_params: Optional vLLM overrides for ``temperature``, ``top_k``,
                ``top_p``, ``repetition_penalty``, and ``seed``. Defaults retain
                greedy decoding. Available only with backend="vllm".
            use_dflash: Enable native DFlash2 in the local vLLM engine (default: False).
                Requires a vLLM build with native Parse and DFlash2 support.
                False selects the legacy pinned plugin. Transformers ignores it.
            num_speculative_tokens: Draft length, one of 8, 16, or 32 (default).
                Applies only to native DFlash2, not remote services or Transformers.

        Raises:
            ValueError: If sampling overrides are unsupported, max_tokens is not positive, the backend is unsupported,
                or the device is invalid for the selected backend.
            RuntimeError: If PyTorch cannot interpret the device.
        """
        super().__init__()
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        if backend not in ("transformers", "vllm"):
            raise ValueError(f"Unsupported Parse 3.5 backend: {backend}")
        self._device = torch.device(device)
        if backend == "vllm" and (self._device.type != "cuda" or self._device.index not in (None, 0)):
            raise ValueError('Parse 3.5 vLLM requires device="cuda" or "cuda:0" in the worker GPU set.')
        allowed_sampling = {"temperature", "top_k", "top_p", "repetition_penalty", "seed"}
        if sampling_params and (backend != "vllm" or set(sampling_params) - allowed_sampling):
            raise ValueError(
                "sampling_params requires vllm and accepts temperature, top_k, top_p, repetition_penalty, seed."
            )
        if num_speculative_tokens not in (8, 16, 32):
            raise ValueError("Parse DFlash2 num_speculative_tokens must be 8, 16, or 32.")
        self._use_dflash = backend == "vllm" and use_dflash
        self._num_speculative_tokens = num_speculative_tokens
        self._sampling_params = dict(sampling_params or {})
        self._backend = backend
        self._hf_cache_dir = hf_cache_dir
        self._task_prompt = task_prompt
        self._max_tokens = max_tokens
        self._model: Any = None
        self._processor: Any = None
        self._generation_config: Any = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if self._device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"Nemotron 3.5 Parse requested device {self._device}, but CUDA is unavailable. "
                    'Use a CUDA-enabled host or explicitly select backend="transformers", device="cpu".'
                )
            if self._device.index is not None and self._device.index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"Nemotron 3.5 Parse requested unavailable CUDA device {self._device}. "
                    'Select a visible CUDA device or explicitly select device="cpu".'
                )
        if self._use_dflash:
            self._validate_native_dflash_runtime()
        from huggingface_hub import snapshot_download

        cache_dir = configure_global_hf_cache_base(self._hf_cache_dir)
        snapshot = snapshot_download(
            NEMOTRON_PARSE35_MODEL,
            revision=(
                NEMOTRON_PARSE35_DFLASH2_REVISION if self._use_dflash else get_hf_revision(NEMOTRON_PARSE35_MODEL)
            ),
            cache_dir=cache_dir,
            allow_patterns=["*.json", "*.py", "*.safetensors", "*.jinja", "*.model", "*.txt", "dflash2/*"],
        )
        if self._backend == "vllm":
            if self._use_dflash:
                self._load_native_dflash(snapshot)
            else:
                self._load_vllm(snapshot)
            return
        from transformers import AutoModel, AutoProcessor, GenerationConfig

        # Loading all assets from the resolved snapshot also keeps nested processor
        # downloads on the same revision and in the configured cache.
        processor = AutoProcessor.from_pretrained(snapshot, trust_remote_code=True)
        generation_config = GenerationConfig.from_pretrained(snapshot)
        generation_config.max_new_tokens = self._max_tokens
        model = AutoModel.from_pretrained(
            snapshot, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        self._model = model
        try:
            model.eval().to(self._device)
        except (RuntimeError, ValueError):
            self.unload()
            raise
        self._processor = processor
        self._generation_config = generation_config

    def _validate_native_dflash_runtime(self) -> None:
        try:
            from vllm import ModelRegistry

            supported = set(ModelRegistry.get_supported_archs())
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "Local Parse DFlash2 requires vLLM with native RadioMBartPrefixForCausalLM and "
                "DFlash2DraftModel support. Use a compatible native vLLM environment; the NRL local "
                "extra pins legacy vLLM 0.26. Alternatively set ParseParams(use_dflash=False)."
            ) from exc
        if not {"RadioMBartPrefixForCausalLM", "DFlash2DraftModel"}.issubset(supported):
            raise RuntimeError(
                "This vLLM build lacks native Parse/DFlash2 support. Use a vLLM build that registers "
                "RadioMBartPrefixForCausalLM and DFlash2DraftModel, or set ParseParams(use_dflash=False) "
                "for the legacy vLLM 0.26 plugin."
            )

    def _load_native_dflash(self, snapshot: str) -> None:
        import json
        import os

        from nemo_retriever.models.inference.nemotron_parse35_vllm import PARSE35_SNAPSHOT_ENV
        from nemo_retriever.models.inference.vllm import apply_vllm_startup_defaults
        from vllm import LLM, SamplingParams

        draft = Path(snapshot) / "dflash2"
        if not (draft / "config.json").is_file() or not (draft / "model.safetensors").is_file():
            raise RuntimeError(
                "Parse DFlash2 drafter assets are missing. Cache the complete pinned Parse snapshot, "
                "including dflash2/config.json and dflash2/model.safetensors, before offline inference."
            )
        config = json.loads((draft / "config.json").read_text())
        if "DFlash2DraftModel" not in config.get("architectures", []):
            raise RuntimeError("The pinned Parse dflash2 directory must contain a DFlash2DraftModel checkpoint.")
        apply_vllm_startup_defaults()
        previous = {
            name: os.environ.get(name)
            for name in (
                "VLLM_WORKER_MULTIPROC_METHOD",
                PARSE35_SNAPSHOT_ENV,
                "NP_VLLM_MTP",
            )
        }
        try:
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            os.environ.pop(PARSE35_SNAPSHOT_ENV, None)
            os.environ.pop("NP_VLLM_MTP", None)
            self._generation_config = SamplingParams(
                **{"temperature": 0, "top_k": 1, "repetition_penalty": 1.1, **self._sampling_params},
                max_tokens=self._max_tokens,
                skip_special_tokens=False,
            )
            self._model = LLM(
                model=snapshot,
                trust_remote_code=True,
                dtype="bfloat16",
                distributed_executor_backend="uni",
                max_model_len=16384,
                max_num_seqs=1,
                max_num_batched_tokens=16384,
                limit_mm_per_prompt={"image": 1},
                enable_prefix_caching=False,
                attention_config={"flash_attn_version": 3},
                gpu_memory_utilization=0.8,
                speculative_config={
                    "method": "dflash",
                    "model": str(draft),
                    "num_speculative_tokens": self._num_speculative_tokens,
                },
                compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"},
            )
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def _load_vllm(self, snapshot: str) -> None:
        import os

        from nemo_retriever.models.inference.nemotron_parse35_vllm import (
            PARSE35_SNAPSHOT_ENV,
            register_nemotron_parse35_vllm,
        )
        from nemo_retriever.models.inference.vllm import apply_vllm_startup_defaults

        apply_vllm_startup_defaults()
        # CUDA availability checks and other extraction stages may initialize
        # CUDA on the actor before engine creation. Fork cannot reuse that state.
        previous_environment = {
            name: os.environ.get(name) for name in (PARSE35_SNAPSHOT_ENV, "VLLM_WORKER_MULTIPROC_METHOD")
        }
        try:
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            os.environ[PARSE35_SNAPSHOT_ENV] = snapshot
            register_nemotron_parse35_vllm()
            from vllm import LLM, SamplingParams

            self._generation_config = SamplingParams(
                **{
                    "temperature": 0,
                    "top_k": 1,
                    "repetition_penalty": 1.1,
                    **self._sampling_params,
                },
                max_tokens=self._max_tokens,
                skip_special_tokens=False,
            )
            self._model = LLM(
                model=snapshot,
                trust_remote_code=True,
                dtype="bfloat16",
                distributed_executor_backend="uni",
                max_model_len=16384,
                max_num_seqs=1,
                max_num_batched_tokens=16384,
                limit_mm_per_prompt={"image": 1},
                enable_prefix_caching=False,
                attention_config={"flash_attn_version": 3},
                gpu_memory_utilization=0.8,
            )
        finally:
            # Workers inherit startup settings; unrelated engines retain the
            # caller's original settings even if plugin registration/loading fails.
            for name, previous_value in previous_environment.items():
                if previous_value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = previous_value

    def preprocess(self, input_data: ImageInput) -> Image.Image:
        """Convert a single page to an RGB image without resizing its content.

        Args:
            input_data: PIL image, image path, HWC/CHW array, or tensor with one,
                three, or four channels and an optional leading batch of size one.
                HWC wins when both first and last dimensions look like channels.
                Floating-point pixels in [0, 1] are scaled to [0, 255].

        Returns:
            RGB PIL image for the checkpoint's native NaViT processor.

        Raises:
            TypeError: If the input type is unsupported.
            ValueError: If the image shape or pixels are invalid.
            OSError: If an image path cannot be opened or decoded.
        """
        return normalize_parse_image(input_data)

    def invoke(self, input_data: ImageInput, task_prompt: str | None = None) -> str:
        """Generate structured Parse output for one page.

        Args:
            input_data: Page image in a format accepted by :meth:`preprocess`.
            task_prompt: Optional native control tokens overriding the configured prompt.

        Returns:
            Decoded text with coordinate and class tags preserved.

        Raises:
            TypeError: If the image type is unsupported.
            ValueError: If image data or processor inputs are invalid.
            OSError: If image or checkpoint assets cannot be loaded.
            RuntimeError: If the requested CUDA device is unavailable, or model
                loading or inference fails, including GPU exhaustion.
        """
        return self.invoke_batch([input_data], task_prompt=task_prompt)[0]

    def invoke_batch(self, inputs: Sequence[ImageInput], task_prompt: str | None = None) -> list[str]:
        """Convert and parse one page at a time, keeping the original order.

        Args:
            inputs: Sequence of pages accepted by :meth:`preprocess`; an empty
                sequence returns immediately without loading the checkpoint.
            task_prompt: Optional native control tokens overriding the configured prompt.

        Returns:
            One tagged output string per input page, in input order.

        Raises:
            TypeError: If an image type is unsupported.
            ValueError: If image data or processor inputs are invalid.
            OSError: If image or checkpoint assets cannot be loaded.
            RuntimeError: If the requested CUDA device is unavailable, or model
                loading or generation fails. No partial batch is returned.
        """
        if not inputs:
            return []
        outputs: list[str] = []
        for image in inputs:
            page = self.preprocess(image)
            self._ensure_model()
            if self._backend == "vllm":
                prompt = task_prompt if task_prompt is not None else self._task_prompt
                generated = self._model.generate(
                    [
                        {
                            "prompt": ("<pad>" if self._use_dflash else "<pad>\n") + prompt,
                            "multi_modal_data": {"image": page},
                        }
                    ],
                    self._generation_config,
                    tokenization_kwargs={"add_special_tokens": False},
                )
                outputs.append(generated[0].outputs[0].text.strip())
                continue
            features = self._processor(
                images=[page],
                text=task_prompt if task_prompt is not None else self._task_prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self._device)
            with torch.inference_mode():
                sequences = self._model.generate(**features, generation_config=self._generation_config)
            outputs.append(self._processor.batch_decode(sequences, skip_special_tokens=False)[0].strip())
        return outputs

    def __call__(self, input_data: ImageInput, task_prompt: str | None = None) -> str:
        """Generate tagged text for a single page, delegating to :meth:`invoke`.

        Args:
            input_data: Page image accepted by :meth:`preprocess`.
            task_prompt: Optional native control tokens overriding the configured prompt.

        Returns:
            Decoded text with coordinate and class tags preserved.

        Raises:
            TypeError: If the image type is unsupported.
            ValueError: If image data or processor inputs are invalid.
            OSError: If image or checkpoint assets cannot be loaded.
            RuntimeError: If the requested CUDA device is unavailable, or
                checkpoint loading or generation fails.
        """
        return self.invoke(input_data, task_prompt=task_prompt)

    def unload(self) -> None:
        """Shut down vLLM workers or release Transformers tensors explicitly.

        Subsequent inference reloads the pinned checkpoint. Repeated calls are safe.

        Returns:
            None.

        Raises:
            RuntimeError: If engine shutdown or PyTorch device cleanup fails.
                References are retained on shutdown failure so cleanup can be retried.
        """
        if self._model is not None:
            if self._backend == "vllm":
                self._model.llm_engine.engine_core.shutdown(timeout=30.0)
            else:
                self._model.to("cpu")
            self._model = None
        self._processor = None
        self._generation_config = None
        if self._device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def model_name(self) -> str:
        """Return the checkpoint's human-readable model name."""
        return "Nemotron-3.5-Parse"

    @property
    def model_type(self) -> str:
        """Return the document-parser model category."""
        return "document-parse"

    @property
    def model_runmode(self) -> ModelRunMode:
        """Return the local inference mode."""
        return "local"

    @property
    def input(self) -> dict[str, Any]:
        """Return the supported RGB image input schema."""
        return {
            "type": "image",
            "format": "RGB",
            "supported_inputs": ["PIL.Image", "path", "torch.Tensor", "np.ndarray"],
        }

    @property
    def output(self) -> dict[str, str]:
        """Return the tagged text output schema."""
        return {"type": "text", "format": "string"}

    @property
    def input_batch_size(self) -> int:
        """Return the number of pages generated concurrently (one)."""
        return 1
