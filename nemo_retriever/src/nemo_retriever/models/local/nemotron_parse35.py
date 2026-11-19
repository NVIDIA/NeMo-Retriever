# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned Transformers and vLLM implementations of local Parse 3.5 inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Literal, Sequence

import numpy as np
import torch
from PIL import Image

from nemo_retriever.common.params.utils import (
    NEMOTRON_PARSE35_MODEL,
    NEMOTRON_PARSE35_DEFAULT_TASK_PROMPT,
    resolve_nemotron_parse_dflash_tokens,
)
from nemo_retriever.models.hf_cache import configure_global_hf_cache_base
from nemo_retriever.models.hf_model_registry import get_hf_revision
from nemo_retriever.models.inference.parse_image import ImageInput, normalize_parse_image
from nemo_retriever.models.model import BaseModel, ModelRunMode


# Pages generated concurrently per engine; Ray UDF threads keep this many in flight.
_MAX_NUM_SEQS = 128
# Rendered pages below this pixel standard deviation have no content to parse.
_BLANK_PIXEL_STD = 0.01


@dataclass(frozen=True)
class ParsePageResult:
    """Generated output and termination evidence for one page.

    Attributes:
        text: Decoded text with coordinate and class tags preserved.
        finish_reason: Generation stop reason reported by the backend, such as
            ``"stop"`` or ``"length"``. ``None`` when the backend does not report one.
        output_tokens: Number of generated tokens, or ``None`` when unavailable.
    """

    text: str
    finish_reason: str | None = None
    output_tokens: int | None = None


class NemotronParse35(BaseModel):
    """Parse page images with the checkpoint's native Transformers or vLLM backend.

    Weights and custom code are loaded lazily from an immutable registry revision.
    The vLLM backend submits each batch to one asynchronous engine, so concurrent
    callers share its scheduler. Transformers converts and generates pages
    sequentially. The vLLM backend registers the
    official plugin from the same pinned snapshot in each engine process.
    """

    def __init__(
        self,
        device: str = "cuda",
        hf_cache_dir: str | None = None,
        task_prompt: str = NEMOTRON_PARSE35_DEFAULT_TASK_PROMPT,
        max_tokens: int = 9000,
        backend: Literal["transformers", "vllm"] = "vllm",
        dflash_tokens: int | Literal["auto"] | None = "auto",
        sampling_params: dict[str, Any] | None = None,
    ) -> None:
        """Configure a lazy BF16 parser without loading weights.

        Args:
            device: PyTorch inference device, normally the worker's ``cuda`` device.
                Select ``cpu`` with ``backend="transformers"`` when CUDA is unavailable.
            hf_cache_dir: Optional Hugging Face snapshot cache directory.
            task_prompt: Native control tokens. Picture text is disabled unless enabled explicitly.
            max_tokens: Positive maximum number of generated tokens per page.
            dflash_tokens: "auto" selects DFlash2-32 for vLLM and leaves Transformers unchanged.
                An integer from 1 to 32 selects a draft length; None preserves the vendor
                one-token MTP default on vLLM. The drafter shares the target checkpoint revision.
            backend: In-process vLLM (the default) or explicit native Transformers.
                vLLM requires CUDA device zero
                in the worker's visible GPU set; Transformers also supports CPU.

            sampling_params: Optional vLLM overrides for ``temperature``, ``top_k``,
                ``top_p``, ``repetition_penalty``, and ``seed``. Defaults retain
                greedy decoding. Available only with backend="vllm".

        Raises:
            ValueError: If max_tokens is not positive, the backend is unsupported,
                the device is invalid, or the DFlash2 configuration is unsupported.
            RuntimeError: If PyTorch cannot interpret the device.
        """
        super().__init__()
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        if backend not in ("transformers", "vllm"):
            raise ValueError(f"Unsupported Parse 3.5 backend: {backend}")
        self._dflash_tokens = resolve_nemotron_parse_dflash_tokens(dflash_tokens, backend=backend)
        self._device = torch.device(device)
        if backend == "vllm" and (self._device.type != "cuda" or self._device.index not in (None, 0)):
            raise ValueError('Parse 3.5 vLLM requires device="cuda" or "cuda:0" in the worker GPU set.')
        allowed_sampling = {"temperature", "top_k", "top_p", "repetition_penalty", "seed"}
        if sampling_params and (backend != "vllm" or set(sampling_params) - allowed_sampling):
            raise ValueError(
                "sampling_params requires vllm and accepts temperature, top_k, top_p, repetition_penalty, seed."
            )
        self._sampling_params = dict(sampling_params or {})
        self._backend = backend
        self._hf_cache_dir = hf_cache_dir
        self._task_prompt = task_prompt
        self._max_tokens = max_tokens
        self._model: Any = None
        self._processor: Any = None
        self._generation_config: Any = None
        self._load_lock = threading.Lock()

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is None:
                self._load()

    def _load(self) -> None:
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
        from huggingface_hub import snapshot_download

        cache_dir = configure_global_hf_cache_base(self._hf_cache_dir)
        snapshot = snapshot_download(
            NEMOTRON_PARSE35_MODEL,
            revision=get_hf_revision(NEMOTRON_PARSE35_MODEL),
            cache_dir=cache_dir,
            allow_patterns=["*.json", "*.py", "*.safetensors", "*.jinja", "*.model", "*.txt"],
            ignore_patterns=["dflash2/*"] if self._dflash_tokens is None else None,
        )
        if self._backend == "vllm":
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
            from vllm import SamplingParams

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
            speculative_kwargs: dict[str, Any] = {}
            if self._dflash_tokens is not None:
                draft = Path(snapshot) / "dflash2"
                missing = [name for name in ("config.json", "model.safetensors") if not (draft / name).is_file()]
                if missing:
                    raise RuntimeError(
                        f"The pinned Parse 3.5 snapshot is missing its DFlash2 checkpoint at {draft} "
                        f"(missing: {', '.join(missing)}). Refresh the pinned {NEMOTRON_PARSE35_MODEL} snapshot "
                        f"at revision {get_hf_revision(NEMOTRON_PARSE35_MODEL)} in your configured Hugging Face "
                        "cache, or select dflash_tokens=None (nemotron_parse_dflash_tokens=None for ingestion) "
                        "to use the MTP fallback."
                    )
                speculative_kwargs = {
                    "hf_overrides": {"architectures": ["NRLNemotronParse35DFlashModel"]},
                    "speculative_config": {
                        "method": "dflash",
                        "model": str(draft),
                        "num_speculative_tokens": self._dflash_tokens,
                    },
                }
            self._model = _AsyncParseEngine(
                model=snapshot,
                trust_remote_code=True,
                dtype="bfloat16",
                distributed_executor_backend="uni",
                max_model_len=16384,
                max_num_seqs=_MAX_NUM_SEQS,
                max_num_batched_tokens=16384,
                limit_mm_per_prompt={"image": 1},
                enable_prefix_caching=False,
                attention_config={"flash_attn_version": 3},
                gpu_memory_utilization=0.3,
                kv_cache_memory_bytes=24 * 1024**3,
                # UDF threads render concurrently, so a cached repeat image could reach the engine before its data.
                mm_processor_cache_gb=0,
                compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"},
                **speculative_kwargs,
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
        """Parse pages in one vLLM submission or one at a time on Transformers, keeping input order.

        vLLM prepends ``<pad>`` directly to the task prompt in both DFlash2 and
        MTP modes. No newline is inserted, including for custom task prompts.

        Args:
            inputs: Sequence of pages accepted by :meth:`preprocess`; an empty
                sequence returns immediately without loading the checkpoint.
            task_prompt: Optional native control tokens overriding the configured prompt.

        Returns:
            One tagged output string per input page, in input order. Use
            :meth:`invoke_batch_results` to check whether generation reached
            ``max_tokens``.

        Raises:
            TypeError: If an image type is unsupported.
            ValueError: If image data or processor inputs are invalid.
            OSError: If image or checkpoint assets cannot be loaded.
            RuntimeError: If the requested CUDA device is unavailable, or model
                loading or generation fails. No partial batch is returned.
        """
        return [result.text for result in self.invoke_batch_results(inputs, task_prompt=task_prompt)]

    def invoke_batch_results(
        self, inputs: Sequence[ImageInput], task_prompt: str | None = None, max_tokens: int | None = None
    ) -> list[ParsePageResult]:
        """Parse pages like :meth:`invoke_batch`, also returning termination evidence.

        Args:
            inputs: Sequence of pages accepted by :meth:`preprocess`; an empty
                sequence returns immediately without loading the checkpoint.
            task_prompt: Optional native control tokens overriding the configured prompt.
            max_tokens: Optional positive output token limit for this call; vLLM only.

        Returns:
            One :class:`ParsePageResult` per input page, in input order. The vLLM
            backend reports ``finish_reason="length"`` when a page reached
            ``max_tokens``; its text is then incomplete. On vLLM, a ``uint8`` NumPy page whose
            pixel standard deviation is below 0.01 is blank: it returns empty text
            with ``finish_reason="stop"`` and zero tokens, without inference. The
            Transformers backend reports no finish reason or token count.

        Raises:
            TypeError: If an image type is unsupported.
            ValueError: If image data or processor inputs are invalid.
            OSError: If image or checkpoint assets cannot be loaded.
            RuntimeError: If the requested CUDA device is unavailable, or model
                loading or generation fails. No partial batch is returned.
        """
        if max_tokens is not None and (max_tokens <= 0 or self._backend != "vllm"):
            raise ValueError("max_tokens overrides must be positive and require the vllm backend.")
        if not inputs:
            return []
        if self._backend == "vllm":
            return self._invoke_vllm(inputs, task_prompt, max_tokens)
        outputs: list[ParsePageResult] = []
        for image in inputs:
            page = self.preprocess(image)
            self._ensure_model()
            features = self._processor(
                images=[page],
                text=task_prompt if task_prompt is not None else self._task_prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self._device)
            with torch.inference_mode():
                sequences = self._model.generate(**features, generation_config=self._generation_config)
            outputs.append(
                ParsePageResult(text=self._processor.batch_decode(sequences, skip_special_tokens=False)[0].strip())
            )
        return outputs

    def _invoke_vllm(
        self, inputs: Sequence[ImageInput], task_prompt: str | None, max_tokens: int | None = None
    ) -> list[ParsePageResult]:
        results = [ParsePageResult(text="", finish_reason="stop", output_tokens=0) for _ in inputs]
        converted = [self.preprocess(image) for image in inputs]
        keep = [
            index
            for index, image in enumerate(inputs)
            if not (isinstance(image, np.ndarray) and image.dtype == np.uint8 and float(image.std()) < _BLANK_PIXEL_STD)
        ]
        if not keep:
            return results
        pages = [converted[index] for index in keep]
        self._ensure_model()
        prompt = "<pad>" + (task_prompt if task_prompt is not None else self._task_prompt)
        params = self._generation_config
        if max_tokens is not None and max_tokens != params.max_tokens:
            params = params.clone()
            params.max_tokens = max_tokens
        generated = self._model.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": page}} for page in pages],
            params,
            tokenization_kwargs={"add_special_tokens": False},
        )
        for index, request in zip(keep, generated, strict=True):
            completion = request.outputs[0]
            results[index] = ParsePageResult(
                text=completion.text.strip(),
                finish_reason=completion.finish_reason,
                output_tokens=len(completion.token_ids),
            )
        return results

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
                self._model.shutdown(timeout=30.0)
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
        """Return the maximum number of pages the vLLM engine generates concurrently."""
        return _MAX_NUM_SEQS if self._backend == "vllm" else 1


class _AsyncParseEngine:
    """Run ``LLM.generate``-style calls on a vLLM ``AsyncLLM`` owned by a private event loop.

    Each page is its own request, so the scheduler batches pages from every
    calling thread together instead of serving one ``generate`` call at a time.
    """

    def __init__(self, **engine_kwargs: Any) -> None:
        import asyncio
        import itertools

        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.usage.usage_lib import UsageContext
        from vllm.v1.engine.async_llm import AsyncLLM

        engine_kwargs.setdefault("disable_log_stats", True)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="nrl-parse-engine")
        self._thread.start()

        async def build() -> Any:
            return AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_kwargs), usage_context=UsageContext.LLM_CLASS)

        try:
            self.engine = asyncio.run_coroutine_threadsafe(build(), self._loop).result()
        except BaseException:
            self._stop_loop()
            raise
        self._request_ids = itertools.count()
        self._render_lock = threading.Lock()

    def generate(self, prompts: list[Any], params: Any, *, tokenization_kwargs: dict | None = None) -> list[Any]:
        """Return each prompt's final ``RequestOutput`` in prompt order; any request failure raises."""
        import asyncio
        import os

        from vllm.renderers.inputs.preprocess import parse_model_prompt
        from vllm.sampling_params import RequestOutputKind

        engine = self.engine
        tok_params = engine.renderer.default_cmpl_tok_params.with_kwargs(**(tokenization_kwargs or {}))
        # Image processing runs on the calling thread, so it overlaps generation for other threads.
        with self._render_lock:
            engine_inputs = engine.renderer.render_cmpl(
                [parse_model_prompt(engine.model_config, prompt) for prompt in prompts], tok_params
            )

        async def one(engine_input: Any, request_id: str) -> Any:
            request_params = params.clone()
            request_params.output_kind = RequestOutputKind.FINAL_ONLY
            output = None
            async for output in engine.generate(engine_input, request_params, request_id):
                pass
            if output is None:
                raise RuntimeError(
                    f"Parse engine returned no output for request {request_id}. "
                    "Check the vLLM engine logs for an error, then retry the batch."
                )
            reason = output.outputs[0].finish_reason
            if reason in ("abort", "error"):
                raise RuntimeError(
                    f"Parse engine request {request_id} finished with reason {reason!r}. "
                    "Check the vLLM engine logs for an error, then retry the batch."
                )
            return output

        async def gather() -> list[Any]:
            tasks = [
                asyncio.ensure_future(one(item, f"nrl-parse-{os.getpid()}-{next(self._request_ids)}"))
                for item in engine_inputs
            ]
            try:
                return await asyncio.gather(*tasks)
            except BaseException:
                # Cancelling a request's stream aborts it in the engine, so a failed batch frees its slots.
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

        return asyncio.run_coroutine_threadsafe(gather(), self._loop).result()

    def shutdown(self, timeout: float | None = None) -> None:
        """Stop the engine core, then the private event loop."""
        self.engine.shutdown(timeout=timeout)
        self._stop_loop(timeout=timeout)

    def _stop_loop(self, timeout: float | None = 30.0) -> None:
        import asyncio

        async def cancel_pending() -> None:
            # vLLM's shutdown only schedules its output handler's cancellation; let it finish before closing.
            tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
            for task in tasks:
                if not task.cancelling():
                    task.cancel()
            if tasks:
                await asyncio.wait(tasks, timeout=timeout)

        asyncio.run_coroutine_threadsafe(cancel_pending(), self._loop).result()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()
