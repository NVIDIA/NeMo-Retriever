# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in NVTX ranges that split profiled wall time by owner.

``NRL_NVTX=1`` emits stage ranges in the ``NRL``, ``vLLM``, and ``torch`` (model forward)
domains. ``NRL_NVTX=full`` also wraps high-frequency calls: ``torch`` host syncs that block on
the GPU and, in Ray workers, ``Ray`` task steps, batch formatting, and serialization.

NRL call sites use :func:`span`. :func:`install` wraps the third-party targets of the named
groups; GPU actors and vLLM model registration (which also runs in EngineCore) call it. When
the variable is unset, :func:`span` returns a shared no-op context and :func:`install` returns
immediately.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import inspect
import logging
import os
import sys
from typing import Any, Callable

logger = logging.getLogger(__name__)

_LEVEL = os.environ.get("NRL_NVTX", "")
ENABLED = _LEVEL in ("1", "full")
FULL = _LEVEL == "full"
COLORS = {"NRL": 0x76B900, "vLLM": 0x2F6FDF, "Ray": 0xF28C00, "torch": 0xEE4C2C}

_NULL = contextlib.nullcontext()
_SPANS: dict[tuple[str, str], "_Span"] = {}
_INSTALLED: set[str] = set()

# (module, attribute path, layer, range name). Generators are ranged per step; coroutines and async
# generators use start/end ranges because their awaits interleave on one thread.
# group -> (module, attribute path, layer, range name). Each call site installs only the groups
# its process uses, so actors do not import unrelated packages. Generators are ranged per step;
# coroutines and async generators use start/end ranges because their awaits interleave on one thread.
_TARGETS: dict[str, tuple[tuple[str, str, str, str], ...]] = {
    "page_elements": (
        (
            "nemo_retriever.models.local.nemotron_page_elements_v3",
            "NemotronPageElementsV3.preprocess",
            "NRL",
            "pe.preprocess",
        ),
        (
            "nemo_retriever.models.local.nemotron_page_elements_v3",
            "NemotronPageElementsV3.invoke",
            "torch",
            "forward:page_elements",
        ),
        (
            "nemo_retriever.models.local.nemotron_page_elements_v3",
            "NemotronPageElementsV3.postprocess",
            "NRL",
            "pe.postprocess",
        ),
    ),
    "ocr": (
        ("nemo_retriever.models.local.nemotron_ocr_v2", "NemotronOCRV2.invoke", "NRL", "ocr.invoke"),
        ("nemotron_ocr.inference.pipeline_v2", "NemotronOCRV2._preprocess_batch", "NRL", "pkg.ocr.preprocess"),
        ("nemotron_ocr.inference.pipeline_v2", "NemotronOCRV2._run_detector_batched", "torch", "forward:ocr.detector"),
        (
            "nemotron_ocr.inference.pipeline_v2",
            "NemotronOCRV2._run_recognizer_chunked",
            "torch",
            "forward:ocr.recognizer",
        ),
        (
            "nemotron_ocr.inference.pipeline_v2",
            "NemotronOCRV2._run_relational_and_build_output",
            "torch",
            "forward:ocr.relational",
        ),
        ("nemotron_ocr.inference.pipeline_v2", "NemotronOCRV2._postprocess_batch", "NRL", "pkg.ocr.postprocess"),
    ),
    "vllm": (
        ("vllm.v1.engine.async_llm", "AsyncLLM.add_request", "vLLM", "AsyncLLM.add_request"),
        ("vllm.v1.engine.async_llm", "AsyncLLM.encode", "vLLM", "AsyncLLM.encode"),
        ("vllm.renderers.base", "BaseRenderer.render_cmpl", "vLLM", "render_cmpl"),
        ("vllm.renderers.base", "BaseRenderer._tokenize_prompt", "vLLM", "tokenize"),
        ("vllm.renderers.base", "BaseRenderer._process_multimodal", "vLLM", "process_multimodal"),
        ("vllm.multimodal.processing.processor", "BaseMultiModalProcessor.apply", "vLLM", "mm_processor.apply"),
        ("vllm.multimodal.processing.processor", "BaseMultiModalProcessor._call_hf_processor", "vLLM", "hf_processor"),
        ("vllm.multimodal.hasher", "MultiModalHasher.hash_kwargs", "vLLM", "mm_hash"),
        ("vllm.v1.engine.input_processor", "InputProcessor.process_inputs", "vLLM", "process_inputs"),
        ("vllm.v1.engine.output_processor", "OutputProcessor.process_outputs", "vLLM", "process_outputs"),
        ("vllm.v1.engine.core", "EngineCore.preprocess_add_request", "vLLM", "core.preprocess_add_request"),
        ("vllm.v1.engine.core", "EngineCore.add_request", "vLLM", "core.add_request"),
        ("vllm.v1.engine.core", "EngineCore.step", "vLLM", "core.step"),
        ("vllm.v1.engine.core", "EngineCore.step_with_batch_queue", "vLLM", "core.step"),
        ("vllm.v1.engine.core", "EngineCoreProc._process_input_queue", "vLLM", "core.wait_input"),
        ("vllm.v1.core.sched.scheduler", "Scheduler.schedule", "vLLM", "core.schedule"),
        ("vllm.v1.core.sched.scheduler", "Scheduler.update_from_output", "vLLM", "core.update_from_output"),
        ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner.execute_model", "vLLM", "runner.execute_model"),
        ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner._update_states", "vLLM", "runner.update_states"),
        ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner._prepare_inputs", "vLLM", "runner.prepare_inputs"),
        ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner._execute_mm_encoder", "vLLM", "runner.mm_encoder"),
        (
            "vllm.v1.worker.gpu_model_runner",
            "GPUModelRunner._gather_mm_embeddings",
            "vLLM",
            "runner.gather_mm_embeddings",
        ),
        ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner._model_forward", "vLLM", "runner.model_forward"),
        ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner._pool", "vLLM", "runner.pooler"),
        (
            "vllm.v1.worker.gpu_model_runner",
            "AsyncGPUPoolingModelRunnerOutput.get_output",
            "vLLM",
            "runner.output_copy_sync",
        ),
    ),
}
# NRL_NVTX=full adds high-frequency host-sync and serialization ranges; they cost steady-state time.
_FULL_TARGETS: dict[str, tuple[tuple[str, str, str, str], ...]] = {
    "torch": (
        ("torch.cuda", "synchronize", "torch", "host_sync:cuda.synchronize"),
        ("torch.cuda", "Event.synchronize", "torch", "host_sync:event.synchronize"),
        ("torch.cuda", "Stream.synchronize", "torch", "host_sync:stream.synchronize"),
    ),
    "ray": (
        ("ray.data._internal.execution.operators.actor_pool_map_operator", "_MapWorker.submit", "Ray", "task_step"),
        ("ray.data._internal.block_batching.util", "_format_batch", "Ray", "format_batch"),
        ("ray.data.block", "BlockAccessor.batch_to_block", "Ray", "batch_to_block"),
        ("ray._private.serialization", "SerializationContext.deserialize_objects", "Ray", "deserialize"),
        ("ray._private.serialization", "SerializationContext.serialize", "Ray", "serialize"),
    ),
    "vllm": (
        ("vllm.v1.serial_utils", "MsgpackEncoder.encode", "vLLM", "msgpack_encode"),
        ("vllm.v1.serial_utils", "MsgpackEncoder.encode_into", "vLLM", "msgpack_encode"),
        ("vllm.v1.serial_utils", "MsgpackDecoder.decode", "vLLM", "msgpack_decode"),
    ),
}


class _Span:
    __slots__ = ("_domain", "_attrs")

    def __init__(self, layer: str, name: str) -> None:
        import nvtx

        self._domain = nvtx.get_domain(layer)
        self._attrs = self._domain.get_event_attributes(message=name, color=COLORS[layer])

    def __enter__(self) -> None:
        self._domain.push_range(self._attrs)

    def __exit__(self, *exc: Any) -> bool:
        self._domain.pop_range()
        return False

    def start(self) -> Any:
        return self._domain.start_range(self._attrs)

    def end(self, range_id: Any) -> None:
        self._domain.end_range(range_id)


def _span(layer: str, name: str) -> _Span:
    span_ = _SPANS.get((layer, name))
    if span_ is None:
        span_ = _SPANS[(layer, name)] = _Span(layer, name)
    return span_


def span(layer: str, name: str) -> Any:
    """Return a context manager that emits ``name`` in ``layer`` when ``NRL_NVTX`` is set, else a no-op."""
    if not ENABLED:
        return _NULL
    return _span(layer, name)


def _wrap(fn: Callable[..., Any], layer: str, name: str) -> Callable[..., Any]:
    rng = _span(layer, name)
    if inspect.isasyncgenfunction(fn):

        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            range_id = rng.start()
            try:
                async for item in fn(*args, **kwargs):
                    yield item
            finally:
                rng.end(range_id)

    elif inspect.iscoroutinefunction(fn):

        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            range_id = rng.start()
            try:
                return await fn(*args, **kwargs)
            finally:
                rng.end(range_id)

    elif inspect.isgeneratorfunction(fn):

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            it = fn(*args, **kwargs)
            try:
                while True:
                    with rng:
                        try:
                            item = next(it)
                        except StopIteration as stop:
                            return stop.value
                    yield item
            finally:
                it.close()

    else:

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with rng:
                return fn(*args, **kwargs)

    functools.update_wrapper(wrapper, fn)
    wrapper.__nrl_nvtx__ = True
    return wrapper


def _patch(module: str, path: str, layer: str, name: str) -> None:
    owner: Any = importlib.import_module(module)
    *parents, attr = path.split(".")
    for parent in parents:
        owner = getattr(owner, parent)
    raw = inspect.getattr_static(owner, attr)
    if isinstance(raw, (classmethod, staticmethod)):
        setattr(owner, attr, type(raw)(_wrap(raw.__func__, layer, name)))
    elif not getattr(raw, "__nrl_nvtx__", False):
        setattr(owner, attr, _wrap(raw, layer, name))


def _install_tensor_syncs() -> None:
    import torch

    def wrap(name: str, blocks: Callable[..., bool]) -> None:
        orig = getattr(torch.Tensor, name)
        if getattr(orig, "__nrl_nvtx__", False):
            return
        rng = _span("torch", f"host_sync:{name.strip('_')}")

        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if self.is_cuda and blocks(args, kwargs):
                with rng:
                    return orig(self, *args, **kwargs)
            return orig(self, *args, **kwargs)

        functools.update_wrapper(wrapper, orig)
        wrapper.__nrl_nvtx__ = True
        setattr(torch.Tensor, name, wrapper)

    for name in ("item", "tolist", "cpu", "numpy", "__bool__", "__int__", "__float__"):
        wrap(name, lambda args, kwargs: True)
    wrap("to", lambda args, kwargs: str(kwargs.get("device", args[0] if args else "")).startswith("cpu"))


def install(*groups: str) -> None:
    """Wrap the targets of ``groups`` once per process; a no-op unless ``NRL_NVTX`` is ``1`` or ``full``.

    ``NRL_NVTX=full`` also wraps CUDA host syncs and, in Ray workers, Ray serialization.
    """
    if not ENABLED:
        return
    if FULL:
        groups += ("torch",) + (("ray",) if "ray" in sys.modules else ())
    missing = []
    for group in dict.fromkeys(groups):
        if group in _INSTALLED:
            continue
        _INSTALLED.add(group)
        targets = _TARGETS.get(group, ()) + (_FULL_TARGETS.get(group, ()) if FULL else ())
        for module, path, layer, name in targets:
            try:
                _patch(module, path, layer, name)
            except (ImportError, AttributeError) as exc:
                missing.append(f"{module}:{path} ({type(exc).__name__})")
        if group == "torch":
            try:
                _install_tensor_syncs()
            except ImportError as exc:
                missing.append(f"torch.Tensor host syncs ({type(exc).__name__})")
    if missing:
        logger.warning("NRL_NVTX: %d targets not instrumented: %s", len(missing), "; ".join(missing))
