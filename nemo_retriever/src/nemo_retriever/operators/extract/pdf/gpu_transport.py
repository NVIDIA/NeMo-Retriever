# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Transient GPU images carried between local Ray model actors through RDT."""

from __future__ import annotations

import asyncio
import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass

import nvtx

_RESOLVED = contextvars.ContextVar("nrl_gpu_images", default=None)
_RECEIVE_CAPACITY = contextvars.ContextVar("nrl_gpu_receive_capacity", default=0)
_RECEIVE_POOLS = {}
_RECEIVE_LOCK = threading.Lock()


def _rgb_pool_capacity(name="CUDA_PDF_RGB_POOL_BYTES"):
    import os

    try:
        value = int(os.environ.get(name, "0"))
    except ValueError:
        raise ValueError(f"{name} must be an integer byte count") from None
    if value < 0 or 0 < value < 3 * 1024**2:
        raise ValueError(f"{name} must be zero or at least 3145728")
    return value


def _receive_pool(capacity):
    import torch
    from cuda_pdf_renderer.buffer_pool import DeviceRgbPool
    from ray.experimental import register_nixl_memory

    key = (torch.cuda.current_device(), capacity)
    with _RECEIVE_LOCK:
        if key not in _RECEIVE_POOLS:
            pool = DeviceRgbPool(capacity)
            register_nixl_memory(pool.storage)
            _RECEIVE_POOLS[key] = pool
        return _RECEIVE_POOLS[key]


class GpuTransportError(RuntimeError):
    """A requested GPU image path failed; host raster fallback is forbidden."""


@dataclass(frozen=True)
class GpuPageImage:
    reference: object
    index: int
    height: int
    width: int
    batch_shapes: tuple[tuple[int, int], ...] | None = None

    def tensor(self):
        import ray
        import torch

        cache = _RESOLVED.get()
        if cache is None:
            raise GpuTransportError("GPU images must be consumed inside a scoped model call")
        if self.reference not in cache:
            with nvtx.annotate("nrl.cuda_rdt::fetch_rgb", color="purple"):
                buffers = None
                capacity = _RECEIVE_CAPACITY.get()
                if capacity:
                    if self.batch_shapes is None:
                        raise GpuTransportError("GPU receive pooling requires complete batch shape metadata")
                    with nvtx.annotate("nrl.cuda_rdt::receive_targets", color="purple"):
                        from ray.experimental import set_target_for_ref

                        pool = _receive_pool(capacity)
                        buffers = []
                        for height, width in self.batch_shapes:
                            target = pool.image(height, width)
                            if target is None:
                                target = torch.empty((height, width, 3), dtype=torch.uint8, device="cuda:0")
                            buffers.append(target)
                        cache[self.reference] = buffers
                        set_target_for_ref(self.reference, buffers)
                # Keep target leases alive until Ray returns the received tensors.
                received = ray.get(self.reference)
                if buffers is not None and (
                    len(received) != len(buffers)
                    or any(actual is not target for actual, target in zip(received, buffers))
                ):
                    raise GpuTransportError("Ray did not return the supplied GPU receive buffers")
                cache[self.reference] = received
        rgb = cache[self.reference][self.index]
        if (
            not isinstance(rgb, torch.Tensor)
            or not rgb.is_cuda
            or rgb.dtype != torch.uint8
            or tuple(rgb.shape) != (self.height, self.width, 3)
        ):
            raise GpuTransportError("Ray Direct Transport did not deliver the expected GPU RGB tensor")
        return rgb


@contextmanager
def gpu_image_scope(receive_pool_bytes=0):
    if _RESOLVED.get() is not None:
        if receive_pool_bytes != _RECEIVE_CAPACITY.get():
            raise GpuTransportError("Nested GPU image scopes must use the same receive-pool capacity")
        yield
        return
    cache = {}
    token = _RESOLVED.set(cache)
    capacity_token = _RECEIVE_CAPACITY.set(receive_pool_bytes)
    try:
        yield
    finally:
        try:
            if receive_pool_bytes and cache:
                import torch

                # These actors process one scoped model batch at a time. Finish
                # its GPU uses before releasing original receive-buffer leases.
                with nvtx.annotate("nrl.cuda_rdt::receive_scope_ready", color="purple"):
                    torch.cuda.synchronize()
        finally:
            _RECEIVE_CAPACITY.reset(capacity_token)
            _RESOLVED.reset(token)


class _GpuProducer:
    """A renderer uses the GPU reservation of its owning model-actor pool."""

    def __init__(self, library, gpu_uuid):
        import os

        # Ray assigned the reservation to the model actor. This child owns no
        # independent reservation and receives that actor's exact GPU UUID.
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
        os.environ["CUDA_PDF_DEVICE"] = "0"
        from cuda_pdf_renderer.device import DeviceRenderer

        self.renderer = DeviceRenderer(library)
        pool_bytes = _rgb_pool_capacity()
        self._rgb_pool = None
        if pool_bytes:
            from cuda_pdf_renderer.buffer_pool import DeviceRgbPool
            from ray.experimental import register_nixl_memory

            self._rgb_pool = DeviceRgbPool(pool_bytes)
            # All leased views share this storage. A permanent registration
            # prevents per-page registration churn and remote metadata changes.
            register_nixl_memory(self._rgb_pool.storage)
        self.pages = 0
        self.gpu_uuid = gpu_uuid
        self.max_scene_bytes = 0
        self._stats_lock = threading.Lock()
        self._thread_state = threading.local()
        self._pages = ThreadPoolExecutor(max_workers=16, thread_name_prefix="cuda-page")

    @nvtx.annotate("nrl.cuda_rdt::render_device", color="green")
    def render(self, scenes):
        results = []
        for scene in scenes:
            rgb, stats = self.renderer.replay(scene)
            if not rgb.is_cuda or stats["cpu_raster_calls"] or stats["readback_ms"]:
                raise GpuTransportError("Renderer violated GPU-only pixel output contract")
            results.append(rgb)
        self.pages += len(results)
        import ray

        with nvtx.annotate("nrl.cuda_rdt::publish_rgb", color="purple"):
            return {"reference": ray.put(results, _tensor_transport="nixl")}

    def _render_shared_page(self, handle):
        import torch

        if not hasattr(self._thread_state, "stream"):
            torch.cuda.set_device(0)
            self._thread_state.stream = torch.cuda.Stream(device=0)
        stream = self._thread_state.stream
        with torch.cuda.stream(stream):
            with nvtx.annotate("nrl.cuda_rdt::native_page_replay", color="green"):
                if self._rgb_pool is None:
                    rgb, stats = self.renderer.replay_handle(handle)
                else:
                    out = self._rgb_pool.image(handle.height, handle.width)
                    rgb, stats = self.renderer.replay_handle(handle, out=out)
            # Publish only completed GPU data. This waits for this page's
            # stream, without copying pixels or synchronizing other pages.
            with nvtx.annotate("nrl.cuda_rdt::page_ready_wait", color="purple"):
                stream.synchronize()
        if not rgb.is_cuda or stats["cpu_raster_calls"] or stats["readback_ms"]:
            raise GpuTransportError("Renderer violated GPU-only pixel output contract")
        return rgb

    @nvtx.annotate("nrl.cuda_rdt::render_shared_scene", color="green")
    def render_handles(self, handles):
        import ray

        futures = [self._pages.submit(self._render_shared_page, handle) for handle in handles]
        try:
            results = [future.result() for future in futures]
        except BaseException:
            # The caller owns the descriptors: drain every reader before
            # returning an error that allows the caller to close them.
            for future in futures:
                try:
                    future.result()
                except BaseException:
                    pass
            raise
        with self._stats_lock:
            self.max_scene_bytes = max([self.max_scene_bytes] + [handle.size for handle in handles])
            self.pages += len(results)
        with nvtx.annotate("nrl.cuda_rdt::publish_rgb", color="purple"):
            return {"reference": ray.put(results, _tensor_transport="nixl")}

    def status(self):
        import torch

        return {
            "pages": self.pages,
            "gpu_uuid": self.gpu_uuid,
            "rgb_readback_bytes": 0,
            "cpu_raster_calls": 0,
            "tensor_allocated_bytes": torch.cuda.memory_allocated(),
            "page_streams": 16,
            "batch_rpc_concurrency": 16,
            "max_scene_bytes": self.max_scene_bytes,
            "tensor_peak_bytes": torch.cuda.max_memory_allocated(),
            "tensor_reserved_bytes": torch.cuda.memory_reserved(),
            "rgb_pool": self._rgb_pool.status() if self._rgb_pool else None,
        }


class RendererPool:
    """Keep GPU-buffer owners alive until the complete ingestion graph drains."""

    def __init__(self, library, rgb_pool_bytes=None):
        self.library = library
        self.rgb_pool_bytes = _rgb_pool_capacity() if rgb_pool_bytes is None else rgb_pool_bytes
        self.producers = {}
        self.next_gpu = 0

    def get(self, node_id, gpu_uuid):
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        key = (node_id, gpu_uuid)
        if key not in self.producers:
            actor = ray.remote(num_cpus=0, num_gpus=0, enable_tensor_transport=True, max_concurrency=16)(_GpuProducer)
            self.producers[key] = actor.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False),
                runtime_env={
                    "env_vars": {
                        "CUDA_VISIBLE_DEVICES": gpu_uuid,
                        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                        "CUDA_PDF_RGB_POOL_BYTES": str(self.rgb_pool_bytes),
                    }
                },
            ).remote(self.library, gpu_uuid)
        return self.producers[key]

    async def get_next(self, node_id):
        # Model constructors register their Ray-assigned GPUs. Awaiting lets
        # registration proceed while CPU frontends retain their native scenes.
        while True:
            available = [actor for (node, _), actor in self.producers.items() if node == node_id]
            if available:
                result = available[self.next_gpu % len(available)]
                self.next_gpu += 1
                return result
            await asyncio.sleep(0.05)

    async def close(self):
        import ray

        try:
            return await asyncio.gather(*[actor.status.remote() for actor in self.producers.values()])
        finally:
            for actor in self.producers.values():
                ray.kill(actor)
            self.producers.clear()


@nvtx.annotate("nrl.cuda_rdt::request_pages", color="green")
def render_page_batch(frame, pool):
    import ray
    import torch
    from cuda_pdf_renderer.device import RecordedScene

    images = [item for item in frame["page_image"]]
    if all(isinstance(item.get("gpu_image"), GpuPageImage) for item in images):
        return frame
    scenes = [RecordedScene(item["cuda_scene"], item["orig_shape_hw"][1], item["orig_shape_hw"][0]) for item in images]
    gpu_uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
    gpu_uuid = gpu_uuid if gpu_uuid.startswith("GPU-") else "GPU-" + gpu_uuid
    producer = ray.get(pool.get.remote(ray.get_runtime_context().get_node_id(), gpu_uuid))
    reference = ray.get(producer.render.remote(scenes))["reference"]
    # Keep the reference itself in the CPU metadata block. Only an enabled GPU
    # model actor resolves it; pixel payloads never enter an Arrow/pandas block.
    result = frame.copy()
    shapes = tuple((scene.height, scene.width) for scene in scenes)
    result["page_image"] = [
        {
            "gpu_image": GpuPageImage(reference, i, scene.height, scene.width, shapes),
            "image_b64": None,
            "encoding": "gpu_rgb",
            "orig_shape_hw": (scene.height, scene.width),
        }
        for i, scene in enumerate(scenes)
    ]
    return result


def image_region_metadata(page):
    """Retain automatic image-region geometry without encoded host pixels."""
    import math

    from nemo_retriever.common.api.util.pdf.pdfium import (
        extract_forms_from_pdfium_page,
        extract_merged_images_from_pdfium_page,
        extract_merged_shapes_from_pdfium_page,
    )

    width, height = page.get_width(), page.get_height()
    crop_width, crop_height = (height, width) if page.get_rotation() % 180 else (width, height)
    boxes = (
        extract_merged_images_from_pdfium_page(page)
        + extract_merged_shapes_from_pdfium_page(page)
        + extract_forms_from_pdfium_page(page)
    )
    records = []
    for box in boxes:
        left, top, right, bottom = box
        if (
            min(math.ceil(right), math.ceil(crop_width)) - max(math.floor(left), 0) < 10
            or min(math.ceil(bottom), math.ceil(crop_height)) - max(math.floor(top), 0) < 10
        ):
            continue
        records.append(
            {
                "bbox_xyxy_norm": [left / width, top / height, right / width, bottom / height],
                "text": "",
                "image_b64": None,
                "pixel_transport": "gpu_deferred",
            }
        )
    return records


def raise_stage_errors(frame, column):
    """Do not index incomplete GPU model results after a failed transport/crop."""
    for payload in frame[column]:
        if isinstance(payload, dict) and payload.get("error"):
            raise GpuTransportError(f"GPU-resident {column} stage failed: {payload['error']}")


@nvtx.annotate("nrl.cuda_rdt::submit_shared_scenes", color="green")
def submit_scene_batch(scenes, pool):
    """Keep native command memfds alive until GPU replay has consumed them."""
    import ray

    producer = ray.get(pool.get_next.remote(ray.get_runtime_context().get_node_id()))
    handles = [scene.handle() for scene in scenes]
    reference = ray.get(producer.render_handles.remote(handles))["reference"]
    shapes = tuple((scene.height, scene.width) for scene in scenes)
    return [GpuPageImage(reference, index, scene.height, scene.width, shapes) for index, scene in enumerate(scenes)]
