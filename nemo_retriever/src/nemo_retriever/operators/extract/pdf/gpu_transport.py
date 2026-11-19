# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Transient GPU images carried between local Ray model actors through RDT."""

from __future__ import annotations

import asyncio
import contextvars
from contextlib import contextmanager
from dataclasses import dataclass

import nvtx

_RESOLVED = contextvars.ContextVar("nrl_gpu_images", default=None)


class GpuTransportError(RuntimeError):
    """A requested GPU image path failed; host raster fallback is forbidden."""


@dataclass(frozen=True)
class GpuPageImage:
    reference: object
    index: int
    height: int
    width: int

    def tensor(self):
        import ray
        import torch

        cache = _RESOLVED.get()
        if cache is None:
            raise GpuTransportError("GPU images must be consumed inside a scoped model call")
        if self.reference not in cache:
            with nvtx.annotate("nrl.cuda_rdt::fetch_rgb", color="purple"):
                cache[self.reference] = ray.get(self.reference)
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
def gpu_image_scope():
    token = _RESOLVED.set({})
    try:
        yield
    finally:
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
        self.pages = 0
        self.gpu_uuid = gpu_uuid
        self.max_scene_bytes = 0

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

        return {"reference": ray.put(results, _tensor_transport="nixl")}

    @nvtx.annotate("nrl.cuda_rdt::render_shared_scene", color="green")
    def render_handles(self, handles):
        import ray

        results = []
        for handle in handles:
            rgb, stats = self.renderer.replay_handle(handle)
            if not rgb.is_cuda or stats["cpu_raster_calls"] or stats["readback_ms"]:
                raise GpuTransportError("Renderer violated GPU-only pixel output contract")
            self.max_scene_bytes = max(self.max_scene_bytes, handle.size)
            results.append(rgb)
        self.pages += len(results)
        return {"reference": ray.put(results, _tensor_transport="nixl")}

    def status(self):
        import torch

        return {
            "pages": self.pages,
            "gpu_uuid": self.gpu_uuid,
            "rgb_readback_bytes": 0,
            "cpu_raster_calls": 0,
            "tensor_allocated_bytes": torch.cuda.memory_allocated(),
            "max_scene_bytes": self.max_scene_bytes,
            "tensor_peak_bytes": torch.cuda.max_memory_allocated(),
            "tensor_reserved_bytes": torch.cuda.memory_reserved(),
        }


class RendererPool:
    """Keep GPU-buffer owners alive until the complete ingestion graph drains."""

    def __init__(self, library):
        self.library = library
        self.producers = {}
        self.next_gpu = 0

    def get(self, node_id, gpu_uuid):
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        key = (node_id, gpu_uuid)
        if key not in self.producers:
            actor = ray.remote(num_cpus=0, num_gpus=0, enable_tensor_transport=True)(_GpuProducer)
            self.producers[key] = actor.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False),
                runtime_env={
                    "env_vars": {"CUDA_VISIBLE_DEVICES": gpu_uuid, "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}
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
    result["page_image"] = [
        {
            "gpu_image": GpuPageImage(reference, i, scene.height, scene.width),
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
    return [GpuPageImage(reference, index, scene.height, scene.width) for index, scene in enumerate(scenes)]
