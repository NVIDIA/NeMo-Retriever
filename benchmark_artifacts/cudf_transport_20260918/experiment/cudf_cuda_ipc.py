# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental Ray Direct Transport backend for ``cudf.DataFrame``.

This is intentionally benchmark-scoped.  Ray 2.56's RDT API is alpha and
Ray Data does not route ``map_batches`` blocks through RDT.  The transport
below targets Ray Core actors on the same node and GPU, using CUDA IPC for
cuDF's device-serialized buffers while carrying only schema and any host
frames in Ray's small metadata message.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import ray
from ray.experimental.rdt.tensor_transport_manager import (
    CommunicatorMetadata,
    TensorTransportManager,
    TensorTransportMetadata,
)


logger = logging.getLogger(__name__)


@dataclass
class CudfCudaIpcCommunicatorMetadata(CommunicatorMetadata):
    """CUDA IPC is one-sided, so no communicator setup is required."""


@dataclass
class CudfCudaIpcMetadata(TensorTransportMetadata):
    dataframe_headers: list[dict[str, Any]] = field(default_factory=list)
    # Per dataframe, entries are (is_cuda, cuda_frame_index, host_frame_index).
    frame_layouts: list[list[tuple[bool, int, int]]] = field(default_factory=list)
    host_frames: list[bytes] = field(default_factory=list)
    # Entries are (IPC handle, allocation bytes, frame offset, frame bytes).
    cuda_ipc_handles: list[tuple[bytes, int, int, int]] = field(default_factory=list)
    event_ipc_handle: Optional[bytes] = None
    ray_gpu_idx: Optional[int] = None
    ray_node_id: Optional[str] = None


class CudfCudaIpcTransport(TensorTransportManager):
    """Same-node, same-GPU CUDA IPC transport for cuDF DataFrames."""

    def __init__(self) -> None:
        # Keep zero-copy torch views alive until Ray releases the RDT ref.
        self._frame_views: dict[str, list[Any]] = {}

    def tensor_transport_backend(self) -> str:
        return "CUDF_CUDA_IPC"

    @staticmethod
    def is_one_sided() -> bool:
        return True

    @staticmethod
    def can_abort_transport() -> bool:
        return False

    def actor_has_tensor_transport(self, actor: ray.actor.ActorHandle) -> bool:
        return True

    def extract_tensor_transport_metadata(self, obj_id: str, rdt_object: list[Any]) -> CudfCudaIpcMetadata:
        import cupy as cp
        import torch
        from cuda.bindings import driver

        dataframe_headers: list[dict[str, Any]] = []
        frame_layouts: list[list[tuple[bool, int, int]]] = []
        host_frames: list[bytes] = []
        cuda_handles: list[tuple[bytes, int, int, int]] = []
        frame_views: list[Any] = []

        event_handle: Optional[bytes] = None
        ray_gpu_idx: Optional[int] = None
        ray_node_id: Optional[str] = None

        if rdt_object:
            ray_gpu_ids = ray.get_gpu_ids()
            if not ray_gpu_ids:
                raise RuntimeError("CUDF_CUDA_IPC source actor has no Ray-assigned GPU")
            if not torch.cuda.is_available():
                raise RuntimeError("CUDF_CUDA_IPC requires a usable PyTorch CUDA runtime in the source actor")
            ray_gpu_idx = int(ray_gpu_ids[0])
            ray_node_id = ray.get_runtime_context().get_node_id()

            event = torch.cuda.Event(interprocess=True)
            torch.cuda.current_stream().record_event(event)
            event_handle = event.ipc_handle()

        for dataframe in rdt_object:
            header, frames = dataframe.device_serialize()
            dataframe_headers.append(header)
            layout: list[tuple[bool, int, int]] = []
            for frame in frames:
                if hasattr(frame, "__cuda_array_interface__"):
                    # View every cuDF buffer as a flat byte tensor without a copy.
                    cupy_view = cp.asarray(frame).view(cp.uint8).reshape(-1)
                    cuda_index = len(cuda_handles)
                    if cupy_view.nbytes == 0:
                        # List columns include a logical data frame for the
                        # parent column even though its bytes live in children.
                        cuda_handles.append((b"", 0, 0, 0))
                        frame_views.append(cupy_view)
                        layout.append((True, cuda_index, -1))
                        continue
                    # CUDA IPC handles describe the owning allocation, not an
                    # arbitrary view into it. cuDF frequently gives us an
                    # interior pointer, so record the allocation plus offset.
                    (status,) = driver.cuInit(0)
                    if status != driver.CUresult.CUDA_SUCCESS:
                        raise RuntimeError(f"cuInit failed: {status}")
                    status, base_ptr, allocation_bytes = driver.cuMemGetAddressRange(int(cupy_view.data.ptr))
                    if status != driver.CUresult.CUDA_SUCCESS:
                        raise RuntimeError(f"cuMemGetAddressRange failed: {status}")
                    base = int(base_ptr)
                    offset = int(cupy_view.data.ptr) - base
                    cuda_handles.append(
                        (
                            cp.cuda.runtime.ipcGetMemHandle(base),
                            int(allocation_bytes),
                            offset,
                            int(cupy_view.nbytes),
                        )
                    )
                    frame_views.append(cupy_view)
                    layout.append((True, cuda_index, -1))
                else:
                    host_index = len(host_frames)
                    host_frames.append(bytes(memoryview(frame)))
                    layout.append((False, -1, host_index))
            frame_layouts.append(layout)

        self._frame_views[obj_id] = frame_views
        # Ray requires one logical metadata entry per reconstructed RDT object.
        logical_meta = [(tuple(int(v) for v in dataframe.shape), np.dtype("O")) for dataframe in rdt_object]
        return CudfCudaIpcMetadata(
            tensor_meta=logical_meta,
            tensor_device="cuda" if rdt_object else None,
            dataframe_headers=dataframe_headers,
            frame_layouts=frame_layouts,
            host_frames=host_frames,
            cuda_ipc_handles=cuda_handles,
            event_ipc_handle=event_handle,
            ray_gpu_idx=ray_gpu_idx,
            ray_node_id=ray_node_id,
        )

    def get_communicator_metadata(
        self,
        src_actor: ray.actor.ActorHandle,
        dst_actor: ray.actor.ActorHandle,
        backend: Optional[str] = None,
    ) -> CudfCudaIpcCommunicatorMetadata:
        return CudfCudaIpcCommunicatorMetadata()

    def recv_multiple_tensors(
        self,
        obj_id: str,
        tensor_transport_metadata: TensorTransportMetadata,
        communicator_metadata: CommunicatorMetadata,
        target_buffers: Optional[list[Any]] = None,
    ) -> list[Any]:
        import cudf
        import cupy as cp
        import torch

        class _IpcAllocationOwner:
            def __init__(self, ptr: int) -> None:
                self.ptr = ptr

            def __del__(self) -> None:
                try:
                    cp.cuda.runtime.ipcCloseMemHandle(self.ptr)
                except cp.cuda.runtime.CUDARuntimeError:
                    logger.warning("Failed to close CUDA IPC memory handle", exc_info=True)

        if target_buffers:
            raise ValueError("CUDF_CUDA_IPC does not support target buffers")
        if not isinstance(tensor_transport_metadata, CudfCudaIpcMetadata):
            raise TypeError("Expected CudfCudaIpcMetadata")

        metadata = tensor_transport_metadata
        if ray.get_runtime_context().get_node_id() != metadata.ray_node_id:
            raise ValueError("CUDF_CUDA_IPC only supports actors on the same node")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDF_CUDA_IPC requires a usable PyTorch CUDA runtime in the destination actor")
        try:
            local_device = ray.get_gpu_ids().index(metadata.ray_gpu_idx)
        except ValueError as exc:
            raise ValueError("CUDF_CUDA_IPC requires source and destination actors on the same GPU") from exc

        device = torch.device(f"cuda:{local_device}")
        if metadata.event_ipc_handle is not None:
            remote_event = torch.cuda.Event.from_ipc_handle(device=device, handle=metadata.event_ipc_handle)
            torch.cuda.current_stream(device).wait_event(remote_event)

        cp.cuda.Device(int(device.index)).use()
        cuda_frames: list[Any] = []
        opened_allocations: dict[bytes, tuple[int, _IpcAllocationOwner, Any]] = {}
        for handle, allocation_bytes, offset, frame_bytes in metadata.cuda_ipc_handles:
            if frame_bytes == 0:
                cuda_frames.append(cp.empty((0,), dtype=cp.uint8))
                continue
            cached = opened_allocations.get(handle)
            if cached is None:
                opened_ptr = cp.cuda.runtime.ipcOpenMemHandle(handle, cp.cuda.runtime.cudaIpcMemLazyEnablePeerAccess)
                owner = _IpcAllocationOwner(opened_ptr)
                memory = cp.cuda.UnownedMemory(opened_ptr, allocation_bytes, owner, device_id=int(device.index))
                cached = (opened_ptr, owner, memory)
                opened_allocations[handle] = cached
            _, _, memory = cached
            memptr = cp.cuda.MemoryPointer(memory, offset)
            cuda_frames.append(cp.ndarray((frame_bytes,), dtype=cp.uint8, memptr=memptr))

        dataframes: list[Any] = []
        for header, layout in zip(metadata.dataframe_headers, metadata.frame_layouts, strict=True):
            frames: list[Any] = []
            for is_cuda, cuda_index, host_index in layout:
                if is_cuda:
                    frames.append(cuda_frames[cuda_index])
                else:
                    frames.append(memoryview(metadata.host_frames[host_index]))
            dataframes.append(cudf.DataFrame.device_deserialize(header, frames))
        return dataframes

    def send_multiple_tensors(
        self,
        tensors: list[Any],
        tensor_transport_metadata: TensorTransportMetadata,
        communicator_metadata: CommunicatorMetadata,
    ) -> None:
        raise NotImplementedError("CUDF_CUDA_IPC is a one-sided transport")

    def garbage_collect(
        self,
        obj_id: str,
        tensor_transport_meta: TensorTransportMetadata,
        tensors: list[Any],
    ) -> None:
        self._frame_views.pop(obj_id, None)

    def abort_transport(self, obj_id: str, communicator_metadata: CommunicatorMetadata) -> None:
        raise NotImplementedError("CUDF_CUDA_IPC cannot abort an in-flight CUDA IPC read")


def register_cudf_cuda_ipc() -> None:
    """Register the custom transport before creating participating actors."""
    import cudf
    from ray.experimental.rdt.util import (
        register_tensor_transport,
        transport_manager_info,
    )

    name = "CUDF_CUDA_IPC"
    if name not in transport_manager_info:
        register_tensor_transport(
            name,
            ["cuda"],
            CudfCudaIpcTransport,
            cudf.DataFrame,
        )
