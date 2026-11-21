# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Publish completed CUDA RGB through a separately registered NIXL transport.

The experimental implementation is pinned to the audited Ray transport version.
It preserves its descriptor ownership and transfers, and validates a completion
proof for each exact tensor instead of synchronizing unrelated device streams.
"""
import contextvars
import threading
import ray
from ray.experimental.rdt.nixl_tensor_transport import NixlTensorTransport, NixlTransportMetadata

NAME = "NRL_READY_NIXL"
_READY = contextvars.ContextVar("nrl_ready_nixl_outputs", default=None)
_LOCK = threading.Lock()


def _check_version():
    if ray.__version__ != "2.56.1":
        raise RuntimeError("CUDA_PDF_READY_TRANSPORT requires Ray 2.56.1")


def _validate(rgb, event):
    import torch

    if (
        not isinstance(rgb, torch.Tensor)
        or not rgb.is_cuda
        or rgb.dtype != torch.uint8
        or rgb.ndim != 3
        or rgb.numel() == 0
        or rgb.shape[-1] != 3
        or not rgb.is_contiguous()
    ):
        raise ValueError("Ready NIXL requires contiguous CUDA uint8 HWC RGB")
    if not isinstance(event, torch.cuda.Event) or event.device != rgb.device or not event.query():
        raise ValueError("RGB completion event is absent, on another device, or pending")


class ReadyNixlTransport(NixlTensorTransport):
    def __init__(self):
        _check_version()
        super().__init__()

    def tensor_transport_backend(self):
        return NAME

    def actor_has_tensor_transport(self, actor):
        def check(self):
            from ray.experimental.rdt.util import get_tensor_transport_manager

            get_tensor_transport_manager(NAME).get_nixl_agent()
            return True

        return ray.get(actor.__ray_call__.options(concurrency_group="_ray_system").remote(check))

    def extract_tensor_transport_metadata(self, obj_id, rdt_object):
        ready = _READY.get()
        if not rdt_object or ready is None:
            raise ValueError("RGB publication requires scoped completion proofs")
        for rgb in rdt_object:
            proof = ready.get(id(rgb))
            if proof is None or proof[0] is not rgb:
                raise ValueError("RGB tensor has no exact completion proof")
            _validate(rgb, proof[1])
        device = rdt_object[0].device
        if any(t.device != device for t in rdt_object):
            raise ValueError("RGB batch must use one CUDA device")
        import torch

        # Completion events belong to page-worker threads. Ray can publish on
        # another thread with no current CUDA context. Bind its primary context
        # before UCX registration; this does not wait for unrelated streams.
        torch.cuda.set_device(device)
        with self._cache_lock:
            if self._memory_pool is not None:
                raise RuntimeError("Ready RGB transport does not use Ray copy pools")
            agent = self.get_nixl_agent()
            self._add_tensor_descs(rdt_object)
            try:
                desc = agent.get_xfer_descs(rdt_object)
                meta = NixlTransportMetadata(
                    tensor_meta=[(t.shape, t.dtype) for t in rdt_object],
                    tensor_device=device.type,
                    nixl_serialized_descs=agent.get_serialized_descs(desc),
                    nixl_agent_meta=agent.get_agent_metadata(),
                    nixl_agent_name=agent.name,
                    nixl_agent_meta_version=self._nixl_agent_meta_version,
                )
                self._put_meta(obj_id, meta)
                return meta
            except BaseException:
                self._remove_tensor_descs(rdt_object)
                raise


def register():
    """Register only this transport; leave Ray's stock NIXL manager unchanged."""
    _check_version()
    import torch
    from ray.experimental import register_tensor_transport
    from ray.experimental.rdt.util import transport_manager_info

    with _LOCK:
        existing = transport_manager_info.get(NAME)
        if existing is None:
            register_tensor_transport(NAME, ["cuda"], ReadyNixlTransport, torch.Tensor)
        elif existing.transport_manager_class is not ReadyNixlTransport:
            raise RuntimeError("Ready transport name collision")


def manager():
    """Use the same manager for source/target arena registration and transfers."""
    register()
    from ray.experimental.rdt.util import get_tensor_transport_manager

    return get_tensor_transport_manager(NAME)


def publish(rgb, events):
    """Publish immutable original tensors whose own completion events are ready.

    Keep proofs only for the synchronous ray.put call. Ray then owns the tensor
    references and registration metadata until all downstream borrowers release.
    """
    if not rgb or len(rgb) != len(events):
        raise ValueError("Every RGB tensor needs a completion event")
    for t, e in zip(rgb, events):
        _validate(t, e)
    register()
    token = _READY.set({id(t): (t, e) for t, e in zip(rgb, events)})
    try:
        return ray.put(rgb, _tensor_transport=NAME)
    finally:
        _READY.reset(token)
