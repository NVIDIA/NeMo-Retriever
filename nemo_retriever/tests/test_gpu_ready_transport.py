# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Completion and cleanup contracts for the opt-in RGB transport."""
from types import SimpleNamespace

import pytest

from nemo_retriever.operators.extract.pdf import gpu_ready_transport as ready
from nemo_retriever.operators.extract.pdf.gpu_transport import _ready_transport_enabled


@pytest.mark.parametrize("value,enabled", [(None, False), ("0", False), ("1", True)])
def test_opt_in(value, enabled, monkeypatch):
    monkeypatch.delenv("CUDA_PDF_READY_TRANSPORT", raising=False)
    if value is not None:
        monkeypatch.setenv("CUDA_PDF_READY_TRANSPORT", value)
    assert _ready_transport_enabled() is enabled


@pytest.mark.parametrize("value", ["", "true", "2", "-1"])
def test_invalid_opt_in(value, monkeypatch):
    monkeypatch.setenv("CUDA_PDF_READY_TRANSPORT", value)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        _ready_transport_enabled()


def test_incompatible_ray_fails_closed(monkeypatch):
    monkeypatch.setattr(ready.ray, "__version__", "different")
    with pytest.raises(RuntimeError, match="requires Ray 2.56.1"):
        ready.register()


@pytest.fixture
def completed_rgb():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA completion events require a GPU")
    rgb = torch.ones((4, 5, 3), device="cuda", dtype=torch.uint8)
    event = torch.cuda.Event()
    event.record()
    event.synchronize()
    return rgb, event


def test_publish_proofs_are_scoped_and_reset_on_error(completed_rgb, monkeypatch):
    rgb, event = completed_rgb
    monkeypatch.setattr(ready, "register", lambda: None)

    def put(tensors, **kwargs):
        assert tensors[0] is rgb
        assert ready._READY.get()[id(rgb)] == (rgb, event)
        assert kwargs == {"_tensor_transport": ready.NAME}
        raise RuntimeError("transport stopped")

    monkeypatch.setattr(ready.ray, "put", put)
    with pytest.raises(RuntimeError, match="transport stopped"):
        ready.publish([rgb], [event])
    assert ready._READY.get() is None


@pytest.mark.parametrize("which", ["empty", "missing", "wrong_dtype", "wrong_shape", "strided", "no_event"])
def test_bad_output_is_rejected_before_ray_put(completed_rgb, which, monkeypatch):
    rgb, event = completed_rgb
    tensors, events = [rgb], [event]
    if which == "empty":
        tensors, events = [], []
    elif which == "missing":
        events = []
    elif which == "wrong_dtype":
        tensors = [rgb.float()]
    elif which == "wrong_shape":
        tensors = [rgb[0]]
    elif which == "strided":
        tensors = [rgb.transpose(0, 1)]
    elif which == "no_event":
        events = [None]
    monkeypatch.setattr(ready.ray, "put", lambda *a, **k: pytest.fail("Unproved pixels reached Ray"))
    with pytest.raises(ValueError):
        ready.publish(tensors, events)


def test_metadata_requires_exact_tensor_identity(completed_rgb):
    rgb, event = completed_rgb
    transport = ready.ReadyNixlTransport()
    with pytest.raises(ValueError, match="scoped completion"):
        transport.extract_tensor_transport_metadata("unused", [rgb])
    token = ready._READY.set({id(rgb): (rgb, event)})
    try:
        with pytest.raises(ValueError, match="exact completion"):
            transport.extract_tensor_transport_metadata("unused", [rgb.view_as(rgb)])
    finally:
        ready._READY.reset(token)


def test_descriptor_failure_rolls_back_registration(completed_rgb, monkeypatch):
    rgb, event = completed_rgb
    transport = ready.ReadyNixlTransport()
    calls = []
    agent = SimpleNamespace(get_xfer_descs=lambda _: (_ for _ in ()).throw(RuntimeError("descriptor failed")))
    monkeypatch.setattr(transport, "get_nixl_agent", lambda: agent)
    monkeypatch.setattr(transport, "_add_tensor_descs", lambda tensors: calls.append(("add", tensors)))
    monkeypatch.setattr(transport, "_remove_tensor_descs", lambda tensors: calls.append(("remove", tensors)))
    token = ready._READY.set({id(rgb): (rgb, event)})
    try:
        with pytest.raises(RuntimeError, match="descriptor failed"):
            transport.extract_tensor_transport_metadata("unused", [rgb])
    finally:
        ready._READY.reset(token)
    assert [name for name, _ in calls] == ["add", "remove"]
    assert all(tensors[0] is rgb for _, tensors in calls)
    assert transport._get_num_managed_meta_nixl() == 0


def test_aborted_fetch_fails_without_registering_targets():
    from ray.experimental.rdt.nixl_tensor_transport import NixlCommunicatorMetadata, NixlTransportMetadata

    transport = ready.ReadyNixlTransport()
    metadata = NixlTransportMetadata(tensor_meta=[], tensor_device="cuda")
    communicator = NixlCommunicatorMetadata()
    transport.abort_transport("aborted", communicator)
    with pytest.raises(RuntimeError, match="transfer aborted"):
        transport.fetch_multiple_tensors("aborted", metadata, communicator, target_buffers=[object()])
    assert transport._get_num_managed_meta_nixl() == 0
    assert not transport._tensor_desc_cache
    assert "aborted" not in transport._aborted_transfer_obj_ids


def test_transport_name_collision_fails_closed(monkeypatch):
    from ray.experimental.rdt import util

    monkeypatch.setattr(util, "transport_manager_info", {ready.NAME: SimpleNamespace(transport_manager_class=object)})
    with pytest.raises(RuntimeError, match="name collision"):
        ready.register()


def test_publication_binds_context_on_fresh_ray_thread(completed_rgb, monkeypatch):
    import ctypes
    from concurrent.futures import ThreadPoolExecutor

    try:
        driver = ctypes.CDLL("libcuda.so.1")
    except OSError:
        pytest.skip("The Linux CUDA driver is required for the thread-context regression")
    driver.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

    def has_context():
        context = ctypes.c_void_p()
        assert driver.cuCtxGetCurrent(ctypes.byref(context)) == 0
        return bool(context.value)

    rgb, event = completed_rgb
    transport = ready.ReadyNixlTransport()
    agent = SimpleNamespace(
        get_xfer_descs=lambda tensors: object(),
        get_serialized_descs=lambda descriptors: b"descriptors",
        get_agent_metadata=lambda: b"agent",
        name="test",
    )

    def register(tensors):
        # UCX may query the current CUDA context while registering memory.
        assert has_context(), "Publication left its thread without a CUDA context"

    monkeypatch.setattr(transport, "get_nixl_agent", lambda: agent)
    monkeypatch.setattr(transport, "_add_tensor_descs", register)

    def publish_on_new_thread():
        assert not has_context()
        token = ready._READY.set({id(rgb): (rgb, event)})
        try:
            return transport.extract_tensor_transport_metadata("thread-test", [rgb])
        finally:
            ready._READY.reset(token)

    # Recreate the old missing-binding path, then verify the repaired path.
    import torch

    with monkeypatch.context() as missing_binding:
        missing_binding.setattr(torch.cuda, "set_device", lambda device: None)
        with ThreadPoolExecutor(max_workers=1) as executor:
            with pytest.raises(AssertionError, match="without a CUDA context"):
                executor.submit(publish_on_new_thread).result(timeout=15)
    with ThreadPoolExecutor(max_workers=1) as executor:
        metadata = executor.submit(publish_on_new_thread).result(timeout=15)
    assert metadata.tensor_meta == [(rgb.shape, rgb.dtype)]
