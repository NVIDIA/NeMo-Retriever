# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""GPU image ownership and crop/preprocessing contracts without model weights."""

import numpy as np
import pytest
from nemo_retriever.operators.extract.pdf.gpu_transport import (
    GpuPageImage,
    GpuTransportError,
    gpu_image_scope,
)


def test_gpu_image_requires_scoped_consumption():
    image = GpuPageImage(object(), 0, 12, 10)
    with pytest.raises(GpuTransportError, match="scoped"):
        image.tensor()


@pytest.fixture
def gpu_rgb():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for device tensor contracts")
    return torch.arange(20 * 30 * 3, device="cuda", dtype=torch.int64).remainder(256).to(torch.uint8).reshape(20, 30, 3)


def test_scoped_transport_caches_gpu_fetch_and_checks_shape(gpu_rgb, monkeypatch):
    import ray

    calls = []

    def fetch(ref):
        calls.append(ref)
        return [gpu_rgb]

    monkeypatch.setattr(ray, "get", fetch)
    ref = object()
    image = GpuPageImage(ref, 0, 20, 30)
    with gpu_image_scope():
        assert image.tensor() is image.tensor() is gpu_rgb
        with pytest.raises(GpuTransportError, match="expected GPU RGB"):
            GpuPageImage(ref, 0, 30, 20).tensor()
    assert calls == [ref]
    with gpu_image_scope():
        assert image.tensor() is gpu_rgb
    assert calls == [ref, ref]


def test_receive_targets_are_reused_within_scope_and_finished_before_release(gpu_rgb, monkeypatch):
    import ray
    import torch
    from nemo_retriever.operators.extract.pdf import gpu_transport

    buffers = []
    calls = []

    class Pool:
        def image(self, height, width):
            result = torch.empty((height, width, 3), dtype=torch.uint8, device="cuda")
            buffers.append(result)
            return result

    monkeypatch.setattr(gpu_transport, "_receive_pool", lambda capacity: Pool())
    monkeypatch.setattr(ray.experimental, "set_target_for_ref", lambda ref, target: calls.append((ref, target)))

    def fetch(ref):
        buffers[0].copy_(gpu_rgb)
        return buffers

    monkeypatch.setattr(ray, "get", fetch)
    original_sync = torch.cuda.synchronize
    syncs = []

    def sync(*args, **kwargs):
        syncs.append(True)
        return original_sync(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    ref = object()
    image = GpuPageImage(ref, 0, 20, 30, ((20, 30),))
    with gpu_image_scope(receive_pool_bytes=3145728):
        with gpu_image_scope(receive_pool_bytes=3145728):
            assert image.tensor() is buffers[0]
            assert image.tensor() is buffers[0]
            assert torch.equal(buffers[0], gpu_rgb)
        assert syncs == []
    assert len(calls) == len(syncs) == 1
    assert calls[0][0] is ref


def test_receive_pool_requires_complete_batch_shapes():
    image = GpuPageImage(object(), 0, 12, 10)
    with gpu_image_scope(receive_pool_bytes=3145728):
        with pytest.raises(GpuTransportError, match="complete batch shape"):
            image.tensor()


def test_nested_scopes_cannot_change_receive_capacity():
    with gpu_image_scope():
        with pytest.raises(GpuTransportError, match="same receive-pool capacity"):
            with gpu_image_scope(receive_pool_bytes=3145728):
                pass


def test_gpu_crops_and_padding_match_pixel_bounds(gpu_rgb, monkeypatch):
    import ray
    import torch
    from nemo_retriever.common.modality.ocr.shared import (
        _crop_all_from_page,
        _prepare_ocr_crop,
    )

    monkeypatch.setattr(ray, "get", lambda ref: [gpu_rgb])
    image = GpuPageImage(object(), 0, 20, 30)
    detections = [
        {"label_name": "table", "bbox_xyxy_norm": [0.1, 0.2, 0.8, 0.9]},
        {"label_name": "table", "bbox_xyxy_norm": [0.2, 0.2, 0.2, 0.8]},
    ]
    with gpu_image_scope():
        crops = _crop_all_from_page(image, detections, {"table"})
        assert len(crops) == 1
        actual = crops[0][2]
        assert actual.is_cuda and torch.equal(actual, gpu_rgb[4:18, 3:24])
        padded, original, square = _prepare_ocr_crop(actual, "bottom_right")
        assert original == (14, 21) and square == (21, 21)
        assert padded.is_cuda and torch.equal(padded[:14, :21], actual)
        assert bool((padded[14:] == 255).all())
        with pytest.raises(ValueError, match="encoded host"):
            _crop_all_from_page(image, detections, {"table"}, as_b64=True)


def test_detector_preprocessing_preserves_rgb_scale_and_padding(gpu_rgb):
    from nemo_retriever.common.modality.page_postprocessing import (
        prepare_detector_input,
        prepare_detector_input_gpu,
    )

    # At native size the image bytes and gray padding must match exactly.
    actual = prepare_detector_input_gpu(gpu_rgb, (20, 40))
    expected = prepare_detector_input(gpu_rgb.cpu().numpy(), (20, 40))
    assert actual.is_cuda and actual.shape == (3, 20, 40)
    np.testing.assert_array_equal(actual.cpu().numpy(), expected)


def test_model_stage_errors_stop_instead_of_becoming_index_rows():
    import pandas as pd
    from nemo_retriever.operators.extract.pdf.gpu_transport import raise_stage_errors

    raise_stage_errors(pd.DataFrame({"ocr": [{"error": None}]}), "ocr")
    with pytest.raises(GpuTransportError, match="stage failed"):
        raise_stage_errors(pd.DataFrame({"ocr": [{"error": {"message": "transfer failed"}}]}), "ocr")


@pytest.mark.parametrize("height,width", [(3, 7), (7, 3), (20, 30)])
def test_gpu_ocr_has_explicit_hwc_contract(height, width, gpu_rgb, monkeypatch):
    import torch
    from nemo_retriever.models.local.nemotron_ocr_v2 import NemotronOCRV2

    image = gpu_rgb[:height, :width]
    model = NemotronOCRV2.__new__(NemotronOCRV2)
    calls = []
    model._model = lambda images, **kwargs: calls.append(images) or ["recognized"]

    def no_cpu(*args, **kwargs):
        raise AssertionError("GPU RGB must not be read back")

    monkeypatch.setattr(torch.Tensor, "cpu", no_cpu)
    assert model.invoke_gpu_rgb([image]) == ["recognized"]
    actual = calls[0][0]
    assert actual.is_cuda and actual.is_contiguous()
    assert actual.shape == (3, height, width)
    assert torch.equal(actual, image.permute(2, 0, 1))
    with pytest.raises(ValueError, match="CUDA uint8 HWC"):
        model.invoke_gpu_rgb(image.float())


def test_legacy_gpu_ocr_tensors_keep_chw_bchw_conversion(gpu_rgb, monkeypatch):
    from nemo_retriever.models.local.nemotron_ocr_v2 import NemotronOCRV2

    model = NemotronOCRV2.__new__(NemotronOCRV2)
    encoded = []
    calls = []
    monkeypatch.setattr(model, "_tensor_to_png_b64", lambda image: encoded.append(image.shape) or "png")
    model._model = lambda image, **kwargs: calls.append(image) or ["recognized"]
    chw = gpu_rgb.permute(2, 0, 1).float() / 255
    assert model.invoke(chw) == ["recognized"]
    assert model.invoke(chw.unsqueeze(0).expand(2, -1, -1, -1)) == ["recognized", "recognized"]
    assert encoded == [chw.shape] * 3 and calls == [b"png"] * 3


def test_gpu_transport_rejects_unsupported_outputs():
    from nemo_retriever.common.params import EmbedParams, ExtractParams
    from nemo_retriever.graph.ingestor_runtime import build_graph

    with pytest.raises(ValueError, match="encoded image outputs"):
        build_graph(extraction_mode="pdf", extract_params=ExtractParams(pdf_renderer="cuda_rdt", extract_images=True))
    with pytest.raises(ValueError, match="text embeddings"):
        build_graph(
            extraction_mode="pdf",
            extract_params=ExtractParams(pdf_renderer="cuda_rdt"),
            embed_params=EmbedParams(embed_modality="image"),
        )


def test_failed_batch_drains_other_descriptor_readers_before_returning():
    import time
    from concurrent.futures import ThreadPoolExecutor

    from nemo_retriever.operators.extract.pdf.gpu_transport import _GpuProducer

    producer = _GpuProducer.__new__(_GpuProducer)
    producer._owner_leases = None
    completed = []

    def read(handle):
        if handle == 0:
            raise RuntimeError("invalid scene descriptor")
        time.sleep(0.02)
        completed.append(handle)
        return object()

    with ThreadPoolExecutor(max_workers=2) as executor:
        producer._pages = executor
        producer._render_shared_page = read
        with pytest.raises(RuntimeError, match="invalid scene"):
            producer.render_handles([0, 1, 2])
        assert sorted(completed) == [1, 2]


def test_owner_image_finishes_gpu_uses_even_without_receive_pool(gpu_rgb, monkeypatch):
    import pandas as pd
    import ray
    import torch
    from types import SimpleNamespace
    from nemo_retriever.operators.extract.pdf import gpu_transport

    events = []

    def control_options(**kwargs):
        assert kwargs == {"concurrency_group": "owner_control"}
        return SimpleNamespace(remote=lambda batches: events.append("ack") or True)

    owner = SimpleNamespace(acknowledge_images=SimpleNamespace(options=control_options))
    ref = object()
    image = GpuPageImage(ref, 0, 20, 30, ((20, 30),), owner, 1)
    monkeypatch.setattr(ray, "get", lambda value: [gpu_rgb] if value is ref else [True])
    original = torch.cuda.synchronize

    def synchronize():
        original()
        events.append("gpu_finished")

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    with gpu_image_scope():
        image.tensor()
    gpu_transport.acknowledge_gpu_images(pd.DataFrame({"page_image": [{"gpu_image": image}]}))
    assert events == ["gpu_finished", "ack"]
    events.clear()
    with gpu_image_scope():
        GpuPageImage(ref, 0, 20, 30).tensor()
    assert events == []
