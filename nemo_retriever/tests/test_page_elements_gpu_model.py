# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Original page-size metadata preserves upstream detector geometry exactly."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
upstream = pytest.importorskip("nemotron_page_elements_v3.model")

from nemo_retriever.common.modality.page_elements import gpu_model  # noqa: E402
from nemo_retriever.common.modality.page_elements.gpu_model import forward_with_cpu_shapes  # noqa: E402


@pytest.fixture
def audited_provider():
    if not gpu_model._supports_forward(upstream.YoloXWrapper):
        pytest.skip("The installed provider uses a different geometry contract; its original CUDA forward is retained")


class FixedPredictions(torch.nn.Module):
    def __init__(self, predictions):
        super().__init__()
        self.register_buffer("predictions", predictions)

    def forward(self, x):
        return self.predictions.clone()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("agnostic", [False, True])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("single", [False, True])
def test_cpu_shapes_match_upstream_geometry(audited_provider, device, normalize, agnostic, empty, single):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for device arithmetic comparison")
    shapes = [(1, 4096), (17, 31), (1009, 2053), (4096, 1)]
    if single:
        shapes = shapes[:1]
    predictions = torch.rand((len(shapes), 96, 11), generator=torch.Generator().manual_seed(713))
    predictions[:, :, :4] *= 2048
    if empty:
        predictions[:, :, 4] = 0
    config = SimpleNamespace(
        device=device,
        size=(1024, 1024),
        min_bbox_size=1,
        normalize_boxes=normalize,
        conf_thresh=0.1,
        iou_thresh=0.5,
        class_agnostic=agnostic,
        thresholds_per_class={},
        labels=[],
        num_classes=6,
    )
    wrapper = upstream.YoloXWrapper(FixedPredictions(predictions.to(device)), config)
    x = torch.full((len(shapes), 3, 16, 16), 114.0, device=device)
    original_shapes = shapes[0] if single else shapes
    expected = wrapper(x, original_shapes)
    actual = forward_with_cpu_shapes(wrapper, x, original_shapes)
    assert len(expected) == len(actual) == len(shapes)
    for left, right in zip(expected, actual):
        assert left.keys() == right.keys()
        for key in left:
            assert left[key].dtype == right[key].dtype
            assert left[key].device == right[key].device
            assert torch.equal(left[key], right[key])


@pytest.mark.parametrize("changed_method", ["forward", "reformat_input"])
def test_changed_provider_keeps_original_invocation(monkeypatch, changed_method):
    from nemo_retriever.models.local.nemotron_page_elements_v3 import NemotronPageElementsV3

    def changed(self, *args):
        raise AssertionError("The compatibility check must not execute provider methods")

    wrapper_type = type("ChangedGeometry", (upstream.YoloXWrapper,), {changed_method: changed})
    model = object.__new__(NemotronPageElementsV3)
    model._model = object.__new__(wrapper_type)
    image, shapes, expected = object(), [(17, 31)], object()
    calls = []

    def reference(x, original_shapes):
        calls.append((x, original_shapes))
        return expected

    monkeypatch.setattr(model, "invoke", reference)
    assert not gpu_model._supports_forward(wrapper_type)
    assert model._invoke_cuda(image, shapes) is expected
    assert calls == [(image, shapes)]


def test_unavailable_provider_source_keeps_original_contract(monkeypatch):
    class UnavailableSource(upstream.YoloXWrapper):
        pass

    def unavailable(function):
        raise OSError("Source is unavailable")

    monkeypatch.setattr(gpu_model.inspect, "getsource", unavailable)
    assert not gpu_model._supports_forward(UnavailableSource)


def test_provider_documentation_does_not_change_geometry_contract(audited_provider, monkeypatch):
    class DifferentDocumentation(upstream.YoloXWrapper):
        pass

    original_source = gpu_model.inspect.getsource

    def different_documentation(function):
        source = original_source(function)
        changed = source.replace("Args:", "Parameters:")
        assert changed != source
        return changed

    monkeypatch.setattr(gpu_model.inspect, "getsource", different_documentation)
    assert gpu_model._supports_forward(DifferentDocumentation)
