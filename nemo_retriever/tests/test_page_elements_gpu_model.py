# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Original page-size metadata preserves upstream detector geometry exactly."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
upstream = pytest.importorskip("nemotron_page_elements_v3.model")

from nemo_retriever.common.modality.page_elements.gpu_model import forward_with_cpu_shapes  # noqa: E402


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
def test_cpu_shapes_match_upstream_geometry(device, normalize, agnostic, empty, single):
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
