# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Packed detector geometry retains values, page boundaries, and metadata types."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from nemo_retriever.common.modality.page_elements import gpu_model  # noqa: E402


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU geometry comparison")
    return request.param


def inputs(device):
    generator = torch.Generator().manual_seed(713)
    predictions = []
    for count in [0, 17, 0, 8]:
        values = torch.rand((count, 7), generator=generator)
        values[:, :4] = values[:, :4] * 4096 - 1024
        values[:, 6] = torch.randint(0, 6, (count,), generator=generator).float()
        predictions.append(values.to(device))
    predictions[0] = None
    sizes = torch.tensor([[1, 4096], [17, 31], [1009, 2053], [1024, 1024]], device=device)
    return predictions, sizes


def assert_predictions_equal(expected, actual):
    assert len(expected) == len(actual)
    for left, right in zip(expected, actual):
        assert left.keys() == right.keys()
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize(
    "case", ["ragged", "empty", "none", "mixed_empty_dtype", "double_predictions", "floating_sizes", "nonfinite"]
)
def test_packed_geometry_matches_reference(device, normalize, case):
    predictions, sizes = inputs(device)
    if case == "empty":
        predictions = [torch.empty((0, 7), device=device) for _ in predictions]
    elif case == "none":
        predictions = [None for _ in predictions]
    elif case == "mixed_empty_dtype":
        predictions[2] = predictions[2].double()
    elif case == "double_predictions":
        predictions = [p.double() if p is not None else None for p in predictions]
    elif case == "floating_sizes":
        sizes = sizes.double()
    elif case == "nonfinite":
        predictions[1][0, 0] = float("nan")
        predictions[3][0, :4] = torch.tensor([float("-inf"), 0, float("inf"), 2048], device=device)
        predictions[3][1, :4] = torch.tensor([0, 0, 1024, 1024], device=device)
        predictions[3][1, 4] = float("nan")
    wrapper = SimpleNamespace(img_size=(1024, 1024), min_bbox_size=1, normalize_boxes=normalize)
    cpu_sizes = sizes.cpu().tolist()
    expected = gpu_model._geometry_reference(wrapper, predictions, sizes, cpu_sizes)
    actual = gpu_model._geometry_bulk(wrapper, predictions, sizes, cpu_sizes)
    assert_predictions_equal(expected, actual)


def test_eligible_geometry_uses_packed_path(device, monkeypatch):
    predictions, sizes = inputs(device)
    wrapper = SimpleNamespace(img_size=(1024, 1024), min_bbox_size=1, normalize_boxes=True)
    cpu_sizes = sizes.cpu().tolist()
    expected = gpu_model._geometry_reference(wrapper, predictions, sizes, cpu_sizes)

    def reference_not_expected(*args):
        raise AssertionError("Eligible predictions must exercise the packed geometry path")

    monkeypatch.setattr(gpu_model, "_geometry_reference", reference_not_expected)
    actual = gpu_model._geometry_bulk(wrapper, predictions, sizes, cpu_sizes)
    assert_predictions_equal(expected, actual)
