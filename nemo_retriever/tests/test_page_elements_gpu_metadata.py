# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Detection metadata keeps exact values and page boundaries during readback."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from nemo_retriever.common.modality.page_elements.gpu_metadata import prediction_batch_to_cpu  # noqa: E402
from nemo_retriever.models.local.nemotron_page_elements_v3 import NemotronPageElementsV3  # noqa: E402


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for detection metadata readback")
    return torch.device("cuda")


def prediction(count, device, dtype=torch.float32):
    return {
        "boxes": torch.arange(count * 4, dtype=dtype, device=device).reshape(count, 4) / 17,
        "labels": torch.arange(count, dtype=dtype, device=device).remainder(7),
        "scores": torch.arange(count, dtype=dtype, device=device) / 13,
    }


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_variable_page_predictions_use_one_exact_readback(cuda_device, dtype, monkeypatch):
    predictions = [prediction(3, cuda_device, dtype), prediction(0, "cpu", dtype), prediction(7, cuda_device, dtype)]
    expected = [{key: tensor.cpu().clone() for key, tensor in page.items()} for page in predictions]
    copies = []
    original_cpu = torch.Tensor.cpu

    def copy_to_cpu(tensor, *args, **kwargs):
        if tensor.is_cuda:
            copies.append(tuple(tensor.shape))
        return original_cpu(tensor, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", copy_to_cpu)
        actual = prediction_batch_to_cpu(predictions)

    assert copies == [(10, 6)]
    for expected_page, actual_page in zip(expected, actual):
        for key, tensor in actual_page.items():
            assert tensor.device.type == "cpu"
            assert tensor.dtype == dtype
            assert torch.equal(tensor, expected_page[key])


def test_integer_labels_do_not_pass_through_float_packing(cuda_device):
    predictions = [prediction(2, cuda_device)]
    predictions[0]["labels"] = torch.tensor([2**30 + 1, 2**30 + 3], dtype=torch.int64, device=cuda_device)
    assert prediction_batch_to_cpu(predictions) is predictions
    assert predictions[0]["labels"].tolist() == [2**30 + 1, 2**30 + 3]


def test_empty_and_cpu_predictions_remain_compatible():
    for predictions in [[], [prediction(0, "cpu")], [prediction(2, "cpu")]]:
        assert prediction_batch_to_cpu(predictions) is predictions


def test_batched_readback_retains_thresholded_postprocessing(cuda_device):
    model = object.__new__(NemotronPageElementsV3)
    model._model = SimpleNamespace(thresholds_per_class={"table": 0.4, "text": 0.5}, labels=["table", "text"])
    predictions = [prediction(3, cuda_device), prediction(0, "cpu"), prediction(2, cuda_device)]
    for page in predictions:
        page["labels"].remainder_(2)
    predictions[0]["scores"][:] = torch.tensor([0.0, 0.5, 0.9], device=cuda_device)
    original = model.postprocess(predictions)
    packed = model._postprocess_cuda(predictions)
    for expected_pages, actual_pages in zip(original, packed):
        assert len(expected_pages) == len(actual_pages) == 3
        for expected, actual in zip(expected_pages, actual_pages):
            assert expected.dtype == actual.dtype
            assert torch.equal(expected, actual)
    original_single = model.postprocess(predictions[0])
    packed_single = model._postprocess_cuda(predictions[0])
    assert all(torch.equal(expected, actual) for expected, actual in zip(original_single, packed_single))


@pytest.mark.parametrize("gpu_input,specialized", [(False, True), (True, True), (True, False)])
def test_extraction_selects_specialized_postprocess_only_for_supported_gpu_inputs(
    cuda_device, monkeypatch, gpu_input, specialized
):
    import numpy as np
    import pandas as pd

    from nemo_retriever.common.modality.page_elements import shared
    from nemo_retriever.operators.extract.pdf.gpu_transport import GpuPageImage

    calls = []
    invocations = []

    class Model:
        input_shape = (8, 8)

        def __call__(self, batch, shapes):
            invocations.append("public")
            return [prediction(0, "cpu") for _ in batch]

        def postprocess(self, predictions):
            calls.append("public")
            return tuple([page[key] for page in predictions] for key in ("boxes", "labels", "scores"))

    class SpecializedModel(Model):
        def _invoke_cuda(self, batch, shapes):
            invocations.append("cuda")
            return [prediction(0, "cpu") for _ in batch]

        def _postprocess_cuda(self, predictions):
            calls.append("cuda")
            return super().postprocess(predictions)

    model = SpecializedModel() if specialized else Model()
    if gpu_input:
        image = torch.zeros((8, 8, 3), dtype=torch.uint8, device=cuda_device)
        monkeypatch.setattr(GpuPageImage, "tensor", lambda self: image)
        page_image = {"gpu_image": GpuPageImage(object(), 0, 8, 8)}
    else:
        monkeypatch.setattr(
            shared, "_decode_b64_image_to_np_array", lambda value: (np.zeros((8, 8, 3), dtype=np.uint8), (8, 8))
        )
        page_image = {"image_b64": "placeholder"}
    result = shared.detect_page_elements_v3(pd.DataFrame({"page_image": [page_image]}), model=model)
    assert result.iloc[0]["page_elements_v3"]["error"] is None
    assert calls == (["cuda", "public"] if gpu_input and specialized else ["public"])
    assert invocations == (["cuda"] if gpu_input and specialized else ["public"])
