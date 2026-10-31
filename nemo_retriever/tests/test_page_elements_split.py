# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io

import pandas as pd
import pyarrow as pa
import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")
pe_model = pytest.importorskip("nemo_retriever.models.local.nemotron_page_elements_v3")

from nemo_retriever.common.modality.page_elements.shared import detect_page_elements_v3  # noqa: E402
from nemo_retriever.common.modality.page_elements.split import (  # noqa: E402
    ERR_COLUMN,
    _load_err,
    out_columns,
    split_detect,
    split_postprocess,
    split_preprocess,
)


class _FakeModel:
    input_shape = (1024, 1024)

    def preprocess(self, tensor):
        return pe_model.NemotronPageElementsV3.preprocess(self, tensor)

    def __call__(self, batch, orig_shapes):
        preds = [{"mean": float(batch[i].mean())} for i in range(int(batch.shape[0]))]
        return preds[0] if isinstance(orig_shapes, tuple) else preds

    def postprocess(self, preds):
        boxes, labels, scores = [], [], []
        for p in preds:
            if p["mean"] > 200:
                raise RuntimeError("postprocess failed")
            f = p["mean"] / 1000.0
            boxes.append(torch.tensor([[0.1, 0.1, 0.5, 0.2 + f], [0.05, 0.3, 0.9, 0.95]], dtype=torch.float32))
            labels.append(torch.tensor([2, 0], dtype=torch.int64))
            scores.append(torch.tensor([0.9, 0.3 + f], dtype=torch.float32))
        return boxes, labels, scores


def _png_b64(value: int, size=(64, 48), color=None) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color or (value, value // 2, 255 - value)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _strip(payload):
    out = {k: v for k, v in payload.items() if k != "timing"}
    if isinstance(out.get("error"), dict):
        out["error"] = {k: v for k, v in out["error"].items() if k != "traceback"}
    return out


def test_split_path_matches_detect_page_elements_v3() -> None:
    pages = pd.DataFrame(
        {
            "page_number": [1, 2, 3, 4, 5],
            "page_image": [
                {"image_b64": _png_b64(40)},
                {"image_b64": "not-an-image"},
                {"image_b64": _png_b64(120, size=(80, 30))},
                {"image_b64": _png_b64(0, color=(250, 250, 250))},
                {"image_b64": _png_b64(90)},
            ],
        }
    )
    model = _FakeModel()
    expected = detect_page_elements_v3(pages, model=model, inference_batch_size=2)
    got = split_postprocess(split_detect(split_preprocess(pages), model=model, inference_batch_size=2))

    assert list(got.columns) == list(expected.columns)
    assert not set(out_columns()) & set(got.columns)
    assert [_strip(p) for p in got["page_elements_v3"]] == [_strip(p) for p in expected["page_elements_v3"]]
    assert got["page_elements_v3_num_detections"].tolist() == expected["page_elements_v3_num_detections"].tolist()
    assert got["page_elements_v3_counts_by_label"].tolist() == expected["page_elements_v3_counts_by_label"].tolist()
    stages = [(p.get("error") or {}).get("stage") for p in got["page_elements_v3"]]
    assert stages == [None, "decode_image", None, "postprocess", "postprocess"]
    assert got["page_elements_v3_num_detections"][0] > 0


def test_split_arrow_round_trip_preserves_valid_pages_with_decode_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_retriever.models import local as local_models
    from nemo_retriever.models import warmup_registry
    from nemo_retriever.operators.extract.page_elements.gpu_actor import PageElementDetectionActor

    pages = pd.DataFrame(
        {
            "page_number": [1, 2],
            "page_image": [
                {"image_b64": _png_b64(40)},
                {"image_b64": "not-an-image"},
            ],
        }
    )
    preprocessed = split_preprocess(pages)
    preprocessed = pa.Table.from_pandas(preprocessed, preserve_index=False).to_pandas(types_mapper=pd.ArrowDtype)
    model = _FakeModel()
    monkeypatch.setattr(local_models, "NemotronPageElementsV3", lambda: model, raising=False)
    monkeypatch.setattr(warmup_registry, "get_warmed_model", lambda _name: None)
    actor = PageElementDetectionActor(split_mode=True, inference_batch_size=2)
    detected = actor(preprocessed)
    detected = pa.Table.from_pandas(detected, preserve_index=False).to_pandas(types_mapper=pd.ArrowDtype)
    got = split_postprocess(detected)

    stages = [(p.get("error") or {}).get("stage") for p in got["page_elements_v3"]]
    assert stages == [None, "decode_image"]
    assert got["page_elements_v3_num_detections"].tolist()[0] > 0
    assert got["page_elements_v3_num_detections"].tolist()[1] == 0


@pytest.mark.parametrize("value", [None, pd.NA, float("nan")])
def test_load_err_treats_missing_scalars_as_empty(value) -> None:
    assert _load_err(value) is None


def test_split_postprocess_handles_arrow_null_intermediates() -> None:
    pages = pd.DataFrame(
        {
            "_pe_boxes": [pd.NA, []],
            "_pe_labels": [pd.NA, []],
            "_pe_scores": [pd.NA, []],
            "_pe_elapsed": [0.1, pd.NA],
            ERR_COLUMN: [None, None],
        }
    )
    pages = pa.Table.from_pandas(pages, preserve_index=False).to_pandas(types_mapper=pd.ArrowDtype)

    got = split_postprocess(pages)
    payloads = got["page_elements_v3"].tolist()

    assert [payload.get("detections") for payload in payloads] == [[], []]
    assert [payload.get("error") for payload in payloads] == [None, None]
    assert got["page_elements_v3_num_detections"].tolist() == [0, 0]
