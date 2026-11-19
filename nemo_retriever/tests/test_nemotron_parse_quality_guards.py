# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image

from nemo_retriever.common.params import ExtractParams, ParseParams
from nemo_retriever.common.stage_errors import iter_stage_errors_from_value
from nemo_retriever.common.params.utils import build_parse_actor_kwargs
from nemo_retriever.operators.extract.parse.nemotron_parse import _text_layer_fallback, nemotron_parse_pages


@pytest.fixture(autouse=True)
def _parse35_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decode tagged Parse 3.5 output without downloading HF's helper package."""
    from nemo_retriever.common.modality.parse import nemotron_parse35_postprocessing as postprocessing
    from nemo_retriever.common.modality.parse.nemotron_parse_postprocessing import extract_classes_bboxes

    def postprocess_document(raw_text: str) -> dict:
        classes, boxes, texts = extract_classes_bboxes(raw_text)
        return {
            "blocks": [
                {"class": cls, "bbox": list(box), "text": text, "content": text}
                for cls, box, text in zip(classes, boxes, texts)
            ]
        }

    monkeypatch.setattr(
        postprocessing, "_load_parse35_helpers", lambda: SimpleNamespace(postprocess_document=postprocess_document)
    )


NATIVE = " ".join(f"native{i} revenue grew strongly" for i in range(10))


def _page_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class _FakeParse:
    model_name = "nvidia/Nemotron-3.5-Parse"

    def __init__(self, texts: list[str]) -> None:
        self._texts = texts

    def invoke_batch_results(self, images, task_prompt=None):
        return [SimpleNamespace(text=t, finish_reason="stop", output_tokens=10) for t in self._texts[: len(images)]]


def _tagged(text: str) -> str:
    return f"<x_0.1><y_0.1>{text}<x_0.9><y_0.2><class_Text>"


def _run(parse_text: str, native: str, min_overlap: float) -> pd.DataFrame:
    df = pd.DataFrame({"text": [native], "page_image": [{"image_b64": _page_b64()}], "path": ["a.pdf"]})
    return nemotron_parse_pages(
        df,
        model=_FakeParse([_tagged(parse_text)]),
        extract_text=True,
        text_layer_min_overlap=min_overlap,
    )


@pytest.mark.parametrize(
    ("parse_text", "native", "expected"),
    [
        ("Ɋǝʌǝnnǝ ƃɹǝʍ ʂʇɹouƃlʎ " * 10, NATIVE, True),
        ("", NATIVE, True),
        ("native1 revenue grew strongly native2", NATIVE, False),
        ("Ɋǝʌǝnnǝ ƃɹǝʍ", "too few native words", False),
    ],
)
def test_text_layer_fallback_rule(parse_text: str, native: str, expected: bool) -> None:
    assert _text_layer_fallback(parse_text, native, 0.3) is expected
    assert _text_layer_fallback(parse_text, native, 0.0) is False


def test_garbled_parse_text_is_replaced_by_text_layer() -> None:
    out = _run("Ɋǝʌǝnnǝ ƃɹǝʍ ʂʇɹouƃlʎ " * 10, NATIVE, 0.3)
    assert out.loc[0, "text"] == NATIVE
    assert out.loc[0, "nemotron_parse_v1_2"]["text_source"] == "text_layer"
    assert out.loc[0, "ordered_page_text"] == NATIVE


def test_matching_parse_text_is_kept_and_zero_disables() -> None:
    kept = _run("native1 revenue grew strongly", NATIVE, 0.3)
    assert kept.loc[0, "text"] == "native1 revenue grew strongly"
    assert "text_source" not in kept.loc[0, "nemotron_parse_v1_2"]
    disabled = _run("Ɋǝʌǝnnǝ ƃɹǝʍ " * 10, NATIVE, 0.0)
    assert disabled.loc[0, "text"] != NATIVE


def test_parse_actor_kwargs_default_and_override() -> None:
    assert build_parse_actor_kwargs(ExtractParams(method="nemotron_parse"))["text_layer_min_overlap"] == 0.3
    params = ExtractParams(method="nemotron_parse", parse=ParseParams(text_layer_min_overlap=0))
    assert build_parse_actor_kwargs(params)["text_layer_min_overlap"] == 0
    assert "text_layer_min_overlap" not in build_parse_actor_kwargs(params, chart_routing=True)
    with pytest.raises(ValueError):
        ParseParams(text_layer_min_overlap=1.5)


class _CappedParse(_FakeParse):
    def __init__(self, first: str) -> None:
        super().__init__([first])
        self.calls: list[int | None] = []

    def invoke_batch_results(self, images, task_prompt=None, max_tokens=None):
        self.calls.append(max_tokens)
        if max_tokens is None:
            return [SimpleNamespace(text=self._texts[0], finish_reason="length", output_tokens=4096) for _ in images]
        return [
            SimpleNamespace(text=_tagged("finished page"), finish_reason="stop", output_tokens=5000) for _ in images
        ]


def _run_capped(first: str, retry_tokens: int | None) -> tuple[pd.DataFrame, _CappedParse]:
    model = _CappedParse(first)
    df = pd.DataFrame({"text": [""], "page_image": [{"image_b64": _page_b64()}], "path": ["a.pdf"]})
    out = nemotron_parse_pages(df, model=model, extract_text=True, length_retry_max_tokens=retry_tokens)
    return out, model


def test_length_capped_page_is_rerun_with_larger_limit() -> None:
    out, model = _run_capped(_tagged("Varied opening text") + "<x_0.1><y_0.3>cut", 8192)
    assert model.calls == [None, 8192]
    assert out.loc[0, "text"] == "finished page"
    meta = out.loc[0, "nemotron_parse_v1_2"]
    assert meta["completion"]["status"] == "complete"
    assert meta["length_retry"] == {"max_tokens": 8192}


@pytest.mark.parametrize("retry_tokens", [None, 4096])
def test_length_retry_disabled_or_not_larger(retry_tokens: int | None) -> None:
    _, model = _run_capped(_tagged("Varied opening text") + "<x_0.1><y_0.3>cut", retry_tokens)
    assert model.calls == [None]


def test_repetitive_capped_page_is_not_rerun() -> None:
    out, model = _run_capped(_tagged("Total 0.00 " * 400) + "<x_0.1><y_0.3>Total 0.00", 8192)
    assert model.calls == [None]
    assert out.loc[0, "nemotron_parse_v1_2"]["completion"]["status"] == "incomplete"
    assert "length_retry" not in out.loc[0, "nemotron_parse_v1_2"]


def test_parse_actor_kwargs_length_retry_default() -> None:
    assert build_parse_actor_kwargs(ExtractParams(method="nemotron_parse"))["length_retry_max_tokens"] == 8192
    params = ExtractParams(method="nemotron_parse", parse=ParseParams(length_retry_max_tokens=None))
    assert build_parse_actor_kwargs(params)["length_retry_max_tokens"] is None


def test_failed_length_retry_keeps_first_pass_output() -> None:
    class _FailingRetry(_CappedParse):
        def invoke_batch_results(self, images, task_prompt=None, max_tokens=None):
            if max_tokens is not None:
                self.calls.append(max_tokens)
                raise RuntimeError("engine stopped")
            return super().invoke_batch_results(images, task_prompt)

    model = _FailingRetry(_tagged("Varied opening text") + "<x_0.1><y_0.3>cut")
    df = pd.DataFrame({"text": [""], "page_image": [{"image_b64": _page_b64()}], "path": ["a.pdf"]})
    out = nemotron_parse_pages(df, model=model, extract_text=True, length_retry_max_tokens=8192)
    meta = out.loc[0, "nemotron_parse_v1_2"]
    assert model.calls == [None, 8192]
    assert out.loc[0, "text"] == "Varied opening text"
    assert meta["error"] is None
    assert meta["completion"]["status"] == "incomplete"
    assert meta["length_retry"]["retry_failure"] == "RuntimeError: engine stopped"
    assert list(iter_stage_errors_from_value(meta)) == []
