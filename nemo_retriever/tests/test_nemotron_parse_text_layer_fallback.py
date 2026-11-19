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
from nemo_retriever.common.params.utils import build_parse_actor_kwargs
from nemo_retriever.operators.extract.parse.nemotron_parse import _text_layer_fallback, nemotron_parse_pages

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
