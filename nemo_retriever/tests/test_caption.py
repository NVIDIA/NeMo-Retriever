# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for image captioning pipeline stage."""

import base64
import io
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def _make_test_png_b64(size: tuple[int, int] = (64, 64)) -> str:
    img = Image.new("RGB", size, color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _make_page_df(num_images=2, captioned=False):
    b64 = _make_test_png_b64()
    images = [
        {"bbox_xyxy_norm": [0.1, 0.2, 0.5, 0.8], "text": "done" if captioned else "", "image_b64": b64}
        for _ in range(num_images)
    ]
    return pd.DataFrame([{"text": "page", "images": images, "tables": [], "charts": [], "infographics": []}])


def test_caption_images_writes_back():
    from nemo_retriever.caption.caption import caption_images

    mock_model = MagicMock()
    mock_model.caption_batch.return_value = ["cap1", "cap2"]
    result = caption_images(_make_page_df(), model=mock_model)
    assert result.iloc[0]["images"][0]["text"] == "cap1"
    assert result.iloc[0]["images"][1]["text"] == "cap2"


def test_caption_images_skips_already_captioned():
    from nemo_retriever.caption.caption import caption_images

    mock_model = MagicMock()
    result = caption_images(_make_page_df(captioned=True), model=mock_model)
    mock_model.caption_batch.assert_not_called()
    assert result.iloc[0]["images"][0]["text"] == "done"


@patch("nemo_retriever.pdf.extract.extract_image_like_objects_from_pdfium_page")
def test_pdf_extraction_populates_images(mock_extract):
    _ext = pytest.importorskip("nemo_retriever.pdf.extract")
    pdfium = pytest.importorskip("pypdfium2")

    mock_img = MagicMock(image=_make_test_png_b64(), bbox=(10, 20, 100, 200), max_width=612, max_height=792)
    mock_extract.return_value = [mock_img]

    doc = pdfium.PdfDocument.new()
    doc.new_page(612, 792)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()

    result = _ext.pdf_extraction(
        pd.DataFrame([{"bytes": buf.getvalue(), "path": "t.pdf", "page_number": 1}]), extract_images=True
    )
    images = result.iloc[0]["images"]
    assert len(images) == 1
    assert images[0]["text"] == ""
    assert abs(images[0]["bbox_xyxy_norm"][0] - 10 / 612) < 1e-6


def test_explode_includes_captioned_images():
    from nemo_retriever.graph.content_transforms import explode_content_to_rows

    b64 = _make_test_png_b64()
    df = pd.DataFrame(
        [
            {
                "text": "page",
                "page_image": {"image_b64": b64},
                "images": [{"text": "a dog", "bbox_xyxy_norm": [0.1, 0.2, 0.5, 0.8], "image_b64": b64}],
                "tables": [],
                "charts": [],
                "infographics": [],
            }
        ]
    )
    result = explode_content_to_rows(df, content_columns=("table", "chart", "infographic", "images"))
    assert len(result) == 2  # page text + image caption

    # Default columns exclude images
    result2 = explode_content_to_rows(df)
    assert len(result2) == 1


def test_context_text_prepended_to_prompt():
    from nemo_retriever.caption.caption import caption_images

    mock_model = MagicMock()
    mock_model.caption_batch.return_value = ["captioned with context"]

    df = _make_page_df(num_images=1)
    df.at[0, "text"] = "The quick brown fox jumps over the lazy dog."

    result = caption_images(df, model=mock_model, context_text_max_chars=100)

    assert result.iloc[0]["images"][0]["text"] == "captioned with context"
    # The prompt passed to caption_batch should contain the page text.
    call_kwargs = mock_model.caption_batch.call_args[1]
    assert "quick brown fox" in call_kwargs["prompt"]
    assert "Text near this image:" in call_kwargs["prompt"]


def test_caption_images_skips_small_images():
    from nemo_retriever.caption.caption import caption_images

    tiny_b64 = _make_test_png_b64(size=(1, 1))
    images = [{"bbox_xyxy_norm": [0.1, 0.2, 0.5, 0.8], "text": "", "image_b64": tiny_b64}]
    df = pd.DataFrame([{"text": "page", "images": images, "tables": [], "charts": [], "infographics": []}])

    mock_model = MagicMock()
    result = caption_images(df, model=mock_model)
    mock_model.caption_batch.assert_not_called()
    assert result.iloc[0]["images"][0]["text"] == ""


# ---------------------------------------------------------------------------
# LLMInferenceParams & CaptionParams tests
# ---------------------------------------------------------------------------


class TestLLMInferenceParamsValidation:
    """Validate LLMInferenceParams range checks and defaults."""

    def test_defaults(self):
        from nemo_retriever.params.models import LLMInferenceParams

        p = LLMInferenceParams()
        assert p.temperature == 1.0
        assert p.top_p is None
        assert p.max_tokens == 1024

    def test_negative_temperature_rejected(self):
        from nemo_retriever.params.models import LLMInferenceParams

        with pytest.raises(ValueError, match="temperature must be >= 0"):
            LLMInferenceParams(temperature=-0.1)

    def test_top_p_out_of_range_rejected(self):
        from nemo_retriever.params.models import LLMInferenceParams

        with pytest.raises(ValueError, match="top_p must be between"):
            LLMInferenceParams(top_p=1.5)
        with pytest.raises(ValueError, match="top_p must be between"):
            LLMInferenceParams(top_p=-0.1)

    def test_zero_max_tokens_rejected(self):
        from nemo_retriever.params.models import LLMInferenceParams

        with pytest.raises(ValueError, match="max_tokens must be > 0"):
            LLMInferenceParams(max_tokens=0)

    def test_valid_top_p_accepted(self):
        from nemo_retriever.params.models import LLMInferenceParams

        p = LLMInferenceParams(top_p=0.95)
        assert p.top_p == 0.95

    def test_to_sampling_kwargs_includes_top_p_when_set(self):
        from nemo_retriever.params.models import LLMInferenceParams

        kw = LLMInferenceParams(temperature=0.3, top_p=0.9, max_tokens=512).to_sampling_kwargs()
        assert kw == {"temperature": 0.3, "top_p": 0.9, "max_tokens": 512}

    def test_to_sampling_kwargs_omits_top_p_when_none(self):
        from nemo_retriever.params.models import LLMInferenceParams

        kw = LLMInferenceParams(temperature=0.5).to_sampling_kwargs()
        assert kw == {"temperature": 0.5, "max_tokens": 1024}
        assert "top_p" not in kw


class TestCaptionParamsInheritance:
    """Verify CaptionParams inherits LLM fields and model_dump stays flat."""

    def test_inherits_llm_fields(self):
        from nemo_retriever.params import CaptionParams

        p = CaptionParams()
        assert p.temperature == 1.0
        assert p.top_p is None
        assert p.max_tokens == 1024

    def test_model_dump_is_flat(self):
        from nemo_retriever.params import CaptionParams

        d = CaptionParams(temperature=0.7, top_p=0.9, max_tokens=512).model_dump()
        assert d["temperature"] == 0.7
        assert d["top_p"] == 0.9
        assert d["max_tokens"] == 512
        assert "llm" not in d

    def test_backward_compat_temperature_only(self):
        from nemo_retriever.params import CaptionParams

        p = CaptionParams(temperature=0.5)
        assert p.temperature == 0.5
        assert p.top_p is None
        assert p.max_tokens == 1024

    def test_validation_inherited(self):
        from nemo_retriever.params import CaptionParams

        with pytest.raises(ValueError, match="temperature must be >= 0"):
            CaptionParams(temperature=-1.0)


class TestCaptionImageParamThreading:
    """Verify top_p and max_tokens flow through to the model / client."""

    def test_top_p_forwarded_to_local_model(self):
        from nemo_retriever.caption.caption import caption_images

        mock_model = MagicMock()
        mock_model.caption_batch.return_value = ["cap"]
        caption_images(_make_page_df(num_images=1), model=mock_model, top_p=0.9)

        call_kwargs = mock_model.caption_batch.call_args[1]
        assert call_kwargs["top_p"] == 0.9

    def test_max_tokens_forwarded_to_local_model(self):
        from nemo_retriever.caption.caption import caption_images

        mock_model = MagicMock()
        mock_model.caption_batch.return_value = ["cap"]
        caption_images(_make_page_df(num_images=1), model=mock_model, max_tokens=512)

        call_kwargs = mock_model.caption_batch.call_args[1]
        assert call_kwargs["max_tokens"] == 512

    def test_defaults_forwarded_when_omitted(self):
        from nemo_retriever.caption.caption import caption_images

        mock_model = MagicMock()
        mock_model.caption_batch.return_value = ["cap"]
        caption_images(_make_page_df(num_images=1), model=mock_model)

        call_kwargs = mock_model.caption_batch.call_args[1]
        assert call_kwargs["top_p"] is None
        assert call_kwargs["max_tokens"] == 1024

    @patch("nemo_retriever.caption.caption._create_remote_client")
    @patch(
        "nemo_retriever.model.local.nemotron_vlm_captioner.resolve_caption_model_name", side_effect=lambda n, target: n
    )
    def test_top_p_and_max_tokens_forwarded_to_remote(self, _mock_resolve, mock_create_client):
        from nemo_retriever.caption.caption import caption_images

        mock_nim = MagicMock()
        mock_nim.infer.return_value = ["remote cap"]
        mock_create_client.return_value = mock_nim

        caption_images(
            _make_page_df(num_images=1),
            endpoint_url="http://fake-vlm/v1",
            top_p=0.85,
            max_tokens=256,
        )

        infer_kwargs = mock_nim.infer.call_args[1]
        assert infer_kwargs["top_p"] == 0.85
        assert infer_kwargs["max_tokens"] == 256
        assert infer_kwargs["temperature"] == 1.0

    @patch("nemo_retriever.caption.caption._create_remote_client")
    @patch(
        "nemo_retriever.model.local.nemotron_vlm_captioner.resolve_caption_model_name", side_effect=lambda n, target: n
    )
    def test_remote_omits_top_p_when_none(self, _mock_resolve, mock_create_client):
        from nemo_retriever.caption.caption import caption_images

        mock_nim = MagicMock()
        mock_nim.infer.return_value = ["remote cap"]
        mock_create_client.return_value = mock_nim

        caption_images(
            _make_page_df(num_images=1),
            endpoint_url="http://fake-vlm/v1",
        )

        infer_kwargs = mock_nim.infer.call_args[1]
        assert "top_p" not in infer_kwargs
        assert infer_kwargs["max_tokens"] == 1024
