# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for multimodal embedding helpers and explode_content_to_rows.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from nemo_retriever.models.embed_errors import LocalEmbedderRowsLostError

# ---------------------------------------------------------------------------
# Pure helpers from main_text_embed (no transitive-import issues)
# ---------------------------------------------------------------------------
from nemo_retriever.models.inference.main_text_embed import (
    TextEmbeddingConfig,
    _format_image_input_string,
    _format_text_image_pair_input_string,
    _image_from_row,
    _multimodal_callable_runner,
    create_text_embeddings_for_df,
)

# ---------------------------------------------------------------------------
# Stub heavy internal modules so the content-transform helpers can be imported
# in lightweight CI (only pytest, pandas, pydantic, pyyaml).
#
# Older ingest modules can pull in ray, torch, nemotron_*, nemo_retriever.common.api,
# etc. And inprocess.py itself imports model/local (torch, nemotron_*),
# page_elements, ocr, and pdf.extract — each with their own heavy transitive
# deps.
#
# Rather than chasing every third-party leaf dependency, we pre-populate
# sys.modules for the heavy *internal* nemo_retriever sub-packages with MagicMock.
# This cuts off the entire transitive tree at the root.
# ---------------------------------------------------------------------------
_HEAVY_INTERNAL = [
    # -- sibling ingest modes (prevents batch.py from loading) ------------------
    "nemo_retriever.ingest_modes.batch",
    # -- model / ML packages (torch, nemotron_*, transformers) ---------------
    "nemo_retriever.models.local",
    "nemo_retriever.models.local.llama_nemotron_embed_1b_v2_embedder",
    "nemo_retriever.models.local.nemotron_page_elements_v3",
    "nemo_retriever.models.local.nemotron_ocr_v1",
    "nemo_retriever.models.local.nemotron_table_structure_v1",
    # -- detection / OCR (nemotron_page_elements_v3, PIL, requests) ----------
    "nemo_retriever.page_elements",
    "nemo_retriever.operators.extract.page_elements.page_elements",
    "nemo_retriever.ocr",
    "nemo_retriever.operators.extract.ocr.ocr",
    # -- table (nemo_retriever.common.api → cv2) ----------------------------------------
    "nemo_retriever.table",
    "nemo_retriever.operators.extract.table.table_detection",
    "nemo_retriever.table.stage",
    # -- PDF (pypdfium2 and heavy extraction dependencies) -------------------
    "nemo_retriever.pdf",
    "nemo_retriever.operators.extract.pdf.extract",
    "nemo_retriever.operators.extract.pdf.split",
]
# Track which modules we injected (vs. ones already loaded) so we can
# remove only our stubs after the import, preventing leaks into other
# test files that need the real modules.
_injected: list[str] = []
for _mod_name in _HEAVY_INTERNAL:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()
        _injected.append(_mod_name)

from nemo_retriever.common.modality.content_transforms import (
    collapse_content_to_page_rows,
    explode_content_to_rows,
)  # noqa: E402

# Clean up injected mocks so they don't poison imports in other test files.
for _mod_name in _injected:
    sys.modules.pop(_mod_name, None)
del _injected


# ===================================================================
# Pure helpers
# ===================================================================


class TestImageFromRow:
    def test_returns_b64_when_present(self):
        row = pd.Series({"_image_b64": "abc123"})
        assert _image_from_row(row) == "abc123"

    @pytest.mark.parametrize("value", [None, "", "   ", 42])
    def test_returns_none_for_missing_empty_whitespace(self, value):
        data = {"_image_b64": value} if value is not None else {}
        row = pd.Series(data)
        assert _image_from_row(row) is None


class TestFormatInputStrings:
    def test_format_image_input_string(self):
        result = _format_image_input_string("AAAA")
        assert result == "data:image/png;base64,AAAA"

    def test_format_image_input_string_custom_mime(self):
        result = _format_image_input_string("BBBB", mime="image/jpeg")
        assert result == "data:image/jpeg;base64,BBBB"

    def test_format_text_image_pair_input_string(self):
        result = _format_text_image_pair_input_string("hello world", "CCCC")
        assert result == "hello world\ndata:image/png;base64,CCCC"


# ===================================================================
# _multimodal_callable_runner
# ===================================================================


class TestMultimodalCallableRunner:
    def test_image_mode(self):
        """Image-only mode calls embedder.embed_images() and returns embeddings."""
        embedder = MagicMock()
        embedder.embed_images.return_value = [[0.1, 0.2], [0.3, 0.4]]

        df = pd.DataFrame(
            {
                "text": ["page one", "page two"],
                "_image_b64": ["img1_b64", "img2_b64"],
            }
        )

        result = _multimodal_callable_runner(
            df,
            embedder=embedder,
            batch_size=64,
            embed_modality="image",
        )

        embedder.embed_images.assert_called_once()
        assert result["embeddings"] == [[0.1, 0.2], [0.3, 0.4]]
        assert len(result["info_msgs"]) == 2

    def test_text_image_fallback(self):
        """text_image mode: rows with images use embed_text_image(), rows without fall back to embed()."""
        embedder = MagicMock()
        # Row 0 has image -> embed_text_image
        # Row 1 has no image -> embed (text-only fallback)
        embedder.embed_text_image.return_value = [[1.0, 2.0]]
        embedder.embed.return_value = [[3.0, 4.0]]

        df = pd.DataFrame(
            {
                "text": ["with image", "text only"],
                "_image_b64": ["imgB64", ""],
            }
        )

        result = _multimodal_callable_runner(
            df,
            embedder=embedder,
            batch_size=64,
            embed_modality="text_image",
        )

        embedder.embed_text_image.assert_called_once()
        embedder.embed.assert_called_once()
        # Order must be preserved: row 0 (multimodal), row 1 (text fallback)
        assert result["embeddings"] == [[1.0, 2.0], [3.0, 4.0]]
        assert len(result["info_msgs"]) == 2


# ===================================================================
# explode_content_to_rows
# ===================================================================


class TestExplodeContentToRows:
    def test_text_mode_tags_modality(self):
        """Default text mode tags every row with _embed_modality='text' and no _image_b64."""
        df = pd.DataFrame(
            {
                "text": ["Hello world"],
                "table": [[{"text": "cell data"}]],
            }
        )

        result = explode_content_to_rows(df)

        assert "_embed_modality" in result.columns
        assert list(result["_embed_modality"]) == ["text", "text"]
        assert "_image_b64" not in result.columns

    def test_arrow_backed_structured_arrays_expand_into_element_rows(self):
        """Ray Arrow-backed list cells expand like their Python-list equivalents."""
        df = pd.DataFrame(
            {
                "text": ["page text"],
                "table": [np.array([{"text": "table text"}], dtype=object)],
                "chart": [np.array([{"text": "chart text"}], dtype=object)],
            }
        )

        result = explode_content_to_rows(df)

        assert result["text"].tolist() == ["page text", "table text", "chart text"]
        assert result["_content_type"].tolist() == ["text", "table", "chart"]
        assert result.iloc[0]["table"] is not result.iloc[1]["table"]

    @pytest.mark.parametrize("value", [np.array(1, dtype=object), np.ones((2, 2))])
    def test_non_collection_arrays_are_not_expanded(self, value):
        result = explode_content_to_rows(pd.DataFrame({"text": ["page text"], "table": [value]}))

        assert result["text"].tolist() == ["page text"]

    @patch("nemo_retriever.common.modality.content_transforms._crop_b64_image_by_norm_bbox")
    def test_text_image_carries_image(self, mock_crop):
        """text_image mode copies page image to _image_b64, crops for structured content."""
        mock_crop.return_value = ("cropped_b64", None)

        df = pd.DataFrame(
            {
                "text": ["some page text"],
                "page_image": [{"image_b64": "full_page_b64"}],
                "table": [[{"text": "table cell", "bbox_xyxy_norm": [0.1, 0.2, 0.9, 0.8]}]],
            }
        )

        result = explode_content_to_rows(df, modality="text_image")

        assert "_image_b64" in result.columns
        images = list(result["_image_b64"])
        modalities = list(result["_embed_modality"])

        # Row 0: page text row gets full page image
        assert images[0] == "full_page_b64"
        assert modalities[0] == "text_image"

        # Row 1: structured content row gets cropped image
        assert images[1] == "cropped_b64"
        assert modalities[1] == "text_image"

        mock_crop.assert_called_once_with(
            "full_page_b64",
            bbox_xyxy_norm=[0.1, 0.2, 0.9, 0.8],
        )


# ===================================================================
# collapse_content_to_page_rows
# ===================================================================


class TestCollapseContentToPageRows:
    def test_text_concatenation(self):
        """Page text + table + chart text are concatenated into one string per page."""
        df = pd.DataFrame(
            {
                "text": ["Hello world"],
                "table": [[{"text": "table data"}]],
                "chart": [[{"text": "chart data"}]],
                "infographic": [[]],
            }
        )

        result = collapse_content_to_page_rows(df)

        assert len(result) == 1
        assert result["text"].iloc[0] == "Hello world\n\ntable data\n\nchart data"
        assert result["_embed_modality"].iloc[0] == "text"

    def test_arrow_backed_structured_arrays_are_collapsed_into_page_text(self):
        """Ray Arrow-backed list cells contribute their text to the page row."""
        df = pd.DataFrame(
            {
                "text": ["page text"],
                "table": [np.array([{"text": "table text"}], dtype=object)],
                "chart": [np.array([{"text": "chart text"}], dtype=object)],
            }
        )

        result = collapse_content_to_page_rows(df)

        assert result["text"].tolist() == ["page text\n\ntable text\n\nchart text"]

    def test_full_page_image_used(self):
        """In image modalities, _image_b64 is the full page image (no cropping)."""
        df = pd.DataFrame(
            {
                "text": ["some text"],
                "page_image": [{"image_b64": "full_page_b64"}],
                "table": [[{"text": "table cell", "bbox_xyxy_norm": [0.1, 0.2, 0.9, 0.8]}]],
            }
        )

        result = collapse_content_to_page_rows(df, modality="text_image")

        assert len(result) == 1
        assert result["_image_b64"].iloc[0] == "full_page_b64"
        assert result["_embed_modality"].iloc[0] == "text_image"

    def test_multiple_pages_produce_one_row_each(self):
        """Each page produces exactly one row in the output."""
        df = pd.DataFrame(
            {
                "text": ["page 1 text", "page 2 text"],
                "table": [[{"text": "t1"}], [{"text": "t2"}]],
            }
        )

        result = collapse_content_to_page_rows(df)

        assert len(result) == 2
        assert "t1" in result["text"].iloc[0]
        assert "t2" in result["text"].iloc[1]

    def test_empty_content_handled(self):
        """Pages with no text and no structured content produce an empty string."""
        df = pd.DataFrame(
            {
                "text": ["", None],
                "table": [[], None],
            }
        )

        result = collapse_content_to_page_rows(df)

        assert len(result) == 2
        assert result["text"].iloc[0] == ""
        assert result["text"].iloc[1] == ""

    def test_image_modality_without_page_image_column(self):
        """When page_image column is missing, _image_b64 is set to None."""
        df = pd.DataFrame(
            {
                "text": ["some text"],
                "table": [[{"text": "data"}]],
            }
        )

        result = collapse_content_to_page_rows(df, modality="image")

        assert len(result) == 1
        assert result["_image_b64"].iloc[0] is None

    def test_empty_dataframe_passthrough(self):
        """Empty DataFrame is returned as-is."""
        df = pd.DataFrame()
        result = collapse_content_to_page_rows(df)
        assert result.empty

    def test_non_dataframe_passthrough(self):
        """Non-DataFrame input is returned as-is."""
        result = collapse_content_to_page_rows(None)
        assert result is None


class _StubVLEmbedder:
    """VL embedder stub whose per-call return length is scripted by the test."""

    def __init__(self, *, images=None, text_image=None, text=None):
        self._images = images
        self._text_image = text_image
        self._text = text

    def embed_images(self, images_b64, *, batch_size=64):
        return self._images

    def embed_text_image(self, texts, images_b64, *, batch_size=64):
        return self._text_image

    def embed(self, texts, *, batch_size=64):
        return self._text


def _image_frame(image_values):
    return pd.DataFrame({"_image_b64": list(image_values), "text": [""] * len(image_values)})


def _run_multimodal(df, *, embedder, embed_modality):
    config = TextEmbeddingConfig(embed_modality=embed_modality, output_payload_column="embedding_result")
    out, _ = create_text_embeddings_for_df(
        df,
        task_config={"endpoint_url": None, "multimodal_embedder": embedder, "local_batch_size": 8},
        transform_config=config,
    )
    return [payload["embedding"] for payload in out["embedding_result"]]


@pytest.mark.parametrize(
    ("images", "returned", "lost", "total"),
    [
        (["b64-a", "b64-b", "b64-c"], [[0.1, 0.2]], 2, 3),
        (["b64-a", "b64-b"], [], 2, 2),
    ],
    ids=["partial-answer", "empty-answer"],
)
def test_image_mode_short_answer_is_fatal(images, returned, lost, total):
    embedder = _StubVLEmbedder(images=returned)

    with pytest.raises(LocalEmbedderRowsLostError) as excinfo:
        _run_multimodal(_image_frame(images), embedder=embedder, embed_modality="image")
    assert excinfo.value.lost == lost
    assert excinfo.value.total == total
    assert "pad or drop those rows" in str(excinfo.value)


def test_text_image_mode_short_multimodal_answer_is_fatal():
    df = pd.DataFrame({"_image_b64": ["b64-a", "b64-b"], "text": ["alpha", "beta"]})
    embedder = _StubVLEmbedder(text_image=[[0.1, 0.2]])

    with pytest.raises(LocalEmbedderRowsLostError):
        _run_multimodal(df, embedder=embedder, embed_modality="text_image")


def test_text_image_mode_short_text_fallback_answer_is_fatal():
    df = pd.DataFrame({"_image_b64": ["", ""], "text": ["alpha", "beta"]})
    embedder = _StubVLEmbedder(text=[[0.1, 0.2]])

    with pytest.raises(LocalEmbedderRowsLostError):
        _run_multimodal(df, embedder=embedder, embed_modality="text_image")


@pytest.mark.parametrize(
    ("df", "embed_modality", "embedder"),
    [
        pytest.param(_image_frame(["b64-a"]), "image", _StubVLEmbedder(images=[[0.1], [0.2]]), id="image"),
        pytest.param(
            pd.DataFrame({"_image_b64": ["b64-a"], "text": ["alpha"]}),
            "text_image",
            _StubVLEmbedder(text_image=[[0.1], [0.2]]),
            id="text-image",
        ),
        pytest.param(
            pd.DataFrame({"_image_b64": [""], "text": ["alpha"]}),
            "text_image",
            _StubVLEmbedder(text=[[0.1], [0.2]]),
            id="text-fallback",
        ),
    ],
)
def test_multimodal_extra_answer_reports_the_cardinality(df, embed_modality, embedder):
    with pytest.raises(ValueError, match=r"returned 2 vectors for 1 submitted input"):
        _run_multimodal(df, embedder=embedder, embed_modality=embed_modality)


def test_image_mode_rows_without_images_do_not_fire_the_guard():
    df = _image_frame(["b64-a", "", "b64-c"])
    embedder = _StubVLEmbedder(images=[[0.1, 0.2], [0.3, 0.4]])

    assert _run_multimodal(df, embedder=embedder, embed_modality="image") == [[0.1, 0.2], None, [0.3, 0.4]]


def test_image_mode_chunk_with_no_images_at_all_does_not_fire_the_guard():
    df = _image_frame(["", ""])
    embedder = _StubVLEmbedder(images=[])

    assert _run_multimodal(df, embedder=embedder, embed_modality="image") == [None, None]


def test_text_image_mode_mixed_rows_do_not_fire_the_guard():
    df = pd.DataFrame({"_image_b64": ["b64-a", "", ""], "text": ["alpha", "beta", "   "]})
    embedder = _StubVLEmbedder(text_image=[[0.1, 0.2]], text=[[0.3, 0.4]])

    assert _run_multimodal(df, embedder=embedder, embed_modality="text_image") == [
        [0.1, 0.2],
        [0.3, 0.4],
        None,
    ]
