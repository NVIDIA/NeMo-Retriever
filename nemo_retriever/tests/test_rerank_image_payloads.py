# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remote reranking image-format contracts without loading models."""

import base64
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image
import pytest
import requests

from nemo_retriever.models import NEMOTRON_35_RERANK_MODEL
from nemo_retriever.operators.rerank import (
    _nim_ranking_payload,
    _rerank_image_data_uri,
    _rerank_via_endpoint,
    rerank_hits,
)


def _encoded_image(image_format: str) -> str:
    buffer = io.BytesIO()
    with Image.new("RGB", (3, 2), "red") as image:
        image.save(buffer, format=image_format)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@pytest.mark.parametrize("image_format,mime_type", [("PNG", "image/png"), ("JPEG", "image/jpeg")])
def test_ranking_payload_wraps_raw_image_with_actual_mime_type(image_format: str, mime_type: str) -> None:
    encoded = _encoded_image(image_format)
    payload = _nim_ranking_payload("query", ["pictured", "text only"], NEMOTRON_35_RERANK_MODEL, [encoded, None])
    assert payload["passages"] == [
        {"text": "pictured", "image": f"data:{mime_type};base64,{encoded}"},
        {"text": "text only"},
    ]
    assert payload["truncate"] == "END"


def test_ranking_payload_preserves_existing_image_uri_without_decoding() -> None:
    image_uri = "data:image/jpeg;base64,already-encoded"
    with patch("PIL.Image.open", side_effect=AssertionError("Do not decode existing data URIs")):
        assert _rerank_image_data_uri(image_uri) == image_uri


@pytest.mark.parametrize("images", [None, [], [None], [""]])
def test_text_only_ranking_payload_stays_unchanged(images: list[str | None] | None) -> None:
    payload = _nim_ranking_payload("query", ["passage"], NEMOTRON_35_RERANK_MODEL, images)
    assert payload["passages"] == [{"text": "passage"}]


@pytest.mark.parametrize("encoded", ["not base64!", "bm90IGFuIGltYWdl", "non-ASCII: é"])
def test_invalid_image_fails_before_http_request(encoded: str) -> None:
    with patch("requests.post") as post:
        with pytest.raises(ValueError, match="Invalid reranking image:"):
            _rerank_via_endpoint("query", ["passage"], endpoint="http://reranker:8000", images_b64=[encoded])
    post.assert_not_called()


def test_image_payload_does_not_hide_programming_errors() -> None:
    with patch("PIL.Image.open", side_effect=TypeError("unexpected implementation error")):
        with pytest.raises(TypeError, match="unexpected implementation error"):
            _rerank_image_data_uri(_encoded_image("PNG"))


def test_remote_rerank_hits_formats_stored_images(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(base64.b64decode(_encoded_image("PNG")))
    response = MagicMock(spec=requests.Response)
    response.json.return_value = {"rankings": [{"index": 0, "logit": 0.9}]}
    with patch("requests.post", return_value=response) as post:
        hits = rerank_hits(
            "query",
            [{"text": "passage", "stored_image_uri": str(image_path)}],
            model_name=NEMOTRON_35_RERANK_MODEL,
            modality="text_image",
            rerank_invoke_url="http://reranker:8000",
        )
    assert post.call_args.kwargs["json"]["passages"][0]["image"].startswith("data:image/png;base64,")
    assert hits[0]["_rerank_score"] == 0.9
