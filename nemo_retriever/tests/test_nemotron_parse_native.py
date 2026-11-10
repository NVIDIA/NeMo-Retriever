# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Native Parse multipart wire contract and extraction failure handling."""

import base64
from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from nemo_retriever.models.nim import nim
from nemo_retriever.operators.extract.parse.nemotron_parse import nemotron_parse_pages


def response(payload, status=200):
    result = requests.Response()
    result.status_code = status
    result._content = b"error" if status >= 400 else b""
    result.json = lambda: payload
    return result


def test_multipart_upload_retries_with_identical_bytes_and_keeps_page_order(monkeypatch):
    uploads = []
    images = [b"\x89PNG\r\n\x1a\nfirst", b"\xff\xd8\xffsecond"]
    attempts = {}

    def post(url, *, headers, files, timeout):
        data = files["document"][1]
        uploads.append((url, files, headers, timeout))
        attempts[data] = attempts.get(data, 0) + 1
        if data == images[0] and attempts[data] == 1:
            return response(None, 503)
        return response({"content": "first" if data == images[0] else "second", "finish_reason": "stop"})

    monkeypatch.setattr(nim.requests, "post", post)
    monkeypatch.setattr(nim.time, "sleep", lambda _: None)
    client = nim.NIMClient(max_pool_workers=2)
    try:
        result = client.invoke_parse_pages(
            invoke_url="http://one/v1/parse,http://two/v1/parse",
            api_key="test-key",
            timeout_s=9,
            image_b64_list=[base64.b64encode(image).decode() for image in images],
            max_retries=2,
        )
    finally:
        client.shutdown()
    assert [item["content"] for item in result] == ["first", "second"]
    assert attempts == {images[0]: 2, images[1]: 1}
    for url, files, headers, timeout in uploads:
        assert headers["Authorization"] == "Bearer test-key"
        assert "Content-Type" not in headers  # requests supplies the multipart boundary.
        assert timeout == 9
        assert url == ("http://one/v1/parse" if files["document"][1] == images[0] else "http://two/v1/parse")
        assert files["document"][2] == ("image/png" if files["document"][1] == images[0] else "image/jpeg")


@pytest.mark.parametrize(
    "payload,recovery",
    [
        ({"text": "wrong contract"}, "Verify invoke_url targets.*native multipart /v1/parse"),
        ({"content": 17}, "Verify invoke_url targets.*native multipart /v1/parse"),
        ({"content": "partial", "finish_reason": "length"}, "Split this page into smaller image regions"),
    ],
)
def test_invalid_or_truncated_native_responses_fail(monkeypatch, payload, recovery):
    monkeypatch.setattr(nim.requests, "post", lambda *args, **kwargs: response(payload))
    client = nim.NIMClient()
    try:
        with pytest.raises(ValueError, match=recovery) as exc_info:
            client.invoke_parse_pages(invoke_url="http://nim/v1/parse", image_b64_list=["aW1hZ2U="])
        assert "page 1" in str(exc_info.value)
    finally:
        client.shutdown()


def test_truncated_native_response_is_returned_only_when_allowed(monkeypatch):
    payload = {"content": "partial", "finish_reason": "length", "usage": {"output_tokens": 9000}}
    monkeypatch.setattr(nim.requests, "post", lambda *args, **kwargs: response(payload))
    client = nim.NIMClient()
    try:
        result = client.invoke_parse_pages(
            invoke_url="http://nim/v1/parse", image_b64_list=["aW1hZ2U="], allow_incomplete=True
        )
    finally:
        client.shutdown()
    assert result == [payload]


def test_length_limited_native_page_keeps_complete_elements_and_is_reported(caplog):
    client = Mock()
    client.invoke_parse_pages.return_value = [
        {
            "content": "<x_0.1><y_0.2>Budget 42000<x_0.8><y_0.3><class_Text>"
            "<x_0.1><y_0.4><chart>| day | value |\n| 1 | 1000 |\n| 1 | 1000 |",
            "finish_reason": "length",
            "usage": {"output_tokens": 9000},
        },
        {"content": "<x_0.1><y_0.2>Next page<x_0.8><y_0.3><class_Text>", "finish_reason": "stop"},
    ]
    batch = pd.DataFrame(
        [
            {"path": "/docs/report.pdf", "page_number": 3, "text": "pdf text", "page_image": {"image_b64": "aW1hZ2U="}},
            {"path": "/docs/report.pdf", "page_number": 4, "text": "pdf text", "page_image": {"image_b64": "aW1hZ2U="}},
        ]
    )
    result = nemotron_parse_pages(
        batch, invoke_url="http://nim/v1/parse", nim_client=client, extract_text=True, extract_charts=True
    )
    assert client.invoke_parse_pages.call_args.kwargs["allow_incomplete"] is True
    assert result["text"].tolist() == ["Budget 42000", "Next page"]
    assert result.iloc[0]["chart"] == []  # The unterminated element is not routed.
    metadata = result["nemotron_parse_v1_2"].tolist()
    assert [item["error"] for item in metadata] == [None, None]
    assert metadata[0]["completion"] == {"status": "incomplete", "finish_reason": "length", "output_tokens": 9000}
    assert metadata[1]["completion"] == {"status": "complete", "finish_reason": "stop", "output_tokens": None}
    output = caplog.text
    assert (
        "Warning: Nemotron Parse page incomplete: path=/docs/report.pdf page=3 finish_reason=length "
        "output_tokens=9000"
    ) in output
    assert "page=4" not in output


@pytest.mark.parametrize("extract_text", [True, False])
def test_native_output_capped_before_first_element_finishes_is_incomplete(caplog, extract_text):
    client = Mock()
    client.invoke_parse_pages.return_value = [
        {
            "content": "<x_0.1><y_0.2>Unfinished table that never reaches its closing tags",
            "finish_reason": "length",
            "usage": {"output_tokens": 9000},
        }
    ]
    batch = pd.DataFrame(
        [{"path": "/docs/report.pdf", "page_number": 7, "text": "pdf text", "page_image": {"image_b64": "aW1hZ2U="}}]
    )
    result = nemotron_parse_pages(
        batch, invoke_url="http://nim/v1/parse", nim_client=client, extract_text=extract_text, extract_tables=True
    )
    metadata = result.iloc[0]["nemotron_parse_v1_2"]
    assert metadata["error"] is None
    assert metadata["completion"] == {"status": "incomplete", "finish_reason": "length", "output_tokens": 9000}
    assert result.iloc[0]["text"] == "pdf text"  # Raw tags replace neither the page text nor the output.
    assert result.iloc[0]["table"] == []
    assert "Warning: Nemotron Parse page incomplete: path=/docs/report.pdf page=7" in caplog.text


def test_native_markdown_capped_output_is_kept_as_page_text():
    client = Mock()
    client.invoke_parse_pages.return_value = [
        {"content": "Plain markdown cut off mid-sent", "finish_reason": "length", "usage": {"output_tokens": 9000}}
    ]
    batch = pd.DataFrame([{"path": "a.pdf", "page_number": 1, "page_image": {"image_b64": "aW1hZ2U="}}])
    result = nemotron_parse_pages(batch, invoke_url="http://nim/v1/parse", nim_client=client, extract_text=True)
    assert result.iloc[0]["text"] == "Plain markdown cut off mid-sent"
    assert result.iloc[0]["nemotron_parse_v1_2"]["completion"]["status"] == "incomplete"


def test_failed_native_batch_marks_every_page_failed(caplog):
    client = Mock()
    client.invoke_parse_pages.side_effect = TimeoutError("NIM unavailable")
    batch = pd.DataFrame([{"page_image": {"image_b64": "aW1hZ2U="}}, {"page_image": {"image_b64": "aW1hZ2U="}}])
    result = nemotron_parse_pages(batch, invoke_url="http://nim/v1/parse", nim_client=client, extract_text=True)
    for metadata in result["nemotron_parse_v1_2"]:
        assert metadata["error"]["type"] == "TimeoutError"
        assert metadata["completion"]["status"] == "failed"
    assert "Warning: Nemotron Parse batch failed: TimeoutError: NIM unavailable" in caplog.text


def test_native_client_does_not_retry_bad_requests(monkeypatch):
    post = Mock(return_value=response(None, 400))
    monkeypatch.setattr(nim.requests, "post", post)
    client = nim.NIMClient()
    try:
        with pytest.raises(requests.HTTPError):
            client.invoke_parse_pages(invoke_url="http://nim/v1/parse", image_b64_list=["aW1hZ2U="])
    finally:
        client.shutdown()
    assert post.call_count == 1


@pytest.mark.parametrize(
    "markdown",
    [
        "# Budget\n\n| Item | Cost |\n| --- | --- |\n| Equipment | 42000 |",
        "# Parse syntax\n\nUse `<class_Text>` to label text; `<class_` begins a class tag.",
        "# Incomplete example\n\n<x_0.1><y_0.2>Budget 42000<class_Text>",
    ],
)
def test_native_markdown_is_preserved_as_page_text_and_usage_is_recorded(markdown):
    client = Mock()
    client.invoke_parse_pages.return_value = [
        {"content": markdown, "finish_reason": "stop", "usage": {"output_tokens": 42}}
    ]
    batch = pd.DataFrame(
        [{"text": "stale", "page_image": {"image_b64": "aW1hZ2U="}}, {"text": "no image", "page_image": None}]
    )
    result = nemotron_parse_pages(
        batch, invoke_url="http://nim/v1/parse", nim_client=client, extract_text=True, extract_tables=True
    )
    assert result["text"].tolist() == [markdown, "no image"]
    assert result["table"].tolist() == [[], []]  # Native response has no element boxes.
    assert result.iloc[0]["nemotron_parse_v1_2"]["native_parse"]["usage"]["output_tokens"] == 42
    assert result.iloc[0]["nemotron_parse_v1_2"]["completion"] == {
        "status": "complete",
        "finish_reason": "stop",
        "output_tokens": 42,
    }
    assert result.iloc[0]["nemotron_parse_v1_2"]["error"] is None
    client.invoke_chat_completions_images.assert_not_called()
    client.invoke_image_inference_batches.assert_not_called()


def test_mixed_endpoint_contracts_produce_row_errors_before_inference():
    client = Mock()
    result = nemotron_parse_pages(
        pd.DataFrame([{"page_image": {"image_b64": "aW1hZ2U="}}]),
        invoke_url="http://nim/v1/parse,http://legacy/v1/chat/completions",
        nim_client=client,
        extract_text=True,
    )
    assert result.iloc[0]["nemotron_parse_v1_2"]["error"]["type"] == "ValueError"
    client.invoke_parse_pages.assert_not_called()


@pytest.mark.parametrize("extract_text", [True, False])
def test_native_tagged_content_routes_text_and_bounding_boxes(extract_text):
    client = Mock()
    client.invoke_parse_pages.return_value = [
        {
            "content": "<x_0.1><y_0.2>Budget 42000<x_0.8><y_0.3><class_Text>"
            "<x_0.1><y_0.4>Equipment 12000<x_0.8><y_0.7><class_Table>",
            "finish_reason": "stop",
        }
    ]
    result = nemotron_parse_pages(
        pd.DataFrame([{"page_image": {"image_b64": "aW1hZ2U="}}]),
        invoke_url="http://nim/v1/parse",
        nim_client=client,
        extract_text=extract_text,
        extract_tables=True,
    )
    assert result.iloc[0]["nemotron_parse_v1_2"]["error"] is None
    assert result.iloc[0]["table"] == [{"text": "Equipment 12000", "bbox_xyxy_norm": [0.1, 0.4, 0.8, 0.7]}]
    if extract_text:
        assert result.iloc[0]["text"] == "Budget 42000"


def test_untagged_native_content_without_text_extraction_is_not_silently_dropped():
    client = Mock()
    client.invoke_parse_pages.return_value = [{"content": "# Budget\n42000", "finish_reason": "stop"}]
    result = nemotron_parse_pages(
        pd.DataFrame([{"page_image": {"image_b64": "aW1hZ2U="}}]),
        invoke_url="http://nim/v1/parse",
        nim_client=client,
        extract_text=False,
        extract_tables=True,
    )
    assert result.iloc[0]["nemotron_parse_v1_2"]["error"]["type"] == "ValueError"
