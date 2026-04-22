# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nemo_retriever.graph.webhook_operator."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.webhook_operator import (
    WebhookNotifyOperator,
    _dataframe_to_records,
    _serialize_value,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_params(**overrides):
    defaults = {
        "endpoint_url": "https://hook.example.com/ingest",
        "columns": [],
        "headers": {},
        "timeout_s": 5.0,
        "max_retries": 1,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# _serialize_value
# ---------------------------------------------------------------------------


class TestSerializeValue:
    @pytest.mark.parametrize("val", [None, "hello", 42, 3.14, True])
    def test_primitives_pass_through(self, val):
        assert _serialize_value(val) is val

    def test_bytes_placeholder(self):
        assert _serialize_value(b"\x00\x01\x02") == "<bytes len=3>"

    def test_list_recursion(self):
        assert _serialize_value([1, b"x", {"a": 2}]) == [1, "<bytes len=1>", {"a": 2}]

    def test_dict_recursion(self):
        assert _serialize_value({1: b"ab"}) == {"1": "<bytes len=2>"}

    def test_arbitrary_object_str(self):
        assert _serialize_value(object).__contains__("object")


# ---------------------------------------------------------------------------
# _dataframe_to_records
# ---------------------------------------------------------------------------


class TestDataframeToRecords:
    def test_all_columns(self):
        df = pd.DataFrame({"a": [1], "b": ["x"]})
        records = _dataframe_to_records(df, columns=None)
        assert records == [{"a": 1, "b": "x"}]

    def test_column_subset(self):
        df = pd.DataFrame({"a": [1], "b": ["x"], "c": [3.0]})
        records = _dataframe_to_records(df, columns=["a", "c"])
        assert records == [{"a": 1, "c": 3.0}]

    def test_missing_columns_logged(self, caplog):
        df = pd.DataFrame({"a": [1]})
        with caplog.at_level("WARNING"):
            records = _dataframe_to_records(df, columns=["a", "missing"])
        assert len(records) == 1
        assert "missing" in caplog.text

    def test_all_columns_missing_returns_all(self, caplog):
        df = pd.DataFrame({"a": [1]})
        with caplog.at_level("WARNING"):
            records = _dataframe_to_records(df, columns=["nope"])
        assert records == [{"a": 1}]

    def test_empty_dataframe(self):
        df = pd.DataFrame({"a": pd.Series([], dtype="int64")})
        assert _dataframe_to_records(df, columns=None) == []


# ---------------------------------------------------------------------------
# WebhookNotifyOperator
# ---------------------------------------------------------------------------


class TestWebhookNotifyOperator:
    def test_inherits_abstract_operator(self):
        assert issubclass(WebhookNotifyOperator, AbstractOperator)

    def test_preprocess_passthrough(self):
        op = WebhookNotifyOperator()
        df = pd.DataFrame({"x": [1]})
        pd.testing.assert_frame_equal(op.preprocess(df), df)

    def test_postprocess_passthrough(self):
        op = WebhookNotifyOperator()
        df = pd.DataFrame({"x": [1]})
        pd.testing.assert_frame_equal(op.postprocess(df), df)

    # -- no-op paths --------------------------------------------------------

    def test_noop_when_no_params(self):
        op = WebhookNotifyOperator(params=None)
        df = pd.DataFrame({"x": [1]})
        result = op.process(df)
        pd.testing.assert_frame_equal(result, df)

    def test_noop_when_endpoint_url_is_none(self):
        op = WebhookNotifyOperator(params=_make_params(endpoint_url=None))
        df = pd.DataFrame({"x": [1]})
        result = op.process(df)
        pd.testing.assert_frame_equal(result, df)

    def test_noop_on_empty_batch(self):
        op = WebhookNotifyOperator(params=_make_params())
        df = pd.DataFrame({"x": pd.Series([], dtype="int64")})
        result = op.process(df)
        pd.testing.assert_frame_equal(result, df)

    # -- happy-path POST ----------------------------------------------------

    @patch("nemo_retriever.graph.webhook_operator.WebhookNotifyOperator._get_session")
    def test_happy_path_post(self, mock_get_session):
        mock_session = MagicMock()
        mock_response = MagicMock(status_code=200)
        mock_session.post.return_value = mock_response
        mock_get_session.return_value = mock_session

        params = _make_params()
        op = WebhookNotifyOperator(params=params)
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

        result = op.process(df)

        pd.testing.assert_frame_equal(result, df)
        mock_session.post.assert_called_once()
        call_kwargs = mock_session.post.call_args
        assert call_kwargs[1]["json"] == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        assert call_kwargs[0][0] == "https://hook.example.com/ingest"
        mock_response.raise_for_status.assert_called_once()

    @patch("nemo_retriever.graph.webhook_operator.WebhookNotifyOperator._get_session")
    def test_post_with_column_filter(self, mock_get_session):
        mock_session = MagicMock()
        mock_session.post.return_value = MagicMock(status_code=200)
        mock_get_session.return_value = mock_session

        params = _make_params(columns=["b"])
        op = WebhookNotifyOperator(params=params)
        df = pd.DataFrame({"a": [1], "b": ["x"]})

        op.process(df)

        payload = mock_session.post.call_args[1]["json"]
        assert payload == [{"b": "x"}]

    # -- error handling -----------------------------------------------------

    @patch("nemo_retriever.graph.webhook_operator.WebhookNotifyOperator._get_session")
    def test_post_failure_logs_and_passes_data_through(self, mock_get_session):
        mock_session = MagicMock()
        mock_session.post.side_effect = ConnectionError("boom")
        mock_get_session.return_value = mock_session

        params = _make_params()
        op = WebhookNotifyOperator(params=params)
        df = pd.DataFrame({"a": [1]})

        result = op.process(df)
        pd.testing.assert_frame_equal(result, df)

    # -- session reuse ------------------------------------------------------

    def test_session_created_once_and_reused(self):
        params = _make_params()
        op = WebhookNotifyOperator(params=params)

        s1 = op._get_session()
        s2 = op._get_session()
        assert s1 is s2

    def test_session_has_default_content_type(self):
        params = _make_params()
        op = WebhookNotifyOperator(params=params)
        session = op._get_session()
        assert session.headers.get("Content-Type") == "application/json"

    def test_session_applies_custom_headers(self):
        params = _make_params(headers={"X-Custom": "val"})
        op = WebhookNotifyOperator(params=params)
        session = op._get_session()
        assert session.headers.get("X-Custom") == "val"

    # -- retry configuration ------------------------------------------------

    def test_session_retry_adapter_mounted(self):
        params = _make_params(max_retries=5)
        op = WebhookNotifyOperator(params=params)
        session = op._get_session()

        https_adapter = session.get_adapter("https://example.com")
        assert https_adapter.max_retries.total == 5
        assert 500 in https_adapter.max_retries.status_forcelist
