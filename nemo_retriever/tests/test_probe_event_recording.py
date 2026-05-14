# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NIM probe result collection, error reporter, and event-log recording."""

from unittest.mock import patch, MagicMock

import requests

from nemo_retriever.nim.error_reporter import (
    _errors,
    drain_errors,
    report_error,
)
from nemo_retriever.nim.probe import (
    _probe_results,
    drain_probe_results,
    probe_endpoint,
)


class TestProbeResultCollection:
    """probe_endpoint() collects ProbeResults for later persistence."""

    def setup_method(self):
        _probe_results.clear()

    def teardown_method(self):
        _probe_results.clear()

    def test_connection_refused_appends_unreachable(self):
        with patch("nemo_retriever.nim.probe.requests.get", side_effect=requests.ConnectionError("refused")):
            probe_endpoint("http://localhost:8009/v1/invoke", name="ocr", prefix="TestActor")

        results = drain_probe_results()
        assert len(results) == 1
        assert results[0].status == "unreachable"
        assert results[0].name == "ocr"
        assert results[0].prefix == "TestActor"
        assert "UNREACHABLE" in results[0].detail

    def test_timeout_appends_timeout(self):
        with patch("nemo_retriever.nim.probe.requests.get", side_effect=requests.Timeout("timed out")):
            probe_endpoint("http://localhost:8009/v1/invoke", name="ocr", prefix="TestActor")

        results = drain_probe_results()
        assert len(results) == 1
        assert results[0].status == "timeout"
        assert results[0].name == "ocr"
        assert "timed out" in results[0].detail

    def test_healthy_endpoint_appends_ok(self):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        with patch("nemo_retriever.nim.probe.requests.get", return_value=mock_resp):
            probe_endpoint("http://localhost:8009/v1/invoke", name="ocr", prefix="TestActor")

        results = drain_probe_results()
        assert len(results) == 1
        assert results[0].status == "ok"

    def test_drain_clears_results(self):
        with patch("nemo_retriever.nim.probe.requests.get", side_effect=requests.ConnectionError()):
            probe_endpoint("http://localhost:8009/v1/invoke", name="ocr", prefix="TestActor")

        first = drain_probe_results()
        second = drain_probe_results()
        assert len(first) == 1
        assert len(second) == 0

    def test_multiple_probes_accumulate(self):
        with patch("nemo_retriever.nim.probe.requests.get", side_effect=requests.ConnectionError()):
            probe_endpoint("http://localhost:8009/v1/invoke", name="ocr", prefix="Actor1")
            probe_endpoint("http://localhost:8006/v1/invoke", name="table-structure", prefix="Actor2")

        results = drain_probe_results()
        assert len(results) == 2
        assert results[0].name == "ocr"
        assert results[1].name == "table-structure"

    def test_step2_connection_refused_appends_unreachable(self):
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 404
        with patch("nemo_retriever.nim.probe.requests.get", return_value=mock_resp):
            with patch("nemo_retriever.nim.probe.requests.post", side_effect=requests.ConnectionError()):
                probe_endpoint(
                    "http://localhost:8009/v1/invoke",
                    name="ocr",
                    prefix="TestActor",
                    api_key="test-key",
                )

        results = drain_probe_results()
        assert len(results) == 1
        assert results[0].status == "unreachable"
        assert "localhost:8009" in results[0].url


class TestOperatorErrorReporter:
    """report_error() collects OperatorErrors for later persistence."""

    def setup_method(self):
        _errors.clear()

    def teardown_method(self):
        _errors.clear()

    def test_report_error_collects(self):
        exc = RuntimeError("NIM returned 500")
        report_error("graphic_elements_ocr_page_elements:ocr", exc, row_index=3)

        results = drain_errors()
        assert len(results) == 1
        assert results[0].stage == "graphic_elements_ocr_page_elements:ocr"
        assert results[0].exc_type == "RuntimeError"
        assert results[0].message == "NIM returned 500"
        assert results[0].row_index == 3
        assert "RuntimeError" in results[0].traceback

    def test_drain_clears(self):
        report_error("embed", ValueError("bad input"))
        first = drain_errors()
        second = drain_errors()
        assert len(first) == 1
        assert len(second) == 0

    def test_multiple_errors_accumulate(self):
        report_error("pdf_split", RuntimeError("corrupt"))
        report_error("embed", TimeoutError("took too long"))
        report_error("table_structure_ocr_page_elements:crop", MemoryError("oom"))

        results = drain_errors()
        assert len(results) == 3
        assert results[0].stage == "pdf_split"
        assert results[1].stage == "embed"
        assert results[2].stage == "table_structure_ocr_page_elements:crop"

    def test_row_index_defaults_to_none(self):
        report_error("embed", RuntimeError("fail"))
        results = drain_errors()
        assert results[0].row_index is None
