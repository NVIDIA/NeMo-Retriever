# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from nemo_retriever.common.inference_capture import (
    InferenceCaptureConfig,
    activate_inference_capture,
    record_binary_request,
    record_json_request,
)


def test_records_sanitized_replay_artifact(tmp_path: Path) -> None:
    with activate_inference_capture(InferenceCaptureConfig(str(tmp_path), failure_mode="required"), operation="query"):
        record_json_request(
            stage="rerank",
            endpoint="https://nim.example/v1/rerank?api_key=secret",
            payload={"query": "hello", "model": "test"},
            model="test",
        )

    capture_dir = next(tmp_path.iterdir())
    manifest = json.loads((capture_dir / "manifest.json").read_text())
    assert manifest["operation"] == "query"
    assert manifest["stage"] == "rerank"
    assert manifest["endpoint"] == "https://nim.example/v1/rerank"
    assert "secret" not in (capture_dir / "manifest.json").read_text()
    assert json.loads((capture_dir / "request.json").read_text()) == {"query": "hello", "model": "test"}


def test_best_effort_does_not_interrupt_inference(tmp_path: Path) -> None:
    target = tmp_path / "file-not-directory"
    target.write_text("not a directory")
    with activate_inference_capture(InferenceCaptureConfig(str(target)), operation="ingest"):
        record_json_request(stage="ocr", endpoint="http://nim/v1/ocr", payload={"image": "x"})


def test_required_capture_failure_raises(tmp_path: Path) -> None:
    target = tmp_path / "file-not-directory"
    target.write_text("not a directory")
    with activate_inference_capture(InferenceCaptureConfig(str(target), failure_mode="required"), operation="ingest"):
        with pytest.raises(RuntimeError, match="Failed to persist inference capture"):
            record_json_request(stage="ocr", endpoint="http://nim/v1/ocr", payload={"image": "x"})


def test_records_binary_transport_artifact(tmp_path: Path) -> None:
    with activate_inference_capture(InferenceCaptureConfig(str(tmp_path), failure_mode="required"), operation="ingest"):
        record_binary_request(
            stage="asr", endpoint="asr.example:50051", payload=b"grpc-input", protocol="grpc",
            model="asr-model", metadata={"input_names": ["audio"]},
        )
    capture_dir = next(tmp_path.iterdir())
    manifest = json.loads((capture_dir / "manifest.json").read_text())
    assert manifest["protocol"] == "grpc"
    assert manifest["metadata"]["input_names"] == ["audio"]
    assert (capture_dir / "request.bin").read_bytes() == b"grpc-input"


def test_environment_config_captures_query_requests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEMO_RETRIEVER_INFERENCE_CAPTURE_URI", str(tmp_path))
    monkeypatch.setenv("NEMO_RETRIEVER_INFERENCE_CAPTURE_FAILURE_MODE", "required")
    monkeypatch.setenv("NEMO_RETRIEVER_INFERENCE_CAPTURE_OPERATION", "query")

    record_json_request(stage="embed", endpoint="http://nim/v1/embeddings", payload={"input": ["q"]})

    capture_dir = next(tmp_path.iterdir())
    manifest = json.loads((capture_dir / "manifest.json").read_text())
    assert manifest["operation"] == "query"
    assert manifest["stage"] == "embed"
