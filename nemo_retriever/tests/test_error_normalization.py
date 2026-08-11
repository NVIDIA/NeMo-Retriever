"""Regression coverage for safe public error normalization."""

from __future__ import annotations

from fastapi.testclient import TestClient
import requests

from nemo_retriever.common.error_normalization import normalize_error
from nemo_retriever.ingestor.graph_ingestor import GraphIngestionError, _StageDiagnostic
from nemo_retriever.service.app import create_app
from nemo_retriever.service.auth import AuthConfig
from nemo_retriever.service.config import ServiceConfig
from nemo_retriever.service.services.job_tracker import get_job_tracker
from nemo_retriever.service.services.metrics import get_metrics


def _http_422() -> requests.HTTPError:
    response = requests.Response()
    response.status_code = 422
    response.url = "http://ocr.local/nemotron-ocr-v2?token=never-show-this"
    response._content = b'{"detail":"invalid image data"}'
    return requests.HTTPError("422 Client Error", response=response)


def _ocr_error() -> GraphIngestionError:
    return GraphIngestionError(
        [{
            "row_index": 0,
            "column": "ocr",
            "path": "error",
            "error": _http_422(),
        }],
        stage_diagnostics={"ocr": _StageDiagnostic(
            column="ocr", display_name="OCR NIM",
            invoke_url="http://ocr.local/nemotron-ocr-v2?token=never-show-this", role="ocr",
        )},
    )


def test_string_is_atomic_and_lists_join_only_at_list_level() -> None:
    assert normalize_error("GraphIngestionError").message == "GraphIngestionError"
    assert ";" not in normalize_error("GraphIngestionError").summary
    assert normalize_error(["first failure", "second failure"]).message == "Error: first failure; Error: second failure"


def test_generic_exception_is_normalized() -> None:
    assert normalize_error(Exception("plain failure")).as_dict() == {
        "type": "Exception", "stage": None, "endpoint": None, "message": "plain failure"
    }


def test_graph_ingestion_string_constructor_never_renders_characters() -> None:
    rendered = str(GraphIngestionError("GraphIngestionError"))
    assert "G;r;a;p;h" not in rendered
    assert "GraphIngestionError" in rendered


def test_nested_ocr_error_preserves_safe_structured_fields() -> None:
    assert normalize_error(_ocr_error()).as_dict() == {
        "type": "GraphIngestionError", "stage": "OCR",
        "endpoint": "http://ocr.local/nemotron-ocr-v2", "message": "HTTP 422: invalid image data",
    }


def test_secrets_and_request_payloads_are_not_rendered() -> None:
    normalized = normalize_error({
        "type": "HTTPError", "endpoint": "https://user:pass@example.test/ocr?api_key=super-secret",
        "message": "Authorization: Bearer top-secret-token invalid image data",
        "request_body": {"image": "base64-private-payload"},
    })
    assert "top-secret-token" not in normalized.summary
    assert "super-secret" not in normalized.summary
    assert "base64-private-payload" not in normalized.summary
    assert normalized.endpoint == "https://example.test/ocr"


def test_status_and_metrics_share_normalized_failure() -> None:
    config = ServiceConfig(mode="standalone", auth=AuthConfig(allow_unscoped_dev=True))
    with TestClient(create_app(config)) as client:
        tracker = get_job_tracker()
        metrics = get_metrics()
        assert tracker is not None and metrics is not None
        tracker.register_job("normalization-job", expected_documents=1)
        tracker.register_document("normalization-doc", job_id="normalization-job")
        tracker.mark_processing("normalization-doc")
        metrics.record_document_accepted(document_id="normalization-doc", job_id="normalization-job")
        response = client.post("/v1/internal/job-callback", json={
            "id": "normalization-doc", "status": "failed", "error_details": normalize_error(_ocr_error()).as_dict(),
        })
        assert response.status_code == 200, response.text
        status = client.get("/v1/ingest/status/normalization-doc")
        metric = client.get("/v1/ingest/metrics/document/normalization-doc")
        assert status.status_code == 200
        assert metric.status_code == 200
        assert status.json()["error_details"] == metric.json()["error_details"]
        assert status.json()["error"] == metric.json()["error"]
        assert "G;r;a;p;h" not in status.json()["error"]
