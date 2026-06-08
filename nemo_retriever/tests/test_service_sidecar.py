# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 6 tests for sidecar metadata in service run_mode.

Three layers:

* ``SidecarStore`` unit behaviour (put / get / consume / TTL / scoping).
* ``POST /v1/ingest/sidecar`` round-trip via FastAPI TestClient.
* Worker-side resolution: ``meta_dataframe_id`` → in-band bytes →
  ``pandas.DataFrame`` inside the pipeline child process.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from nemo_retriever.service.app import create_app
from nemo_retriever.service.config import (
    PipelineOverridesConfig,
    PipelinePoolConfig,
    ServiceConfig,
    SinksConfig,
)
from nemo_retriever.common.schemas.pipeline_spec import PipelineSpec
from nemo_retriever.common.policy import PolicyError, validate_pipeline_spec
from nemo_retriever.service.services.pipeline_executor import (
    _materialize_sidecar_bytes,
    _resolve_sidecar_in_spec,
)
from nemo_retriever.service.services.pipeline_pool import WorkItem
from nemo_retriever.service.services.sidecar_store import (
    SidecarStore,
    init_sidecar_store,
    shutdown_sidecar_store,
)


# ----------------------------------------------------------------------
# SidecarStore unit behaviour
# ----------------------------------------------------------------------


def test_store_put_get_consume_roundtrip() -> None:
    store = SidecarStore()
    entry = store.put(
        filename="meta.csv",
        content_type="text/csv",
        payload=b"id,title\n1,foo\n",
    )
    assert entry.sidecar_id

    fetched = store.get(entry.sidecar_id)
    assert fetched is not None
    assert fetched.payload == b"id,title\n1,foo\n"

    consumed = store.consume(entry.sidecar_id)
    assert consumed is not None
    # Default consume_on_read removes the entry.
    assert store.get(entry.sidecar_id) is None


def test_store_consume_on_read_false_keeps_entry() -> None:
    store = SidecarStore()
    entry = store.put(
        filename="meta.csv",
        content_type="text/csv",
        payload=b"x,y\n1,2\n",
        consume_on_read=False,
    )
    assert store.consume(entry.sidecar_id) is not None
    assert store.get(entry.sidecar_id) is not None  # still present


def test_store_expiry_evicts_on_read() -> None:
    store = SidecarStore(default_ttl_s=0.05)
    entry = store.put(filename="m.csv", content_type="text/csv", payload=b"x\n")
    time.sleep(0.1)
    assert store.get(entry.sidecar_id) is None


def test_store_owner_token_scoping() -> None:
    store = SidecarStore()
    entry = store.put(filename="m.csv", content_type="text/csv", payload=b"x\n", owner_token="alice")
    assert store.get(entry.sidecar_id, owner_token="alice") is not None
    assert store.get(entry.sidecar_id, owner_token="eve") is None
    # Without a token when one was set: forbidden.
    assert store.get(entry.sidecar_id, owner_token=None) is None


def test_store_max_entries_guard() -> None:
    store = SidecarStore(max_entries=2)
    store.put(filename="a", content_type="text/csv", payload=b"a")
    store.put(filename="b", content_type="text/csv", payload=b"b")
    with pytest.raises(RuntimeError, match="full"):
        store.put(filename="c", content_type="text/csv", payload=b"c")


# ----------------------------------------------------------------------
# Worker-side materialisation
# ----------------------------------------------------------------------


def _csv_bytes() -> bytes:
    return b"source,title\nfoo,Foo Doc\nbar,Bar Doc\n"


def _parquet_bytes() -> bytes:
    import pandas as pd

    df = pd.DataFrame({"source": ["foo", "bar"], "title": ["Foo Doc", "Bar Doc"]})
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def _json_bytes() -> bytes:
    return json.dumps([{"source": "foo", "title": "Foo Doc"}]).encode()


def test_materialize_csv_to_dataframe() -> None:
    kwargs = {
        "_meta_dataframe_bytes": _csv_bytes(),
        "_meta_dataframe_content_type": "text/csv",
        "_meta_dataframe_filename": "meta.csv",
        "meta_source_field": "source",
        "meta_fields": ["title"],
    }
    out = _materialize_sidecar_bytes(kwargs)
    df = out["meta_dataframe"]
    assert list(df.columns) == ["source", "title"]
    assert len(df) == 2
    # Private keys removed.
    assert "_meta_dataframe_bytes" not in out


def test_materialize_parquet_to_dataframe() -> None:
    kwargs = {
        "_meta_dataframe_bytes": _parquet_bytes(),
        "_meta_dataframe_content_type": "application/x-parquet",
        "_meta_dataframe_filename": "meta.parquet",
    }
    out = _materialize_sidecar_bytes(kwargs)
    df = out["meta_dataframe"]
    assert len(df) == 2


def test_materialize_no_payload_is_passthrough() -> None:
    kwargs = {"vdb_op": "lancedb"}
    out = _materialize_sidecar_bytes(kwargs)
    assert out == {"vdb_op": "lancedb"}


def test_resolve_sidecar_consumes_store_entry() -> None:
    store = init_sidecar_store()
    try:
        entry = store.put(filename="meta.csv", content_type="text/csv", payload=_csv_bytes())
        spec = {
            "vdb_upload_params": {
                "vdb_op": "lancedb",
                "vdb_kwargs": {"lancedb_uri": "s3://x/db"},
                "meta_dataframe_id": entry.sidecar_id,
                "meta_source_field": "source",
                "meta_fields": ["title"],
            }
        }
        resolved = _resolve_sidecar_in_spec(spec)
        vdb = resolved["vdb_upload_params"]
        assert "meta_dataframe_id" not in vdb
        assert vdb["_meta_dataframe_bytes"] == _csv_bytes()
        assert vdb["_meta_dataframe_content_type"] == "text/csv"
        # Entry has been consumed.
        assert store.get(entry.sidecar_id) is None
    finally:
        shutdown_sidecar_store()


def test_resolve_sidecar_missing_id_raises() -> None:
    init_sidecar_store()
    try:
        spec = {
            "vdb_upload_params": {
                "meta_dataframe_id": "nonexistent",
                "meta_source_field": "source",
                "meta_fields": ["title"],
            }
        }
        with pytest.raises(RuntimeError, match="not found"):
            _resolve_sidecar_in_spec(spec)
    finally:
        shutdown_sidecar_store()


def test_resolve_sidecar_no_vdb_is_passthrough() -> None:
    spec = {"extract_params": {"dpi": 200}}
    assert _resolve_sidecar_in_spec(spec) == spec


# ----------------------------------------------------------------------
# Policy: meta_dataframe_id is allowed; raw meta_dataframe is not
# ----------------------------------------------------------------------


def test_policy_admits_meta_dataframe_id() -> None:
    cfg = PipelineOverridesConfig(sinks=SinksConfig(vdb_uri_schemes=["s3://"]))
    spec = PipelineSpec(
        vdb_upload_params={
            "vdb_op": "lancedb",
            "vdb_kwargs": {"lancedb_uri": "s3://x/db"},
            "meta_dataframe_id": "abc123",
            "meta_source_field": "source",
            "meta_fields": ["title"],
        }
    )
    out = validate_pipeline_spec(spec, cfg.to_policy())
    assert out is spec


def test_policy_rejects_raw_meta_dataframe() -> None:
    cfg = PipelineOverridesConfig(sinks=SinksConfig(vdb_uri_schemes=["s3://"]))
    spec = PipelineSpec(
        vdb_upload_params={
            "vdb_op": "lancedb",
            "vdb_kwargs": {"lancedb_uri": "s3://x/db"},
            "meta_dataframe": "./local/path.csv",
            "meta_source_field": "source",
            "meta_fields": ["title"],
        }
    )
    with pytest.raises(PolicyError, match="POST /v1/ingest/sidecar"):
        validate_pipeline_spec(spec, cfg.to_policy())


# ----------------------------------------------------------------------
# Router round-trip via TestClient
# ----------------------------------------------------------------------


@pytest.fixture
def captured_items() -> list[WorkItem]:
    return []


@pytest.fixture
def app_with_sidecars(monkeypatch: pytest.MonkeyPatch, captured_items: list[WorkItem]):
    async def _stub_work(item: WorkItem) -> tuple[int, list[dict[str, Any]]]:
        captured_items.append(item)
        return 1, []

    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.create_realtime_work_fn",
        lambda _cfg: _stub_work,
    )
    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.create_batch_work_fn",
        lambda _cfg: _stub_work,
    )

    cfg = ServiceConfig(
        mode="standalone",
        pipeline=PipelinePoolConfig(realtime_workers=1, batch_workers=1),
        pipeline_overrides=PipelineOverridesConfig(sinks=SinksConfig(vdb_uri_schemes=["s3://"])),
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        yield client


def test_post_sidecar_returns_id(app_with_sidecars: TestClient) -> None:
    resp = app_with_sidecars.post(
        "/v1/ingest/sidecar",
        files={"file": ("meta.csv", _csv_bytes(), "text/csv")},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["sidecar_id"]
    assert body["filename"] == "meta.csv"
    assert body["content_type"] == "text/csv"
    assert body["size_bytes"] == len(_csv_bytes())
    assert "expires_at" in body


def test_post_sidecar_rejects_empty_upload(app_with_sidecars: TestClient) -> None:
    resp = app_with_sidecars.post(
        "/v1/ingest/sidecar",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert resp.status_code == 400


def test_delete_sidecar_returns_204(app_with_sidecars: TestClient) -> None:
    resp = app_with_sidecars.post(
        "/v1/ingest/sidecar",
        files={"file": ("m.csv", _csv_bytes(), "text/csv")},
    )
    sidecar_id = resp.json()["sidecar_id"]

    resp_del = app_with_sidecars.delete(f"/v1/ingest/sidecar/{sidecar_id}")
    assert resp_del.status_code == 204


def test_ingest_with_meta_dataframe_id_propagates_through_router(
    app_with_sidecars: TestClient, captured_items: list[WorkItem]
) -> None:
    """End-to-end: upload sidecar → reference by id in ingest request → worker sees resolved bytes."""
    from .conftest import create_test_job

    up = app_with_sidecars.post(
        "/v1/ingest/sidecar",
        files={"file": ("meta.csv", _csv_bytes(), "text/csv")},
    )
    sidecar_id = up.json()["sidecar_id"]

    metadata = {
        "pipeline": {
            "vdb_upload_params": {
                "vdb_op": "lancedb",
                "vdb_kwargs": {"lancedb_uri": "s3://corpus/lancedb"},
                "meta_dataframe_id": sidecar_id,
                "meta_source_field": "source",
                "meta_fields": ["title"],
            }
        }
    }
    job_id = create_test_job(app_with_sidecars)
    resp = app_with_sidecars.post(
        f"/v1/ingest/job/{job_id}/document",
        files={"file": ("doc.pdf", b"%PDF-1.4\n", "application/pdf")},
        data={"metadata": json.dumps(metadata)},
    )
    assert resp.status_code == 202, resp.text

    # Wait for the stub worker to pick up the item.
    deadline = time.monotonic() + 5.0
    while not captured_items and time.monotonic() < deadline:
        time.sleep(0.05)
    assert len(captured_items) == 1
    item = captured_items[0]
    # The pipeline_spec received by the worker still carries the id —
    # the bytes-resolution happens later in _work() right before
    # ProcessPoolExecutor.submit().
    assert item.pipeline_spec is not None
    assert item.pipeline_spec["vdb_upload_params"]["meta_dataframe_id"] == sidecar_id


def test_ingest_rejects_raw_meta_dataframe_through_router(
    app_with_sidecars: TestClient,
) -> None:
    from .conftest import create_test_job

    metadata = {
        "pipeline": {
            "vdb_upload_params": {
                "vdb_op": "lancedb",
                "vdb_kwargs": {"lancedb_uri": "s3://x/db"},
                "meta_dataframe": "/local/path.csv",
                "meta_source_field": "source",
                "meta_fields": ["title"],
            }
        }
    }
    job_id = create_test_job(app_with_sidecars)
    resp = app_with_sidecars.post(
        f"/v1/ingest/job/{job_id}/document",
        files={"file": ("doc.pdf", b"%PDF", "application/pdf")},
        data={"metadata": json.dumps(metadata)},
    )
    assert resp.status_code == 400, resp.text
    assert "POST /v1/ingest/sidecar" in resp.json()["detail"]


# ----------------------------------------------------------------------
# Client: ServiceIngestor.vdb_upload() auto-uploads + substitutes
# ----------------------------------------------------------------------


def test_client_uploads_sidecar_csv_and_replaces_with_id(
    app_with_sidecars: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end: ServiceIngestor.vdb_upload(meta_dataframe=path) → POST → id substituted."""
    from nemo_retriever.common.params import VdbUploadParams
    from nemo_retriever.service.service_ingestor import ServiceIngestor

    # Stub urllib.request.urlopen to drive the request through the TestClient.
    import urllib.request

    class _FakeResp:
        def __init__(self, payload: bytes, status: int = 201) -> None:
            self._payload = payload
            self.status = status

        def read(self) -> bytes:
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            return None

    captured_uploads: list[bytes] = []

    def _fake_urlopen(req, timeout=None):
        captured_uploads.append(req.data or b"")
        # Route to TestClient.
        resp = app_with_sidecars.post(
            req.full_url.replace("http://example:7670", ""),
            content=req.data,
            headers=dict(req.header_items()),
        )
        return _FakeResp(resp.content, status=resp.status_code)

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    sidecar_path = tmp_path / "meta.csv"
    sidecar_path.write_bytes(_csv_bytes())

    ing = ServiceIngestor(base_url="http://example:7670")
    ing.vdb_upload(
        VdbUploadParams(
            vdb_op="lancedb",
            vdb_kwargs={"lancedb_uri": "s3://corpus/lancedb"},
            meta_dataframe=str(sidecar_path),
            meta_source_field="source",
            meta_fields=["title"],
        )
    )

    spec = ing._pipeline_spec["vdb_upload_params"]
    assert "meta_dataframe" not in spec
    assert "meta_dataframe_id" in spec
    assert spec["meta_source_field"] == "source"
    assert spec["meta_fields"] == ["title"]
    # The upload actually moved bytes across the wire.
    assert len(captured_uploads) == 1
    assert _csv_bytes() in captured_uploads[0]
