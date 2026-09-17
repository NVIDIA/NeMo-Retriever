# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

from nemo_retriever.common.vdb.adt_vdb import VDB
from nemo_retriever.common.vdb import factory as vdb_factory
from nemo_retriever.common.policy import PipelineOverridesPolicy, PolicyError, SinkUrlAllowlist, validate_pipeline_spec
from nemo_retriever.common.schemas.pipeline_spec import PipelineSpec
from nemo_retriever.common.vdb.factory import (
    clear_vdb_op_registry_cache,
    extract_vdb_connection_uri,
    get_vdb_op_cls,
    list_vdb_ops,
)
from nemo_retriever.common.vdb.lancedb import LanceDB
from nemo_retriever.service.config import PipelineOverridesConfig, SinksConfig
from nemo_retriever.service.service_ingestor import ServiceIngestor
from nemo_retriever.common.params import VdbUploadParams


class _EntryPointVDB(VDB):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def create_index(self, **kwargs: Any) -> None:
        return None

    def write_to_index(self, records: list, **kwargs: Any) -> None:
        return None

    def retrieval(self, vectors: list, **kwargs: Any) -> list[list[dict[str, Any]]]:
        return [[] for _ in vectors]

    def run(self, records: Any) -> dict[str, Any]:
        return {"records": records}


@pytest.fixture(autouse=True)
def _reset_vdb_registry() -> None:
    clear_vdb_op_registry_cache()
    yield
    clear_vdb_op_registry_cache()


def test_list_vdb_ops_includes_builtin_lancedb() -> None:
    assert "lancedb" in list_vdb_ops()


def test_get_vdb_op_cls_returns_lancedb() -> None:
    assert get_vdb_op_cls("lancedb") is LanceDB


def test_get_vdb_op_cls_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Invalid vdb_op"):
        get_vdb_op_cls("does-not-exist")


def test_entry_point_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_registry() -> dict[str, type[VDB]]:
        return {"lancedb": LanceDB, "custom": _EntryPointVDB}

    monkeypatch.setattr(vdb_factory, "_load_vdb_op_registry", fake_registry)
    clear_vdb_op_registry_cache()

    assert get_vdb_op_cls("custom") is _EntryPointVDB
    assert "custom" in list_vdb_ops()


@pytest.mark.parametrize(
    ("vdb_kwargs", "expected"),
    [
        ({"lancedb_uri": "s3://bucket/db"}, "s3://bucket/db"),
        ({"uri": "gs://bucket/db"}, "gs://bucket/db"),
        ({"host": "https://opensearch.example.com"}, "https://opensearch.example.com"),
    ],
)
def test_extract_vdb_connection_uri(vdb_kwargs: dict[str, str], expected: str) -> None:
    assert extract_vdb_connection_uri(vdb_kwargs) == expected


def test_validate_rejects_disallowed_vdb_op() -> None:
    cfg = PipelineOverridesConfig(
        allowed_vdb_ops=["lancedb"],
        extra_vdb_kwargs_keys=["host"],
        sinks=SinksConfig(vdb_uri_schemes=["https://"]),
    )
    spec = PipelineSpec(
        vdb_upload_params={
            "vdb_op": "opensearch",
            "vdb_kwargs": {"host": "https://opensearch.example.com"},
        }
    )
    with pytest.raises(PolicyError, match="allowlist"):
        validate_pipeline_spec(spec, cfg.to_policy())


def test_validate_admits_custom_vdb_op_when_allowed() -> None:
    cfg = PipelineOverridesConfig(
        allowed_vdb_ops=["lancedb", "opensearch"],
        extra_vdb_kwargs_keys=["host", "index"],
        sinks=SinksConfig(vdb_uri_schemes=["https://"]),
    )
    spec = PipelineSpec(
        vdb_upload_params={
            "vdb_op": "opensearch",
            "vdb_kwargs": {"host": "https://opensearch.example.com", "index": "corpus"},
        }
    )
    out = validate_pipeline_spec(spec, cfg.to_policy())
    assert out is spec


def test_service_ingestor_accepts_custom_vdb_op_with_host_uri() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.vdb_upload(
        VdbUploadParams(
            vdb_op="opensearch",
            vdb_kwargs={"host": "https://opensearch.example.com", "index": "corpus"},
        )
    )
    payload = ing._pipeline_payload()
    assert payload is not None
    assert payload["vdb_upload_params"]["vdb_op"] == "opensearch"
    assert payload["vdb_upload_params"]["vdb_kwargs"]["host"] == "https://opensearch.example.com"


def test_policy_describe_includes_registered_vdb_ops() -> None:
    policy = PipelineOverridesPolicy(sinks=SinkUrlAllowlist(vdb_uri_schemes=["s3://"]))
    described = policy.describe()
    assert "lancedb" in described["registered_vdb_ops"]
    assert described["allowed_vdb_ops"] == ["lancedb"]
