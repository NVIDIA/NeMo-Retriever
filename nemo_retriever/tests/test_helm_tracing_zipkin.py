# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Helm-rendered tracing and Zipkin resources."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest import SkipTest

import yaml


RELEASE = "tracing-regression"
FULLNAME = f"{RELEASE}-nemo-retriever"
ZIPKIN_NAME = f"{FULLNAME}-zipkin"
OTEL_NAME = f"{FULLNAME}-otel"
OTEL_CONFIG_NAME = f"{OTEL_NAME}-config"

NIMSERVICE_NAMES = {
    "audio",
    "llama-nemotron-embed-vl-1b-v2",
    "llama-nemotron-rerank-vl-1b-v2",
    "nemotron-3-nano-omni-30b-a3b-reasoning",
    "nemotron-ocr-v1",
    "nemotron-page-elements-v3",
    "nemotron-parse",
    "nemotron-table-structure-v1",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _helm_template(extra_sets: list[str] | None = None) -> list[dict]:
    helm = shutil.which("helm")
    if helm is None:
        raise SkipTest("`helm` binary not available in this environment.")
    chart_path = _repo_root() / "nemo_retriever/helm"
    cmd = [
        helm,
        "template",
        RELEASE,
        str(chart_path),
        "--set",
        "ngcImagePullSecret.create=false",
        "--set",
        "ngcApiSecret.create=false",
        "--set",
        "nimOperator.rerankqa.enabled=true",
        "--set",
        "nimOperator.audio.enabled=true",
        "--set",
        "nimOperator.nemotron_parse.enabled=true",
        "--set",
        "nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.enabled=true",
        "--api-versions",
        "apps.nvidia.com/v1alpha1",
    ]
    for flag in extra_sets or []:
        cmd.extend(["--set", flag])
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f"`helm template` failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return [doc for doc in yaml.safe_load_all(proc.stdout) if doc]


def _find(docs: list[dict], kind: str, name: str) -> dict:
    for doc in docs:
        if doc.get("kind") == kind and doc.get("metadata", {}).get("name") == name:
            return doc
    raise AssertionError(f"Rendered {kind}/{name} not found.")


def _names_for_kind(docs: list[dict], kind: str) -> set[str]:
    return {doc.get("metadata", {}).get("name") for doc in docs if doc.get("kind") == kind}


def _deployment_env(doc: dict) -> list[dict]:
    return doc["spec"]["template"]["spec"]["containers"][0]["env"]


def _nim_env(doc: dict) -> list[dict]:
    return doc["spec"]["env"]


def _env_values(env: list[dict]) -> dict[str, str]:
    return {item["name"]: item.get("value") for item in env}


def _assert_unique_env_names(env: list[dict]) -> None:
    names = [item["name"] for item in env]
    assert len(names) == len(set(names))


def test_default_renders_zipkin_deployment_and_service() -> None:
    docs = _helm_template()

    _find(docs, "Deployment", ZIPKIN_NAME)
    service = _find(docs, "Service", ZIPKIN_NAME)
    _find(docs, "Deployment", OTEL_NAME)
    _find(docs, "Service", OTEL_NAME)
    _find(docs, "ConfigMap", OTEL_CONFIG_NAME)

    assert service["spec"]["type"] == "ClusterIP"
    assert service["spec"]["ports"] == [{"name": "http", "protocol": "TCP", "port": 9411, "targetPort": "http"}]


def test_zipkin_disabled_omits_zipkin_resources() -> None:
    docs = _helm_template(["topology.zipkin.enabled=false"])

    deployment_names = _names_for_kind(docs, "Deployment")
    service_names = _names_for_kind(docs, "Service")

    assert ZIPKIN_NAME not in deployment_names
    assert ZIPKIN_NAME not in service_names
    assert OTEL_NAME in deployment_names
    assert OTEL_NAME in service_names


def test_otel_config_exports_traces_to_rendered_zipkin_endpoint() -> None:
    docs = _helm_template()
    config = yaml.safe_load(_find(docs, "ConfigMap", OTEL_CONFIG_NAME)["data"]["config.yaml"])

    assert config["exporters"]["zipkin"]["endpoint"] == f"http://{ZIPKIN_NAME}:9411/api/v2/spans"
    trace_exporters = config["service"]["pipelines"]["traces"]["exporters"]
    assert "zipkin" in trace_exporters
    assert trace_exporters.count("zipkin") == 1


def test_standalone_service_gets_otel_env_without_duplicate_user_overrides() -> None:
    docs = _helm_template(
        [
            "service.env[0].name=OTEL_SERVICE_NAME",
            "service.env[0].value=user-service-name",
        ]
    )
    env = _deployment_env(_find(docs, "Deployment", FULLNAME))
    values = _env_values(env)

    _assert_unique_env_names(env)
    assert values["OTEL_EXPORTER_OTLP_ENDPOINT"] == f"http://{OTEL_NAME}:4317"
    assert values["OTEL_SERVICE_NAME"] == "user-service-name"
    assert values["OTEL_TRACES_EXPORTER"] == "otlp"
    assert values["OTEL_METRICS_EXPORTER"] == "otlp"
    assert values["OTEL_LOGS_EXPORTER"] == "none"
    assert values["OTEL_PROPAGATORS"] == "tracecontext,baggage"
    assert values["OTEL_RESOURCE_ATTRIBUTES"] == "service.namespace=nemo-retriever,service.role=standalone"
    assert values["OTEL_PYTHON_EXCLUDED_URLS"] == "health"


def test_split_roles_get_otel_env() -> None:
    docs = _helm_template(["topology.mode=split"])

    for role in ("gateway", "realtime", "batch"):
        env = _deployment_env(_find(docs, "Deployment", f"{FULLNAME}-{role}"))
        values = _env_values(env)

        _assert_unique_env_names(env)
        assert values["OTEL_EXPORTER_OTLP_ENDPOINT"] == f"http://{OTEL_NAME}:4317"
        assert values["OTEL_SERVICE_NAME"] == "nemo-retriever-service"
        assert values["OTEL_RESOURCE_ATTRIBUTES"] == f"service.namespace=nemo-retriever,service.role={role}"


def test_all_enabled_nimservices_inherit_otel_env() -> None:
    docs = _helm_template()
    nimservices = [doc for doc in docs if doc.get("kind") == "NIMService"]

    assert {doc["metadata"]["name"] for doc in nimservices} == NIMSERVICE_NAMES
    for doc in nimservices:
        name = doc["metadata"]["name"]
        env = _nim_env(doc)
        values = _env_values(env)

        _assert_unique_env_names(env)
        assert values["NIM_ENABLE_OTEL"] == "true"
        assert values["NIM_OTEL_SERVICE_NAME"] == name
        assert values["NIM_OTEL_TRACES_EXPORTER"] == "otlp"
        assert values["NIM_OTEL_METRICS_EXPORTER"] == "console"
        assert values["NIM_OTEL_EXPORTER_OTLP_ENDPOINT"] == f"http://{OTEL_NAME}:4318"
        assert values["TRITON_OTEL_URL"] == f"http://{OTEL_NAME}:4318/v1/traces"
        assert values["TRITON_OTEL_RATE"] == "1"


def test_per_nim_otel_opt_out_and_override() -> None:
    docs = _helm_template(
        [
            "nimOperator.page_elements.otel.enabled=false",
            "nimOperator.otel.endpoint=http://custom-otel:4318",
            "nimOperator.rerankqa.otel.serviceName=custom-rerank",
            "nimOperator.rerankqa.otel.env.NIM_OTEL_METRICS_EXPORTER=none",
            "nimOperator.rerankqa.env[0].name=NIM_OTEL_TRACES_EXPORTER",
            "nimOperator.rerankqa.env[0].value=zipkin",
        ]
    )

    page_elements = _find(docs, "NIMService", "nemotron-page-elements-v3")
    assert "NIM_ENABLE_OTEL" not in _env_values(_nim_env(page_elements))

    rerank = _find(docs, "NIMService", "llama-nemotron-rerank-vl-1b-v2")
    env = _nim_env(rerank)
    values = _env_values(env)

    _assert_unique_env_names(env)
    assert values["NIM_OTEL_SERVICE_NAME"] == "custom-rerank"
    assert values["NIM_OTEL_TRACES_EXPORTER"] == "zipkin"
    assert values["NIM_OTEL_METRICS_EXPORTER"] == "none"
    assert values["NIM_OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://custom-otel:4318"
    assert values["TRITON_OTEL_URL"] == "http://custom-otel:4318/v1/traces"
