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
SERVICE_OTEL_ENABLED = ["service.otel.enabled=true"]
NIM_OTEL_ENABLED = ["nimOperator.otel.enabled=true"]
POD_OTEL_ENABLED = SERVICE_OTEL_ENABLED + NIM_OTEL_ENABLED

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


def _helm_template_cmd(
    extra_sets: list[str] | None = None,
    extra_args: list[str] | None = None,
    release_name: str = RELEASE,
) -> list[str]:
    helm = shutil.which("helm")
    if helm is None:
        raise SkipTest("`helm` binary not available in this environment.")
    chart_path = _repo_root() / "nemo_retriever/helm"
    cmd = [
        helm,
        "template",
        release_name,
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
    cmd.extend(extra_args or [])
    return cmd


def _helm_template_process(
    extra_sets: list[str] | None = None,
    extra_args: list[str] | None = None,
    release_name: str = RELEASE,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _helm_template_cmd(extra_sets=extra_sets, extra_args=extra_args, release_name=release_name),
        check=False,
        capture_output=True,
        text=True,
    )


def _helm_template(
    extra_sets: list[str] | None = None,
    extra_args: list[str] | None = None,
    release_name: str = RELEASE,
) -> list[dict]:
    proc = _helm_template_process(extra_sets=extra_sets, extra_args=extra_args, release_name=release_name)
    if proc.returncode != 0:
        raise AssertionError(f"`helm template` failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return [doc for doc in yaml.safe_load_all(proc.stdout) if doc]


def _write_values_file(tmp_path: Path, values: dict) -> Path:
    values_path = tmp_path / "values.yaml"
    values_path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return values_path


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


def _container_names(doc: dict) -> set[str]:
    return {container.get("name") for container in doc["spec"]["template"]["spec"].get("containers", [])}


def _find_deployment_by_container(docs: list[dict], container_name: str) -> dict:
    for doc in docs:
        if doc.get("kind") == "Deployment" and container_name in _container_names(doc):
            return doc
    raise AssertionError(f"Rendered Deployment with container {container_name!r} not found.")


def _find_service_by_port_name(docs: list[dict], port_name: str) -> dict:
    for doc in docs:
        if doc.get("kind") != "Service":
            continue
        port_names = {port.get("name") for port in doc.get("spec", {}).get("ports", [])}
        if port_name in port_names:
            return doc
    raise AssertionError(f"Rendered Service with port {port_name!r} not found.")


def _find_configmap_with_key(docs: list[dict], key: str) -> dict:
    for doc in docs:
        if doc.get("kind") == "ConfigMap" and key in doc.get("data", {}):
            return doc
    raise AssertionError(f"Rendered ConfigMap with key {key!r} not found.")


def _find_by_component(docs: list[dict], kind: str, component: str) -> dict:
    for doc in docs:
        if (
            doc.get("kind") == kind
            and doc.get("metadata", {}).get("labels", {}).get("app.kubernetes.io/component") == component
        ):
            return doc
    raise AssertionError(f"Rendered {kind} with component {component!r} not found.")


def test_long_release_preserves_tracing_name_suffixes_and_references() -> None:
    long_release = "a" * 53
    long_fullname = "b" * 63
    docs = _helm_template(
        SERVICE_OTEL_ENABLED,
        release_name=long_release,
        extra_args=["--set-string", f"fullnameOverride={long_fullname}"],
    )

    otel_deployment = _find_deployment_by_container(docs, "otel-collector")
    zipkin_deployment = _find_deployment_by_container(docs, "zipkin")
    service_deployment = _find_deployment_by_container(docs, "nemo-retriever")
    otel_service = _find_service_by_port_name(docs, "otlp-grpc")
    zipkin_service = _find_by_component(docs, "Service", "zipkin")
    otel_config = _find_configmap_with_key(docs, "config.yaml")

    deployment_names = [doc["metadata"]["name"] for doc in docs if doc.get("kind") == "Deployment"]
    service_names = [doc["metadata"]["name"] for doc in docs if doc.get("kind") == "Service"]
    assert len(deployment_names) == len(set(deployment_names))
    assert len(service_names) == len(set(service_names))

    otel_name = otel_deployment["metadata"]["name"]
    zipkin_name = zipkin_deployment["metadata"]["name"]
    relevant_names = {
        otel_name,
        otel_service["metadata"]["name"],
        otel_config["metadata"]["name"],
        zipkin_name,
        zipkin_service["metadata"]["name"],
    }
    assert all(len(name) <= 63 for name in relevant_names)
    assert otel_name.endswith("-otel")
    assert otel_service["metadata"]["name"] == otel_name
    assert zipkin_name.endswith("-zipkin")
    assert zipkin_service["metadata"]["name"] == zipkin_name

    volume_config_name = otel_deployment["spec"]["template"]["spec"]["volumes"][0]["configMap"]["name"]
    assert volume_config_name == otel_config["metadata"]["name"]

    service_env = _env_values(_deployment_env(service_deployment))
    assert service_env["OTEL_EXPORTER_OTLP_ENDPOINT"] == f"http://{otel_name}:4317"

    otel_config_data = yaml.safe_load(otel_config["data"]["config.yaml"])
    assert otel_config_data["exporters"]["zipkin"]["endpoint"] == f"http://{zipkin_name}:9411/api/v2/spans"


def test_null_otel_env_maps_render_as_empty_maps() -> None:
    docs = _helm_template(
        POD_OTEL_ENABLED,
        extra_args=[
            "--set-json",
            "service.otel.env=null",
            "--set-json",
            "nimOperator.page_elements.otel=null",
            "--set-json",
            "nimOperator.otel.env=null",
            "--set-json",
            "nimOperator.rerankqa.otel.env=null",
        ],
    )

    service_env = _env_values(_deployment_env(_find(docs, "Deployment", FULLNAME)))
    assert service_env["OTEL_EXPORTER_OTLP_ENDPOINT"] == f"http://{OTEL_NAME}:4317"

    page_env = _env_values(_nim_env(_find(docs, "NIMService", "nemotron-page-elements-v3")))
    assert page_env["NIM_ENABLE_OTEL"] == "true"
    assert page_env["NIM_OTEL_EXPORTER_OTLP_ENDPOINT"] == f"http://{OTEL_NAME}:4318"

    rerank_env = _env_values(_nim_env(_find(docs, "NIMService", "llama-nemotron-rerank-vl-1b-v2")))
    assert rerank_env["NIM_ENABLE_OTEL"] == "true"
    assert rerank_env["NIM_OTEL_EXPORTER_OTLP_ENDPOINT"] == f"http://{OTEL_NAME}:4318"


def test_default_renders_zipkin_deployment_and_service() -> None:
    docs = _helm_template()

    _find(docs, "Deployment", ZIPKIN_NAME)
    service = _find(docs, "Service", ZIPKIN_NAME)
    _find(docs, "Deployment", OTEL_NAME)
    _find(docs, "Service", OTEL_NAME)
    _find(docs, "ConfigMap", OTEL_CONFIG_NAME)

    assert service["spec"]["type"] == "ClusterIP"
    assert service["spec"]["ports"] == [{"name": "http", "protocol": "TCP", "port": 9411, "targetPort": "http"}]


def test_default_injects_managed_service_and_nim_otel_env() -> None:
    docs = _helm_template()
    service_env = _deployment_env(_find(docs, "Deployment", FULLNAME))
    service_values = _env_values(service_env)

    _assert_unique_env_names(service_env)
    assert service_values["OTEL_EXPORTER_OTLP_ENDPOINT"] == f"http://{OTEL_NAME}:4317"
    assert service_values["OTEL_SERVICE_NAME"] == "nemo-retriever-service"
    assert service_values["OTEL_TRACES_EXPORTER"] == "otlp"
    assert service_values["OTEL_METRICS_EXPORTER"] == "otlp"
    assert service_values["OTEL_LOGS_EXPORTER"] == "none"
    assert service_values["OTEL_PROPAGATORS"] == "tracecontext,baggage"
    assert service_values["OTEL_RESOURCE_ATTRIBUTES"] == "service.namespace=nemo-retriever,service.role=standalone"
    assert service_values["OTEL_PYTHON_EXCLUDED_URLS"] == "health"

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


def test_tracing_pods_include_image_pull_secrets() -> None:
    docs = _helm_template(extra_args=["--set", "imagePullSecrets[0].name=trace-registry"])

    for deployment_name in (OTEL_NAME, ZIPKIN_NAME):
        pod_spec = _find(docs, "Deployment", deployment_name)["spec"]["template"]["spec"]

        assert pod_spec["imagePullSecrets"] == [
            {"name": "ngc-secret"},
            {"name": "trace-registry"},
        ]


def test_zipkin_disabled_omits_zipkin_resources_and_exporter() -> None:
    docs = _helm_template(["topology.zipkin.enabled=false"])

    deployment_names = _names_for_kind(docs, "Deployment")
    service_names = _names_for_kind(docs, "Service")

    assert ZIPKIN_NAME not in deployment_names
    assert ZIPKIN_NAME not in service_names
    assert OTEL_NAME in deployment_names
    assert OTEL_NAME in service_names

    config = yaml.safe_load(_find(docs, "ConfigMap", OTEL_CONFIG_NAME)["data"]["config.yaml"])
    assert "zipkin" not in config["exporters"]
    assert "zipkin" not in config["service"]["pipelines"]["traces"]["exporters"]
    assert ZIPKIN_NAME not in _find(docs, "ConfigMap", OTEL_CONFIG_NAME)["data"]["config.yaml"]


def test_zipkin_disabled_with_external_endpoint_still_exports_to_zipkin() -> None:
    external_endpoint = "http://external-zipkin:9411/api/v2/spans"
    docs = _helm_template(
        ["topology.zipkin.enabled=false"],
        extra_args=[
            "--set-string",
            f"topology.zipkin.exporter.endpoint={external_endpoint}",
        ],
    )
    config = yaml.safe_load(_find(docs, "ConfigMap", OTEL_CONFIG_NAME)["data"]["config.yaml"])

    assert ZIPKIN_NAME not in _names_for_kind(docs, "Deployment")
    assert config["exporters"]["zipkin"]["endpoint"] == external_endpoint
    assert "zipkin" in config["service"]["pipelines"]["traces"]["exporters"]


def test_null_otel_config_fails_with_chart_authored_message() -> None:
    proc = _helm_template_process(extra_args=["--set-json", "topology.otel.config=null"])

    assert proc.returncode != 0
    assert "topology.otel.config must be a map when topology.otel.enabled=true" in proc.stderr
    assert "reflect:" not in proc.stderr
    assert "nil pointer" not in proc.stderr


def test_null_otel_ports_fails_with_chart_authored_message() -> None:
    proc = _helm_template_process(extra_args=["--set-json", "topology.otel.ports=null"])

    assert proc.returncode != 0
    assert "topology.otel.ports must be a map when topology.otel.enabled=true" in proc.stderr
    assert "reflect:" not in proc.stderr
    assert "nil pointer" not in proc.stderr


def test_null_zipkin_subtree_omits_zipkin_resources_and_exporter() -> None:
    docs = _helm_template(extra_args=["--set-json", "topology.zipkin=null"])

    deployment_names = _names_for_kind(docs, "Deployment")
    service_names = _names_for_kind(docs, "Service")
    config_text = _find(docs, "ConfigMap", OTEL_CONFIG_NAME)["data"]["config.yaml"]
    config = yaml.safe_load(config_text)

    assert ZIPKIN_NAME not in deployment_names
    assert ZIPKIN_NAME not in service_names
    assert OTEL_NAME in deployment_names
    assert OTEL_NAME in service_names
    assert "zipkin" not in config["exporters"]
    assert "zipkin" not in config["service"]["pipelines"]["traces"]["exporters"]
    assert ZIPKIN_NAME not in config_text


def test_null_zipkin_exporter_omits_exporter_injection() -> None:
    docs = _helm_template(extra_args=["--set-json", "topology.zipkin.exporter=null"])

    config = yaml.safe_load(_find(docs, "ConfigMap", OTEL_CONFIG_NAME)["data"]["config.yaml"])

    assert ZIPKIN_NAME in _names_for_kind(docs, "Deployment")
    assert ZIPKIN_NAME in _names_for_kind(docs, "Service")
    assert "zipkin" not in config["exporters"]
    assert "zipkin" not in config["service"]["pipelines"]["traces"]["exporters"]


def test_null_zipkin_image_fails_with_chart_authored_message() -> None:
    proc = _helm_template_process(extra_args=["--set-json", "topology.zipkin.image=null"])

    assert proc.returncode != 0
    assert "topology.zipkin.image must be a map when topology.zipkin.enabled=true" in proc.stderr
    assert "reflect:" not in proc.stderr
    assert "nil pointer" not in proc.stderr


def test_null_zipkin_port_fails_with_chart_authored_message() -> None:
    proc = _helm_template_process(extra_args=["--set-json", "topology.zipkin.port=null"])

    assert proc.returncode != 0
    assert "topology.zipkin.port is required when topology.zipkin.enabled=true" in proc.stderr
    assert "reflect:" not in proc.stderr
    assert "nil pointer" not in proc.stderr
    assert f"http://{ZIPKIN_NAME}:/api/v2/spans" not in proc.stdout
    assert "port: null" not in proc.stdout


def test_zipkin_exporter_injection_preserves_custom_exporter_settings() -> None:
    external_endpoint = "http://external-zipkin:9411/api/v2/spans"
    docs = _helm_template(
        extra_args=[
            "--set-string",
            "topology.otel.config.exporters.zipkin.endpoint=http://stale-zipkin:9411/api/v2/spans",
            "--set-string",
            "topology.otel.config.exporters.zipkin.timeout=15s",
            "--set",
            "topology.otel.config.exporters.zipkin.sending_queue.enabled=true",
            "--set-string",
            f"topology.zipkin.exporter.endpoint={external_endpoint}",
        ]
    )
    config = yaml.safe_load(_find(docs, "ConfigMap", OTEL_CONFIG_NAME)["data"]["config.yaml"])

    zipkin_exporter = config["exporters"]["zipkin"]
    assert zipkin_exporter["endpoint"] == external_endpoint
    assert zipkin_exporter["timeout"] == "15s"
    assert zipkin_exporter["sending_queue"]["enabled"] is True


def test_zipkin_deployment_renders_security_contexts() -> None:
    docs = _helm_template(
        extra_args=[
            "--set-json",
            'topology.zipkin.podSecurityContext={"runAsNonRoot":true,"seccompProfile":{"type":"RuntimeDefault"}}',
            "--set-json",
            (
                'topology.zipkin.securityContext={"allowPrivilegeEscalation":false,'
                '"readOnlyRootFilesystem":true,"capabilities":{"drop":["ALL"]}}'
            ),
        ]
    )
    deployment = _find(docs, "Deployment", ZIPKIN_NAME)
    pod_spec = deployment["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]

    assert pod_spec["securityContext"] == {
        "runAsNonRoot": True,
        "seccompProfile": {"type": "RuntimeDefault"},
    }
    assert container["securityContext"] == {
        "allowPrivilegeEscalation": False,
        "readOnlyRootFilesystem": True,
        "capabilities": {"drop": ["ALL"]},
    }


def test_zipkin_deployment_renders_default_tcp_health_probes() -> None:
    docs = _helm_template()
    deployment = _find(docs, "Deployment", ZIPKIN_NAME)
    container = deployment["spec"]["template"]["spec"]["containers"][0]

    for probe_name in ("livenessProbe", "readinessProbe", "startupProbe"):
        assert container[probe_name]["tcpSocket"] == {"port": "http"}


def test_zipkin_deployment_omits_disabled_probe() -> None:
    docs = _helm_template(["topology.zipkin.readinessProbe.enabled=false"])
    deployment = _find(docs, "Deployment", ZIPKIN_NAME)
    container = deployment["spec"]["template"]["spec"]["containers"][0]

    assert "readinessProbe" not in container
    assert "livenessProbe" in container
    assert "startupProbe" in container


def test_zipkin_deployment_allows_null_probe_maps_from_values_file(
    tmp_path: Path,
) -> None:
    values_path = _write_values_file(
        tmp_path,
        {
            "topology": {
                "zipkin": {
                    "startupProbe": None,
                    "livenessProbe": None,
                    "readinessProbe": None,
                }
            }
        },
    )
    docs = _helm_template(extra_args=["--values", str(values_path)])
    deployment = _find(docs, "Deployment", ZIPKIN_NAME)
    container = deployment["spec"]["template"]["spec"]["containers"][0]

    assert "startupProbe" not in container
    assert "livenessProbe" not in container
    assert "readinessProbe" not in container


def test_zipkin_deployment_omits_null_resources_from_values_file(
    tmp_path: Path,
) -> None:
    values_path = _write_values_file(tmp_path, {"topology": {"zipkin": {"resources": None}}})
    docs = _helm_template(extra_args=["--values", str(values_path)])
    deployment = _find(docs, "Deployment", ZIPKIN_NAME)
    container = deployment["spec"]["template"]["spec"]["containers"][0]

    assert "resources" not in container


def test_zipkin_injection_allows_traces_pipeline_without_processors() -> None:
    docs = _helm_template(
        extra_args=[
            "--set-json",
            "topology.otel.config.service.pipelines.traces.processors=[]",
        ]
    )
    config = yaml.safe_load(_find(docs, "ConfigMap", OTEL_CONFIG_NAME)["data"]["config.yaml"])
    traces = config["service"]["pipelines"]["traces"]

    assert traces["receivers"] == ["otlp"]
    assert traces["processors"] == []
    assert "zipkin" in traces["exporters"]
    assert traces["exporters"].count("zipkin") == 1


def test_zipkin_injection_initializes_missing_trace_exporters() -> None:
    docs = _helm_template(
        extra_args=[
            "--set-json",
            "topology.otel.config.service.pipelines.traces.exporters=null",
        ]
    )
    config = yaml.safe_load(_find(docs, "ConfigMap", OTEL_CONFIG_NAME)["data"]["config.yaml"])

    assert config["service"]["pipelines"]["traces"]["exporters"] == ["zipkin"]


def test_zipkin_injection_allows_empty_trace_component_maps() -> None:
    docs = _helm_template(
        extra_args=[
            "--set-json",
            "topology.otel.config.receivers.custom={}",
            "--set-json",
            "topology.otel.config.processors.custom={}",
            "--set-json",
            "topology.otel.config.exporters.custom={}",
            "--set-json",
            'topology.otel.config.service.pipelines.traces.receivers=["custom"]',
            "--set-json",
            'topology.otel.config.service.pipelines.traces.processors=["custom"]',
            "--set-json",
            'topology.otel.config.service.pipelines.traces.exporters=["custom"]',
        ]
    )
    config = yaml.safe_load(_find(docs, "ConfigMap", OTEL_CONFIG_NAME)["data"]["config.yaml"])
    traces = config["service"]["pipelines"]["traces"]

    assert config["receivers"]["custom"] == {}
    assert config["processors"]["custom"] == {}
    assert config["exporters"]["custom"] == {}
    assert traces["receivers"] == ["custom"]
    assert traces["processors"] == ["custom"]
    assert traces["exporters"] == ["custom", "zipkin"]


def test_zipkin_injection_requires_existing_traces_pipeline() -> None:
    proc = _helm_template_process(extra_args=["--set-json", "topology.otel.config.service.pipelines.traces=null"])

    assert proc.returncode != 0
    assert (
        "topology.zipkin.exporter.enabled requires topology.otel.config.service.pipelines.traces "
        "with non-empty receivers; provide that traces pipeline or set topology.zipkin.exporter.enabled=false"
    ) in proc.stderr


def test_zipkin_injection_requires_non_empty_trace_receivers() -> None:
    expected = (
        "topology.zipkin.exporter.enabled requires topology.otel.config.service.pipelines.traces "
        "with non-empty receivers; provide that traces pipeline or set topology.zipkin.exporter.enabled=false"
    )

    for receivers_override in (
        "topology.otel.config.service.pipelines.traces.receivers=null",
        "topology.otel.config.service.pipelines.traces.receivers=[]",
    ):
        proc = _helm_template_process(extra_args=["--set-json", receivers_override])

        assert proc.returncode != 0
        assert expected in proc.stderr


def test_zipkin_injection_requires_referenced_receiver_to_exist() -> None:
    proc = _helm_template_process(extra_args=["--set-json", "topology.otel.config.receivers.otlp=null"])

    assert proc.returncode != 0
    assert 'trace receiver "otlp" is missing or null' in proc.stderr
    assert "fix topology.otel.config or set topology.zipkin.exporter.enabled=false" in proc.stderr


def test_zipkin_injection_requires_referenced_processor_to_exist() -> None:
    proc = _helm_template_process(extra_args=["--set-json", "topology.otel.config.processors.batch=null"])

    assert proc.returncode != 0
    assert 'trace processor "batch" is missing or null' in proc.stderr
    assert "fix topology.otel.config or set topology.zipkin.exporter.enabled=false" in proc.stderr


def test_zipkin_injection_requires_referenced_exporter_to_exist() -> None:
    proc = _helm_template_process(extra_args=["--set-json", "topology.otel.config.exporters.debug=null"])

    assert proc.returncode != 0
    assert 'trace exporter "debug" is missing or null' in proc.stderr
    assert "fix topology.otel.config or set topology.zipkin.exporter.enabled=false" in proc.stderr


def test_otel_config_exports_traces_to_rendered_zipkin_endpoint() -> None:
    docs = _helm_template()
    config = yaml.safe_load(_find(docs, "ConfigMap", OTEL_CONFIG_NAME)["data"]["config.yaml"])

    assert config["exporters"]["zipkin"]["endpoint"] == f"http://{ZIPKIN_NAME}:9411/api/v2/spans"
    trace_exporters = config["service"]["pipelines"]["traces"]["exporters"]
    assert "zipkin" in trace_exporters
    assert trace_exporters.count("zipkin") == 1


def test_standalone_service_gets_otel_env_without_duplicate_user_overrides() -> None:
    docs = _helm_template(
        SERVICE_OTEL_ENABLED
        + [
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
    docs = _helm_template(SERVICE_OTEL_ENABLED + ["topology.mode=split"])

    for role in ("gateway", "realtime", "batch"):
        env = _deployment_env(_find(docs, "Deployment", f"{FULLNAME}-{role}"))
        values = _env_values(env)

        _assert_unique_env_names(env)
        assert values["OTEL_EXPORTER_OTLP_ENDPOINT"] == f"http://{OTEL_NAME}:4317"
        assert values["OTEL_SERVICE_NAME"] == "nemo-retriever-service"
        assert values["OTEL_RESOURCE_ATTRIBUTES"] == f"service.namespace=nemo-retriever,service.role={role}"


def test_all_enabled_nimservices_inherit_otel_env() -> None:
    docs = _helm_template(NIM_OTEL_ENABLED)
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


def test_service_otel_disable_omits_managed_env() -> None:
    docs = _helm_template(["service.otel.enabled=false"])
    env = _deployment_env(_find(docs, "Deployment", FULLNAME))
    values = _env_values(env)
    chart_managed_names = {
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_SERVICE_NAME",
        "OTEL_TRACES_EXPORTER",
        "OTEL_METRICS_EXPORTER",
        "OTEL_LOGS_EXPORTER",
        "OTEL_PROPAGATORS",
        "OTEL_RESOURCE_ATTRIBUTES",
        "OTEL_PYTHON_EXCLUDED_URLS",
    }

    _assert_unique_env_names(env)
    assert chart_managed_names.isdisjoint(values)


def test_chart_wide_nim_otel_disable_omits_managed_env() -> None:
    docs = _helm_template(["nimOperator.otel.enabled=false"])
    nimservices = [doc for doc in docs if doc.get("kind") == "NIMService"]
    chart_managed_names = {
        "NIM_ENABLE_OTEL",
        "NIM_OTEL_EXPORTER_OTLP_ENDPOINT",
        "TRITON_OTEL_URL",
    }

    assert {doc["metadata"]["name"] for doc in nimservices} == NIMSERVICE_NAMES
    for doc in nimservices:
        env = _nim_env(doc)
        values = _env_values(env)

        _assert_unique_env_names(env)
        assert chart_managed_names.isdisjoint(values)

    table_values = _env_values(_nim_env(_find(docs, "NIMService", "nemotron-table-structure-v1")))
    assert table_values["NIM_TRITON_CUDA_MEMORY_POOL_MB"] == "2048"


def test_per_nim_otel_endpoint_overrides_chart_endpoint() -> None:
    docs = _helm_template(
        NIM_OTEL_ENABLED
        + [
            "nimOperator.otel.endpoint=http://chart-otel:4318",
            "nimOperator.rerankqa.otel.endpoint=http://per-nim-otel:4318",
        ]
    )

    page_elements = _find(docs, "NIMService", "nemotron-page-elements-v3")
    page_values = _env_values(_nim_env(page_elements))
    assert page_values["NIM_OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://chart-otel:4318"
    assert page_values["TRITON_OTEL_URL"] == "http://chart-otel:4318/v1/traces"

    rerank = _find(docs, "NIMService", "llama-nemotron-rerank-vl-1b-v2")
    rerank_values = _env_values(_nim_env(rerank))
    assert rerank_values["NIM_OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://per-nim-otel:4318"
    assert rerank_values["TRITON_OTEL_URL"] == "http://per-nim-otel:4318/v1/traces"


def test_chart_nim_otel_env_endpoint_drives_triton_url() -> None:
    docs = _helm_template(
        NIM_OTEL_ENABLED + ["nimOperator.otel.env.NIM_OTEL_EXPORTER_OTLP_ENDPOINT=http://env-otel:4318"]
    )

    page_elements = _find(docs, "NIMService", "nemotron-page-elements-v3")
    page_values = _env_values(_nim_env(page_elements))

    assert page_values["NIM_OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://env-otel:4318"
    assert page_values["TRITON_OTEL_URL"] == "http://env-otel:4318/v1/traces"


def test_per_nim_otel_env_endpoint_drives_triton_url() -> None:
    docs = _helm_template(
        NIM_OTEL_ENABLED + ["nimOperator.rerankqa.otel.env.NIM_OTEL_EXPORTER_OTLP_ENDPOINT=http://per-env-otel:4318"]
    )

    rerank = _find(docs, "NIMService", "llama-nemotron-rerank-vl-1b-v2")
    rerank_values = _env_values(_nim_env(rerank))

    assert rerank_values["NIM_OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://per-env-otel:4318"
    assert rerank_values["TRITON_OTEL_URL"] == "http://per-env-otel:4318/v1/traces"


def test_nim_otel_env_triton_url_override_is_preserved() -> None:
    docs = _helm_template(
        NIM_OTEL_ENABLED
        + [
            "nimOperator.otel.env.NIM_OTEL_EXPORTER_OTLP_ENDPOINT=http://env-otel:4318",
            "nimOperator.otel.env.TRITON_OTEL_URL=http://explicit-triton/v1/traces",
        ]
    )

    page_elements = _find(docs, "NIMService", "nemotron-page-elements-v3")
    page_values = _env_values(_nim_env(page_elements))

    assert page_values["NIM_OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://env-otel:4318"
    assert page_values["TRITON_OTEL_URL"] == "http://explicit-triton/v1/traces"


def test_existing_nim_env_endpoint_drives_triton_url_without_duplicate_endpoint() -> None:
    docs = _helm_template(
        NIM_OTEL_ENABLED
        + [
            "nimOperator.rerankqa.env[0].name=NIM_OTEL_EXPORTER_OTLP_ENDPOINT",
            "nimOperator.rerankqa.env[0].value=http://manual-otel:4318",
        ]
    )

    rerank = _find(docs, "NIMService", "llama-nemotron-rerank-vl-1b-v2")
    env = _nim_env(rerank)
    values = _env_values(env)

    _assert_unique_env_names(env)
    assert values["NIM_OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://manual-otel:4318"
    assert values["TRITON_OTEL_URL"] == "http://manual-otel:4318/v1/traces"


def test_existing_nim_env_endpoint_value_from_omits_chart_managed_triton_url() -> None:
    docs = _helm_template(
        NIM_OTEL_ENABLED
        + [
            "nimOperator.page_elements.env[0].name=NIM_OTEL_EXPORTER_OTLP_ENDPOINT",
            "nimOperator.page_elements.env[0].valueFrom.secretKeyRef.name=otel-endpoint",
            "nimOperator.page_elements.env[0].valueFrom.secretKeyRef.key=endpoint",
        ]
    )

    page_elements = _find(docs, "NIMService", "nemotron-page-elements-v3")
    env = _nim_env(page_elements)
    values = _env_values(env)

    _assert_unique_env_names(env)
    assert values["NIM_OTEL_EXPORTER_OTLP_ENDPOINT"] is None
    assert next(item for item in env if item["name"] == "NIM_OTEL_EXPORTER_OTLP_ENDPOINT")["valueFrom"] == {
        "secretKeyRef": {"name": "otel-endpoint", "key": "endpoint"}
    }
    assert "TRITON_OTEL_URL" not in values


def test_existing_nim_env_triton_url_override_is_preserved() -> None:
    docs = _helm_template(
        NIM_OTEL_ENABLED
        + [
            "nimOperator.rerankqa.env[0].name=NIM_OTEL_EXPORTER_OTLP_ENDPOINT",
            "nimOperator.rerankqa.env[0].value=http://manual-otel:4318",
            "nimOperator.rerankqa.env[1].name=TRITON_OTEL_URL",
            "nimOperator.rerankqa.env[1].value=http://manual-triton/v1/traces",
        ]
    )

    rerank = _find(docs, "NIMService", "llama-nemotron-rerank-vl-1b-v2")
    env = _nim_env(rerank)
    values = _env_values(env)

    _assert_unique_env_names(env)
    assert values["NIM_OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://manual-otel:4318"
    assert values["TRITON_OTEL_URL"] == "http://manual-triton/v1/traces"


def test_per_nim_otel_opt_out_and_override() -> None:
    docs = _helm_template(
        NIM_OTEL_ENABLED
        + [
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
