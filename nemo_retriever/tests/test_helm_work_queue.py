# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess

import pytest

from tests.test_helm_shared_results import _render, _service_deployments


def test_split_work_queue_identity_spool_and_gateway_url() -> None:
    documents = _render(
        "--set",
        "topology.mode=split",
        "--set",
        "serviceMonitor.autoEnableInSplitMode=false",
    )
    deployments = _service_deployments(documents)
    assert len(deployments) == 3

    for deployment in deployments:
        component = deployment["metadata"]["labels"]["app.kubernetes.io/component"]
        pod_spec = deployment["spec"]["template"]["spec"]
        container = next(item for item in pod_spec["containers"] if item["name"] == "nemo-retriever")
        env = {item["name"]: item for item in container["env"]}
        assert env["POD_UID"]["valueFrom"]["fieldRef"]["fieldPath"] == "metadata.uid"
        assert env["POD_IP"]["valueFrom"]["fieldRef"]["fieldPath"] == "status.podIP"
        if component == "gateway":
            tmp = next(item for item in pod_spec["volumes"] if item["name"] == "tmp")
            assert tmp["emptyDir"]["sizeLimit"] == "20Gi"

    configmaps = [document for document in documents if document.get("kind") == "ConfigMap"]
    service_configs = [
        document["data"]["retriever-service.yaml"]
        for document in configmaps
        if "retriever-service.yaml" in document.get("data", {})
    ]
    assert len(service_configs) == 3
    assert all(
        'gateway_url: "http://shared-results-test-nemo-retriever-gateway:7670"' in cfg for cfg in service_configs
    )


def test_hpa_uses_central_gateway_backlog_average_value() -> None:
    documents = _render("--set", "topology.mode=split")
    hpas = [document for document in documents if document.get("kind") == "HorizontalPodAutoscaler"]
    assert len(hpas) == 2
    for hpa in hpas:
        backlog = next(metric for metric in hpa["spec"]["metrics"] if metric["type"] == "External")
        assert backlog["external"]["metric"]["name"] == "nemo_retriever_gateway_work_queue_backlog"
        assert backlog["external"]["target"]["type"] == "AverageValue"

    rules = next(
        document
        for document in documents
        if document.get("kind") == "ConfigMap" and document["metadata"]["name"].endswith("prom-adapter-rules")
    )
    assert "nemo_retriever_work_queue_demand" in rules["data"]["rules.yaml"]
    assert "nemo_retriever_work_queue_items" not in rules["data"]["rules.yaml"]
    assert "sum by (pool)" in rules["data"]["rules.yaml"]


def test_split_gateway_uses_single_writer_pvc_and_recreate_strategy() -> None:
    documents = _render("--set", "topology.mode=split")
    deployments = _service_deployments(documents)
    gateway = next(
        item for item in deployments if item["metadata"]["labels"]["app.kubernetes.io/component"] == "gateway"
    )
    assert gateway["spec"]["strategy"]["type"] == "Recreate"
    pod_spec = gateway["spec"]["template"]["spec"]
    assert next(volume for volume in pod_spec["volumes"] if volume["name"] == "data")["persistentVolumeClaim"]
    container = next(item for item in pod_spec["containers"] if item["name"] == "nemo-retriever")
    assert (
        next(mount for mount in container["volumeMounts"] if mount["name"] == "data")["mountPath"]
        == "/var/lib/nemo-retriever"
    )


def test_persistence_disabled_uses_ephemeral_spool() -> None:
    documents = _render("--set", "topology.mode=split", "--set", "persistence.enabled=false")
    gateway = next(
        item
        for item in _service_deployments(documents)
        if item["metadata"]["labels"]["app.kubernetes.io/component"] == "gateway"
    )
    assert all(volume["name"] != "data" for volume in gateway["spec"]["template"]["spec"]["volumes"])
    configs = [
        item["data"]["retriever-service.yaml"]
        for item in documents
        if item.get("kind") == "ConfigMap" and "retriever-service.yaml" in item.get("data", {})
    ]
    assert all('spool_directory: "/tmp/nemo-retriever-work"' in config for config in configs)
    assert all("persistence_enabled: false" in config for config in configs)


def test_legacy_queue_keys_override_defaults_and_annotate_fractional_substitution() -> None:
    documents = _render(
        "--set",
        "topology.mode=split",
        "--set",
        "topology.realtime.hpa.metrics.queueDepthRatio.enabled=true",
        "--set",
        "topology.realtime.hpa.metrics.queueDepthRatio.target=0.5",
        "--set",
        "autoscaling.queueDepth.prometheusAdapter.queueDepthRatioMetric=legacy_demand",
    )
    realtime = next(
        item
        for item in documents
        if item.get("kind") == "HorizontalPodAutoscaler" and item["metadata"]["name"].endswith("realtime")
    )
    assert (
        realtime["metadata"]["annotations"]["nemo-retriever.nvidia.com/legacy-queue-depth-ratio"]
        == "fractional target substituted with backlog-count default"
    )
    metric = next(item for item in realtime["spec"]["metrics"] if item["type"] == "External")
    assert metric["external"]["metric"]["name"] == "legacy_demand"
    assert metric["external"]["target"]["averageValue"] == "24"


def test_durable_split_rejects_multiple_gateway_replicas() -> None:
    with pytest.raises(subprocess.CalledProcessError):
        _render(
            "--set",
            "topology.mode=split",
            "--set",
            "topology.gateway.replicas=2",
        )
