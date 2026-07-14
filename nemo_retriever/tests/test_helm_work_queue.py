# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
        container = next(
            item for item in pod_spec["containers"] if item["name"] == "nemo-retriever"
        )
        env = {item["name"]: item for item in container["env"]}
        assert env["POD_UID"]["valueFrom"]["fieldRef"]["fieldPath"] == "metadata.uid"
        assert env["POD_IP"]["valueFrom"]["fieldRef"]["fieldPath"] == "status.podIP"
        if component == "gateway":
            tmp = next(item for item in pod_spec["volumes"] if item["name"] == "tmp")
            assert tmp["emptyDir"]["sizeLimit"] == "20Gi"

    configmaps = [
        document for document in documents if document.get("kind") == "ConfigMap"
    ]
    service_configs = [
        document["data"]["retriever-service.yaml"]
        for document in configmaps
        if "retriever-service.yaml" in document.get("data", {})
    ]
    assert len(service_configs) == 3
    assert all(
        'gateway_url: "http://shared-results-test-nemo-retriever-gateway:7670"' in cfg
        for cfg in service_configs
    )


def test_hpa_uses_central_gateway_backlog_average_value() -> None:
    documents = _render("--set", "topology.mode=split")
    hpas = [
        document
        for document in documents
        if document.get("kind") == "HorizontalPodAutoscaler"
    ]
    assert len(hpas) == 2
    for hpa in hpas:
        backlog = next(
            metric for metric in hpa["spec"]["metrics"] if metric["type"] == "External"
        )
        assert (
            backlog["external"]["metric"]["name"]
            == "nemo_retriever_gateway_work_queue_backlog"
        )
        assert backlog["external"]["target"]["type"] == "AverageValue"

    rules = next(
        document
        for document in documents
        if document.get("kind") == "ConfigMap"
        and document["metadata"]["name"].endswith("prom-adapter-rules")
    )
    assert "nemo_retriever_work_queue_items" in rules["data"]["rules.yaml"]
    assert "sum by (pool)" in rules["data"]["rules.yaml"]
