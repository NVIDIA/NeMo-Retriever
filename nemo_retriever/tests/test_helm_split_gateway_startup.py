# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helm wiring that keeps split-topology startup free of a readiness deadlock."""

from __future__ import annotations

from tests.test_helm_shared_results import _render, _service_deployments

GATEWAY_SERVICE = "shared-results-test-nemo-retriever-gateway"
STARTUP_SERVICE = "shared-results-test-nemo-retriever-gateway-startup"


def _services(documents: list[dict]) -> dict[str, dict]:
    return {document["metadata"]["name"]: document for document in documents if document.get("kind") == "Service"}


def _init_container_script(documents: list[dict], component: str) -> str:
    deployment = next(
        item
        for item in _service_deployments(documents)
        if item["metadata"]["labels"]["app.kubernetes.io/component"] == component
    )
    init_container = next(
        item for item in deployment["spec"]["template"]["spec"]["initContainers"] if item["name"] == "wait-for-gateway"
    )
    return init_container["command"][-1]


def test_split_startup_service_publishes_unready_gateway_address() -> None:
    documents = _render(
        "--set",
        "topology.mode=split",
        "--set",
        "serviceMonitor.autoEnableInSplitMode=false",
    )
    services = _services(documents)

    startup = services[STARTUP_SERVICE]
    assert startup["spec"]["type"] == "ClusterIP"
    assert startup["spec"]["publishNotReadyAddresses"] is True
    assert startup["spec"]["selector"]["app.kubernetes.io/component"] == "gateway"
    assert startup["spec"]["ports"] == [{"name": "http", "protocol": "TCP", "port": 7670, "targetPort": "http"}]

    # Client traffic keeps the deep-readiness gate introduced for NVBug 6610671.
    assert "publishNotReadyAddresses" not in services[GATEWAY_SERVICE]["spec"]


def test_split_worker_init_containers_wait_on_startup_service() -> None:
    documents = _render(
        "--set",
        "topology.mode=split",
        "--set",
        "serviceMonitor.autoEnableInSplitMode=false",
    )

    for component in ("realtime", "batch"):
        script = _init_container_script(documents, component)
        assert f'GATEWAY="{STARTUP_SERVICE}"' in script
        assert "/v1/live" in script


def test_split_startup_service_follows_network_service_port() -> None:
    documents = _render(
        "--set",
        "topology.mode=split",
        "--set",
        "networkService.port=18080",
        "--set",
        "serviceMonitor.autoEnableInSplitMode=false",
    )

    startup = _services(documents)[STARTUP_SERVICE]
    assert startup["spec"]["ports"][0]["port"] == 18080
    assert startup["spec"]["ports"][0]["targetPort"] == "http"
    assert 'PORT="18080"' in _init_container_script(documents, "realtime")


def test_split_startup_service_is_not_scraped_by_the_gateway_service_monitor() -> None:
    documents = _render(
        "--set",
        "topology.mode=split",
        "--set",
        "serviceMonitor.autoEnableInSplitMode=true",
    )
    startup_labels = _services(documents)[STARTUP_SERVICE]["metadata"]["labels"]
    monitors = [document for document in documents if document.get("kind") == "ServiceMonitor"]

    assert len(monitors) == 3
    for monitor in monitors:
        selector = monitor["spec"]["selector"]["matchLabels"]
        assert not all(startup_labels.get(key) == value for key, value in selector.items())


def test_long_release_keeps_the_startup_service_distinct_and_referenced() -> None:
    documents = _render(
        "--set",
        "topology.mode=split",
        "--set",
        "serviceMonitor.autoEnableInSplitMode=false",
        "--set-string",
        f"fullnameOverride={'b' * 63}",
    )
    services = _services(documents)

    startup = next(
        name
        for name, service in services.items()
        if service["metadata"]["labels"].get("app.kubernetes.io/component") == "gateway-startup"
    )
    assert startup.endswith("-gateway-startup")
    assert len(startup) <= 63
    assert startup not in {
        name
        for name, service in services.items()
        if service["metadata"]["labels"].get("app.kubernetes.io/component") == "gateway"
    }
    assert f'GATEWAY="{startup}"' in _init_container_script(documents, "realtime")


def test_standalone_renders_no_startup_service_or_init_container() -> None:
    documents = _render()

    assert all(not service["spec"].get("publishNotReadyAddresses") for service in _services(documents).values())
    assert STARTUP_SERVICE not in _services(documents)
    assert all(
        "initContainers" not in deployment["spec"]["template"]["spec"] for deployment in _service_deployments(documents)
    )
