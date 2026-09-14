# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default split Helm images must be documented and overridable for air-gap."""

from __future__ import annotations

from tests.test_helm_shared_results import CHART, _render, _service_deployments
from tests.test_helm_tracing_zipkin import _find, _helm_template_process

BUSYBOX = "busybox:1.37"
OTEL = "otel/opentelemetry-collector-contrib:0.127.0"
ZIPKIN = "openzipkin/zipkin:3.5.0"
README = CHART / "README.md"


def _split_render(*extra_args: str) -> list[dict]:
    return _render(
        "--set",
        "topology.mode=split",
        "--set",
        "serviceMonitor.autoEnableInSplitMode=false",
        *extra_args,
    )


def _rendered_images(documents: list[dict]) -> set[str]:
    images: set[str] = set()
    for document in documents:
        if document.get("kind") != "Deployment":
            continue
        spec = document["spec"]["template"]["spec"]
        for container in spec.get("initContainers", []) + spec.get("containers", []):
            image = container.get("image")
            if image:
                images.add(image)
    return images


def test_default_split_renders_auxiliary_images_listed_in_readme() -> None:
    readme = README.read_text(encoding="utf-8")
    images = _rendered_images(_split_render())

    for reference in (BUSYBOX, OTEL, ZIPKIN):
        assert reference in images
        assert reference in readme


def test_wait_for_gateway_image_is_overridable() -> None:
    documents = _split_render(
        "--set",
        "topology.waitForGateway.image.repository=registry.internal/busybox",
        "--set",
        "topology.waitForGateway.image.tag=airgap",
        "--set",
        "topology.waitForGateway.image.pullPolicy=Always",
    )
    deployments = _service_deployments(documents)
    workers = [
        item
        for item in deployments
        if item["metadata"]["labels"]["app.kubernetes.io/component"] in {"realtime", "batch"}
    ]
    assert len(workers) == 2
    for deployment in workers:
        init_container = next(
            item
            for item in deployment["spec"]["template"]["spec"]["initContainers"]
            if item["name"] == "wait-for-gateway"
        )
        assert init_container["image"] == "registry.internal/busybox:airgap"
        assert init_container["imagePullPolicy"] == "Always"


def test_null_wait_for_gateway_image_fails_in_split_mode() -> None:
    proc = _helm_template_process(
        extra_sets=["topology.mode=split"],
        extra_args=["--set-json", "topology.waitForGateway.image=null"],
    )
    assert proc.returncode != 0
    assert "topology.waitForGateway.image must be a map when topology.mode=split" in proc.stderr


def test_values_yaml_defines_busybox_repository_and_tag() -> None:
    values = (CHART / "values.yaml").read_text(encoding="utf-8")
    assert "waitForGateway:" in values
    assert "repository: busybox" in values
    assert 'tag: "1.37"' in values


def test_default_otel_and_zipkin_images_match_inventory() -> None:
    documents = _split_render()
    otel = _find(documents, "Deployment", "shared-results-test-nemo-retriever-otel")
    zipkin = _find(documents, "Deployment", "shared-results-test-nemo-retriever-zipkin")
    assert otel["spec"]["template"]["spec"]["containers"][0]["image"] == OTEL
    assert zipkin["spec"]["template"]["spec"]["containers"][0]["image"] == ZIPKIN
