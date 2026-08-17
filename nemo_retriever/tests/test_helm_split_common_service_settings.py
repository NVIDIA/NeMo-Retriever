# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common service settings must apply to every split-topology workload."""

from __future__ import annotations

from tests.test_helm_shared_results import _render, _service_deployments


def _main_container(deployment: dict) -> dict:
    return next(
        container
        for container in deployment["spec"]["template"]["spec"]["containers"]
        if container["name"] == "nemo-retriever"
    )


def test_split_roles_inherit_common_service_settings() -> None:
    security_context = {
        "allowPrivilegeEscalation": False,
        "capabilities": {"drop": ["ALL"]},
        "runAsNonRoot": True,
        "seccompProfile": {"type": "RuntimeDefault"},
    }
    extra_volume = {"name": "common-settings", "emptyDir": {}}
    extra_mount = {"name": "common-settings", "mountPath": "/common-settings", "readOnly": True}
    spread_constraint = {
        "labelSelector": {"matchLabels": {"example.com/common": "enabled"}},
        "maxSkew": 1,
        "topologyKey": "kubernetes.io/hostname",
        "whenUnsatisfiable": "ScheduleAnyway",
    }

    documents = _render(
        "--set",
        "topology.mode=split",
        "--set",
        "service.priorityClassName=common-priority",
        "--set",
        "service.terminationGracePeriodSeconds=77",
        "--set-json",
        'service.podLabels={"example.com/common":"enabled"}',
        "--set-json",
        'service.podAnnotations={"example.com/common":"annotation"}',
        "--set-json",
        (
            'service.securityContext={"runAsNonRoot":true,'
            '"allowPrivilegeEscalation":false,'
            '"capabilities":{"drop":["ALL"]},'
            '"seccompProfile":{"type":"RuntimeDefault"}}'
        ),
        "--set-json",
        'service.topologySpreadConstraints=[{"maxSkew":1,'
        '"topologyKey":"kubernetes.io/hostname",'
        '"whenUnsatisfiable":"ScheduleAnyway",'
        '"labelSelector":{"matchLabels":{"example.com/common":"enabled"}}}]',
        "--set-json",
        'service.envFrom=[{"configMapRef":{"name":"common-settings"}}]',
        "--set-json",
        'service.extraVolumes=[{"name":"common-settings","emptyDir":{}}]',
        "--set-json",
        'service.extraVolumeMounts=[{"name":"common-settings",'
        '"mountPath":"/common-settings","readOnly":true}]',
    )

    deployments = _service_deployments(documents)
    assert len(deployments) == 3

    observed_roles = set()
    for deployment in deployments:
        role = deployment["metadata"]["labels"]["app.kubernetes.io/component"]
        observed_roles.add(role)
        pod_template = deployment["spec"]["template"]
        pod_spec = pod_template["spec"]
        container = _main_container(deployment)

        assert pod_template["metadata"]["labels"]["example.com/common"] == "enabled"
        assert pod_template["metadata"]["annotations"]["example.com/common"] == "annotation"
        assert pod_spec["priorityClassName"] == "common-priority"
        assert pod_spec["terminationGracePeriodSeconds"] == 77
        assert pod_spec["topologySpreadConstraints"] == [spread_constraint]
        assert container["securityContext"] == security_context
        assert container["envFrom"] == [{"configMapRef": {"name": "common-settings"}}]
        assert extra_volume in pod_spec["volumes"]
        assert extra_mount in container["volumeMounts"]

        if role == "gateway":
            assert "initContainers" not in pod_spec
        else:
            assert pod_spec["initContainers"] == [
                {
                    "name": "wait-for-gateway",
                    "image": "busybox:1.37",
                    "securityContext": security_context,
                    "command": pod_spec["initContainers"][0]["command"],
                }
            ]

    assert observed_roles == {"gateway", "realtime", "batch"}
