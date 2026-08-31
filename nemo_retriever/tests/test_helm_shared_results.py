# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Helm wiring for the shared split-topology result store."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest import SkipTest

import yaml


CHART = Path(__file__).resolve().parents[1] / "helm"


def _render(*extra_args: str) -> list[dict]:
    helm = shutil.which("helm")
    if helm is None:
        raise SkipTest("`helm` binary not available in this environment.")

    command = [
        helm,
        "template",
        "shared-results-test",
        str(CHART),
        "--set",
        "nims.enabled=false",
        "--set",
        "serviceConfig.vectordb.enabled=false",
        *extra_args,
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return [document for document in yaml.safe_load_all(completed.stdout) if document]


def _service_deployments(documents: list[dict]) -> list[dict]:
    return [
        document
        for document in documents
        if document.get("kind") == "Deployment"
        and any(
            container.get("name") == "nemo-retriever"
            for container in document["spec"]["template"]["spec"].get("containers", [])
        )
    ]


def test_split_gateway_alone_mounts_and_configures_result_claim() -> None:
    documents = _render(
        "--set",
        "topology.mode=split",
        "--set",
        "serviceMonitor.autoEnableInSplitMode=false",
    )
    deployments = _service_deployments(documents)

    assert len(deployments) == 3
    components: set[str] = set()
    for deployment in deployments:
        component = deployment["metadata"]["labels"]["app.kubernetes.io/component"]
        components.add(component)
        pod_spec = deployment["spec"]["template"]["spec"]
        container = next(item for item in pod_spec["containers"] if item["name"] == "nemo-retriever")
        env = {item["name"]: item.get("value") for item in container["env"]}
        mounts = {item["name"]: item["mountPath"] for item in container["volumeMounts"]}
        volumes = {item["name"]: item for item in pod_spec["volumes"]}

        assert env["NEMO_RETRIEVER_RESULTS_TTL_SECONDS"] == "28800"
        if component == "gateway":
            assert env["NEMO_RETRIEVER_RESULTS_DIR"] == "/retriever_results"
            assert mounts["retriever-results"] == "/retriever_results"
            assert "persistentVolumeClaim" in volumes["retriever-results"]
        else:
            assert "NEMO_RETRIEVER_RESULTS_DIR" not in env
            assert "retriever-results" not in mounts
            assert "retriever-results" not in volumes

    assert components == {"gateway", "realtime", "batch"}
    result_claim = next(
        document
        for document in documents
        if document.get("kind") == "PersistentVolumeClaim"
        and document["metadata"]["name"].endswith("-retriever-results")
    )
    assert result_claim["spec"]["accessModes"] == ["ReadWriteOnce"]


def test_disabled_shared_results_are_not_wired() -> None:
    documents = _render("--set", "retrieverResults.enabled=false")
    deployment = _service_deployments(documents)[0]
    pod_spec = deployment["spec"]["template"]["spec"]
    container = next(item for item in pod_spec["containers"] if item["name"] == "nemo-retriever")

    assert all(item["name"] != "NEMO_RETRIEVER_RESULTS_DIR" for item in container["env"])
    assert all(item["name"] != "NEMO_RETRIEVER_RESULTS_TTL_SECONDS" for item in container["env"])
    assert all(item["name"] != "retriever-results" for item in container["volumeMounts"])
    assert all(item["name"] != "retriever-results" for item in pod_spec["volumes"])
