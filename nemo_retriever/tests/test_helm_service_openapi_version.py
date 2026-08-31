# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helm must inject RETRIEVER_SERVICE_VERSION from the service image tag."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence
from unittest import SkipTest, TestCase, main

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHART_DIR = _REPO_ROOT / "nemo_retriever/helm"


def _helm_template(extra_args: Sequence[str] = ()) -> subprocess.CompletedProcess[str]:
    helm = shutil.which("helm")
    if helm is None:
        raise SkipTest("`helm` binary not available in this environment.")
    if not _CHART_DIR.is_dir():
        raise SkipTest(f"Chart directory missing: {_CHART_DIR}")
    cmd = [
        helm,
        "template",
        "nrl-openapi-version",
        str(_CHART_DIR),
        "--set",
        "ngcImagePullSecret.create=false",
        "--set",
        "ngcApiSecret.create=false",
        "--set",
        "serviceConfig.vectordb.enabled=false",
        *extra_args,
    ]
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _deployments(docs: list[dict]) -> list[dict]:
    return [doc for doc in docs if doc and doc.get("kind") == "Deployment"]


def _nemo_retriever_containers(docs: list[dict]) -> list[dict]:
    containers: list[dict] = []
    for deployment in _deployments(docs):
        for container in deployment["spec"]["template"]["spec"]["containers"]:
            if container.get("name") == "nemo-retriever":
                containers.append(container)
    return containers


def _env_value(container: dict, name: str) -> str | None:
    for item in container.get("env") or []:
        if item.get("name") == name:
            return item.get("value")
    return None


class TestHelmServiceOpenApiVersion(TestCase):
    def test_standalone_injects_image_tag_as_service_version(self) -> None:
        proc = _helm_template(
            (
                "--set",
                "topology.mode=standalone",
                "--set",
                "service.image.tag=26.08.1",
            )
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"`helm template` failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
        )
        docs = list(yaml.safe_load_all(proc.stdout))
        containers = _nemo_retriever_containers(docs)
        self.assertEqual(len(containers), 1)
        self.assertEqual(_env_value(containers[0], "RETRIEVER_SERVICE_VERSION"), "26.08.1")

    def test_split_injects_image_tag_on_gateway_and_workers(self) -> None:
        proc = _helm_template(
            (
                "--set",
                "topology.mode=split",
                "--set",
                "service.image.tag=26.08.1",
            )
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"`helm template` failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
        )
        docs = list(yaml.safe_load_all(proc.stdout))
        containers = _nemo_retriever_containers(docs)
        self.assertGreaterEqual(len(containers), 2)
        for container in containers:
            self.assertEqual(_env_value(container, "RETRIEVER_SERVICE_VERSION"), "26.08.1")


if __name__ == "__main__":
    main()
