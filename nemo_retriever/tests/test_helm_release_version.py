# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for the Helm and service 26.8.2 release alignment."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from unittest import SkipTest, TestCase, main

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHART_DIR = _REPO_ROOT / "nemo_retriever/helm"
_CHART_YAML = _CHART_DIR / "Chart.yaml"
_VALUES_YAML = _CHART_DIR / "values.yaml"
_README_MD = _CHART_DIR / "README.md"
_OPENSHIFT_MD = _CHART_DIR / "openshift.md"
_RELEASE_VERSION = "26.8.2"
_SERVICE_IMAGE = f"nvcr.io/nvidia/nemo-microservices/nrl-service:{_RELEASE_VERSION}"
_MINIMAL_INSTALL_ANCHOR = "#recommended-minimal-install-2682"


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise SkipTest(f"Required file not present in this test environment: {path}")
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise SkipTest(f"Required file not present in this test environment: {path}")
    return path.read_text(encoding="utf-8")


def _helm_template(topology: str) -> subprocess.CompletedProcess[str]:
    helm = shutil.which("helm")
    if helm is None:
        raise SkipTest("`helm` binary not available in this environment.")
    if not _CHART_DIR.is_dir():
        raise SkipTest(f"Chart directory missing: {_CHART_DIR}")
    cmd = [
        helm,
        "template",
        "nrl-release-version",
        str(_CHART_DIR),
        "--set",
        f"topology.mode={topology}",
        "--set",
        "ngcImagePullSecret.create=false",
        "--set",
        "ngcApiSecret.create=false",
        "--set",
        "serviceConfig.vectordb.enabled=false",
    ]
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _service_containers(docs: list[dict]) -> list[dict]:
    containers: list[dict] = []
    for doc in docs:
        if not doc or doc.get("kind") != "Deployment":
            continue
        pod_spec = doc["spec"]["template"]["spec"]
        containers.extend(
            container for container in pod_spec["containers"] if container.get("name") == "nemo-retriever"
        )
    return containers


def _env_value(container: dict, name: str) -> str | None:
    for item in container.get("env") or []:
        if item.get("name") == name:
            return item.get("value")
    return None


class TestHelmReleaseVersion(TestCase):
    def test_chart_metadata_and_default_service_tag_are_aligned(self) -> None:
        chart = _read_yaml(_CHART_YAML)
        values = _read_yaml(_VALUES_YAML)

        self.assertEqual(chart["version"], _RELEASE_VERSION)
        self.assertEqual(chart["appVersion"], _RELEASE_VERSION)
        self.assertEqual(values["service"]["image"]["tag"], _RELEASE_VERSION)

    def _assert_default_render(self, topology: str, minimum_containers: int) -> None:
        proc = _helm_template(topology)
        self.assertEqual(
            proc.returncode,
            0,
            f"`helm template` failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
        )
        containers = _service_containers(list(yaml.safe_load_all(proc.stdout)))
        self.assertGreaterEqual(len(containers), minimum_containers)
        for container in containers:
            self.assertEqual(container["image"], _SERVICE_IMAGE)
            self.assertEqual(_env_value(container, "RETRIEVER_SERVICE_VERSION"), _RELEASE_VERSION)

    def test_default_standalone_render_uses_release_service_version(self) -> None:
        self._assert_default_render("standalone", minimum_containers=1)

    def test_default_split_render_uses_release_service_version(self) -> None:
        self._assert_default_render("split", minimum_containers=2)

    def test_readme_uses_release_image_and_minimal_install_anchor(self) -> None:
        readme = _read_text(_README_MD)

        self.assertIn(_SERVICE_IMAGE, readme)
        self.assertRegex(
            readme,
            rf"\|\s*`service\.image\.tag`\s*\|\s*`{re.escape(_RELEASE_VERSION)}`\s*\|",
        )
        self.assertIn(f"--set service.image.tag={_RELEASE_VERSION}", readme)
        self.assertNotIn("nrl-service:26.5.0", readme)
        self.assertNotIn("--set service.image.tag=26.8.1", readme)
        self.assertIn(
            f"### Recommended minimal install ({_RELEASE_VERSION}) {{ {_MINIMAL_INSTALL_ANCHOR} }}",
            readme,
        )
        self.assertEqual(readme.count(f"({_MINIMAL_INSTALL_ANCHOR})"), 5)
        self.assertNotIn("#recommended-minimal-install-2608", readme)
        self.assertNotIn("#recommended-minimal-install-26081", readme)

    def test_openshift_examples_use_release_service_tag(self) -> None:
        openshift = _read_text(_OPENSHIFT_MD)

        self.assertEqual(openshift.count(f"--set service.image.tag={_RELEASE_VERSION}"), 2)
        self.assertNotIn("--set service.image.tag=26.8.1", openshift)


if __name__ == "__main__":
    main()
