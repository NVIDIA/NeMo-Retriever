from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_DIR = REPO_ROOT / "nemo_retriever" / "helm"
PROFILE_DIR = REPO_ROOT / "nemo_retriever" / "harness" / "helm-profiles"


@pytest.mark.parametrize("profile", ["core", "no-nims-external", "all-optional", "audio-video", "split"])
def test_managed_helm_profiles_are_named_values_files(profile: str) -> None:
    path = PROFILE_DIR / f"{profile}.yaml"

    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert isinstance(data, dict)
    rendered_source = path.read_text(encoding="utf-8")
    assert "docker compose" not in rendered_source.lower()
    assert "nv-ingest-ms-runtime" not in rendered_source


def _helm_command() -> list[str]:
    helm = shutil.which("helm")
    if helm is not None:
        return [helm]
    microk8s = shutil.which("microk8s")
    if microk8s is not None:
        return [microk8s, "helm"]
    pytest.skip("helm binary is not available")


def _helm_template(profile: str) -> str:
    result = subprocess.run(
        [
            *_helm_command(),
            "template",
            "nrl-test",
            str(CHART_DIR),
            "--api-versions",
            "apps.nvidia.com/v1alpha1",
            "--api-versions",
            "monitoring.coreos.com/v1",
            "-f",
            str(PROFILE_DIR / f"{profile}.yaml"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.parametrize("profile", ["core", "all-optional", "audio-video", "split"])
def test_managed_helm_profiles_render(profile: str) -> None:
    manifest = _helm_template(profile)

    assert "kind: Deployment" in manifest
    assert "nv-ingest-ms-runtime" not in manifest
    if profile == "core":
        assert "name: nemotron-page-elements-v3" in manifest
        assert "name: llama-nemotron-rerank-1b-v2" not in manifest
    elif profile == "all-optional":
        assert "name: llama-nemotron-rerank-1b-v2" in manifest
        assert "name: nemotron-parse" in manifest
        assert "name: audio" in manifest
        assert "app.kubernetes.io/component: otel" in manifest
    elif profile == "audio-video":
        assert "name: INSTALL_FFMPEG" in manifest
        assert 'value: "true"' in manifest
        assert "name: audio" in manifest
    elif profile == "split":
        assert "app.kubernetes.io/component: gateway" in manifest
        assert "app.kubernetes.io/component: realtime" in manifest
        assert "app.kubernetes.io/component: batch" in manifest
        assert "kind: HorizontalPodAutoscaler" in manifest
        assert "kind: ServiceMonitor" in manifest
