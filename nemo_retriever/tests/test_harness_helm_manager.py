# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_retriever.harness.config import HarnessConfig
from nemo_retriever.harness.helm_manager import HelmServiceManager


def _managed_cfg(tmp_path: Path, **overrides) -> HarnessConfig:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(exist_ok=True)
    values_file = tmp_path / "values.yaml"
    values_file.write_text("nims:\n  enabled: false\n", encoding="utf-8")
    kwargs = {
        "dataset_dir": str(dataset_dir),
        "dataset_label": "tiny",
        "preset": "base",
        "run_mode": "service",
        "manage_service": True,
        "helm_chart": "nim-nvstaging/nemo-retriever",
        "helm_chart_version": "26.05-RC6",
        "helm_release": "nrl-smoke",
        "helm_namespace": "nrl-smoke-ns",
        "helm_values_file": str(values_file),
        "helm_set": {
            "service.image.repository": "nvcr.io/nvstaging/nim/nrl-service",
            "service.image.tag": "26.05-RC6",
            "ngcApiSecret.create": True,
            "service.env": [{"name": "PYTHONFAULTHANDLER", "value": "1"}],
        },
        "helm_bin": "microk8s helm",
        "kubectl_bin": "microk8s kubectl",
        "helm_sudo": True,
        "kubectl_sudo": True,
        "helm_timeout": 900,
        "helm_service_local_port": 17670,
    }
    kwargs.update(overrides)
    return HarnessConfig(**kwargs)


def test_build_upgrade_command_supports_remote_chart_version_values_and_inline_sets(tmp_path: Path) -> None:
    cfg = _managed_cfg(tmp_path)
    manager = HelmServiceManager(cfg)

    cmd = manager.build_upgrade_command()

    assert cmd[:7] == [
        "sudo",
        "microk8s",
        "helm",
        "upgrade",
        "--install",
        "nrl-smoke",
        "nim-nvstaging/nemo-retriever",
    ]
    assert "--namespace" in cmd
    assert cmd[cmd.index("--namespace") + 1] == "nrl-smoke-ns"
    assert "--version" in cmd
    assert cmd[cmd.index("--version") + 1] == "26.05-RC6"
    assert "-f" in cmd
    assert cmd[cmd.index("-f") + 1] == cfg.helm_values_file
    assert "--timeout" in cmd
    assert cmd[cmd.index("--timeout") + 1] == "900s"
    assert "--set" in cmd
    assert "service.image.repository=nvcr.io/nvstaging/nim/nrl-service" in cmd
    assert "service.image.tag=26.05-RC6" in cmd
    assert "ngcApiSecret.create=true" in cmd
    assert "--set-json" in cmd
    assert 'service.env=[{"name":"PYTHONFAULTHANDLER","value":"1"}]' in cmd


def test_service_discovery_uses_component_label_selector(monkeypatch, tmp_path: Path) -> None:
    cfg = _managed_cfg(tmp_path)
    manager = HelmServiceManager(cfg)
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="service/nrl-smoke-nemo-retriever\n", stderr="")

    import nemo_retriever.harness.helm_manager as helm_manager

    monkeypatch.setattr(helm_manager.subprocess, "run", fake_run)

    services = manager.find_services_by_component("service")

    assert services == ["nrl-smoke-nemo-retriever"]
    assert calls
    assert "nv-ingest" not in " ".join(calls[0])
    assert "-l" in calls[0]
    selector = calls[0][calls[0].index("-l") + 1]
    assert selector == "app.kubernetes.io/instance=nrl-smoke,app.kubernetes.io/component=service"


def test_service_urls_use_forwarded_local_port_and_health_path(tmp_path: Path) -> None:
    cfg = _managed_cfg(tmp_path, helm_service_local_port=17670)
    manager = HelmServiceManager(cfg)

    assert manager.get_service_url() == "http://localhost:17670"
    assert manager.get_service_url("health") == "http://localhost:17670/v1/health"


def test_readiness_polling_does_not_swallow_unexpected_errors(monkeypatch, tmp_path: Path) -> None:
    cfg = _managed_cfg(tmp_path)
    manager = HelmServiceManager(cfg)

    import nemo_retriever.harness.helm_manager as helm_manager

    monkeypatch.setattr(
        helm_manager.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad url")),
    )
    monkeypatch.setattr(helm_manager.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(helm_manager.time, "time", iter([0.0, 0.0, 2.0]).__next__)

    with pytest.raises(ValueError, match="bad url"):
        manager._poll_http_200("http://localhost:17670/v1/health", timeout_s=1)


def test_main_service_resolution_falls_back_to_gateway_for_split_topology(monkeypatch, tmp_path: Path) -> None:
    cfg = _managed_cfg(tmp_path)
    manager = HelmServiceManager(cfg)
    selectors: list[str] = []

    def fake_find(component: str) -> list[str]:
        selectors.append(component)
        if component == "gateway":
            return ["nrl-smoke-nemo-retriever-gateway"]
        return []

    monkeypatch.setattr(manager, "find_services_by_component", fake_find)

    assert manager.resolve_main_service_name(timeout_s=1, interval_s=0) == "nrl-smoke-nemo-retriever-gateway"
    assert selectors[:2] == ["service", "gateway"]


def test_display_command_redacts_secret_inline_values(tmp_path: Path) -> None:
    cfg = _managed_cfg(
        tmp_path,
        helm_set={
            "ngcApiSecret.password": "super-secret-token",
            "ngcImagePullSecret.dockerconfigjson": "encoded-secret",
            "service.image.tag": "26.05-RC6",
        },
    )
    manager = HelmServiceManager(cfg)

    display = manager.format_command(manager.build_upgrade_command())

    assert "super-secret-token" not in display
    assert "encoded-secret" not in display
    assert "ngcApiSecret.password=<redacted>" in display
    assert "ngcImagePullSecret.dockerconfigjson=<redacted>" in display
    assert "service.image.tag=26.05-RC6" in display


def test_stop_uninstalls_release_when_port_forward_signal_is_denied(monkeypatch, tmp_path: Path) -> None:
    cfg = _managed_cfg(tmp_path)
    manager = HelmServiceManager(cfg)
    fake_proc = SimpleNamespace(pid=12345)
    manager.port_forward_processes = [fake_proc]
    calls: dict[str, object] = {}

    import nemo_retriever.harness.helm_manager as helm_manager

    monkeypatch.setattr(helm_manager.os, "getpgid", lambda _pid: 67890)

    def fake_killpg(_pgid, _signal):
        raise PermissionError("operation not permitted")

    def fake_run(cmd, **_kwargs):
        calls["uninstall"] = cmd
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(helm_manager.os, "killpg", fake_killpg)
    monkeypatch.setattr(helm_manager.subprocess, "run", fake_run)

    assert manager.stop() == 0
    assert calls["uninstall"] == ["sudo", "microk8s", "helm", "uninstall", "nrl-smoke", "--namespace", "nrl-smoke-ns"]
    assert manager.port_forward_processes == []


def test_optional_nimcache_wait_uses_completed_condition(monkeypatch, tmp_path: Path) -> None:
    cfg = _managed_cfg(tmp_path)
    manager = HelmServiceManager(cfg)
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        joined = " ".join(cmd)
        if "get crd" in joined:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "get nimcache nemotron-page-elements-v3" in joined:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "get nimservice" in joined or "get nimcache" in joined:
            return SimpleNamespace(returncode=1, stdout="", stderr="not found")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    import nemo_retriever.harness.helm_manager as helm_manager

    monkeypatch.setattr(manager, "find_services_by_component", lambda _component: [])
    monkeypatch.setattr(helm_manager.subprocess, "run", fake_run)

    assert manager.wait_for_optional_resources(timeout_s=1) is True
    wait_cmds = [cmd for cmd in calls if "wait" in cmd and "nimcache" in cmd]
    assert wait_cmds
    assert "--for=condition=NIM_CACHE_JOB_COMPLETED" in wait_cmds[0]
