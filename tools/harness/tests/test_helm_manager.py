from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from nv_ingest_harness.service_manager import helm as helm_module
from nv_ingest_harness.service_manager.helm import HelmManager


def _manager(**overrides):
    config = SimpleNamespace(
        helm_bin="helm",
        helm_sudo=False,
        kubectl_bin="kubectl",
        kubectl_sudo=False,
        helm_release="nv-ingest",
        helm_namespace="nv-ingest",
        hostname="localhost",
        **overrides,
    )
    return HelmManager(config, Path("/repo"))


class HelmManagerTest(TestCase):
    def test_helm_health_url_uses_nemo_retriever_health_endpoint(self):
        manager = _manager()

        self.assertEqual(manager.get_service_url("health"), "http://localhost:7670/v1/health")

    def test_health_url_uses_configured_main_service_local_port(self):
        manager = _manager(
            helm_port_forwards=[
                {
                    "component": "service",
                    "local_port": 17670,
                    "remote_port": 7670,
                }
            ]
        )

        self.assertEqual(manager.get_service_url("health"), "http://localhost:17670/v1/health")

    def test_explicit_service_pattern_requires_service_to_exist(self):
        manager = _manager()

        def fake_run(cmd, capture_output, text, timeout):
            return SimpleNamespace(returncode=0, stdout="service/nv-ingest-nemo-retriever\n", stderr="")

        with patch.object(helm_module.subprocess, "run", fake_run):
            self.assertEqual(manager._find_services_by_pattern("nv-ingest"), [])
            self.assertEqual(
                manager._find_services_by_pattern("nv-ingest-nemo-retriever"),
                ["nv-ingest-nemo-retriever"],
            )

    def test_port_forward_component_resolves_nemo_retriever_service_by_release_label(self):
        manager = _manager(
            helm_port_forwards=[
                {
                    "component": "service",
                    "local_port": 7670,
                    "remote_port": 7670,
                }
            ]
        )
        commands = []

        def fake_run(cmd, capture_output, text, timeout):
            commands.append(cmd)
            return SimpleNamespace(returncode=0, stdout="service/nv-ingest-nemo-retriever\n", stderr="")

        started = []
        with patch.object(helm_module.subprocess, "run", fake_run), patch.object(
            manager,
            "_start_single_port_forward",
            lambda service_name, port_pairs: started.append((service_name, port_pairs)),
        ):
            manager._start_port_forwards()

        self.assertEqual(started, [("nv-ingest-nemo-retriever", [(7670, 7670)])])
        self.assertEqual(
            commands,
            [
                [
                    "kubectl",
                    "get",
                    "services",
                    "-n",
                    "nv-ingest",
                    "-l",
                    "app.kubernetes.io/instance=nv-ingest,app.kubernetes.io/component=service",
                    "-o",
                    "name",
                ]
            ],
        )
