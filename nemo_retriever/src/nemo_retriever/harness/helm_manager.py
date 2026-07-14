# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
import shlex
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

from nemo_retriever.harness.config import NEMO_RETRIEVER_ROOT


logger = logging.getLogger(__name__)


class HelmServiceManager:
    """Manage a nemo-retriever service deployment with Helm for harness runs."""

    MAIN_SERVICE_COMPONENTS = ("service", "gateway")
    SPLIT_COMPONENTS = ("gateway", "realtime", "batch")
    SENSITIVE_KEY_PARTS = ("password", "token", "secret", "api_key", "apikey", "dockerconfigjson")
    SENSITIVE_FLAGS = ("--password", "--token", "--api-key", "--api_key")
    NIM_OPERATOR_RESOURCES = (
        "nemotron-page-elements-v3",
        "nemotron-table-structure-v1",
        "nemotron-ocr-v2",
        "llama-nemotron-embed-vl-1b-v2",
        "llama-nemotron-rerank-1b-v2",
        "nemotron-parse",
        "nemotron-3-nano-omni-30b-a3b-reasoning",
        "audio",
    )

    def __init__(self, config: Any, repo_root: Path | None = None) -> None:
        self.config = config
        self.repo_root = repo_root or NEMO_RETRIEVER_ROOT.parent
        self.release_name = config.helm_release
        self.namespace = config.helm_namespace or config.helm_release
        self.chart_ref = config.helm_chart or str(NEMO_RETRIEVER_ROOT / "helm")
        self.local_port = int(config.helm_service_local_port)
        self.remote_port = 7670
        self.port_forward_processes: list[subprocess.Popen] = []
        self._forwarded_service_name: str | None = None
        self._forwarded_component: str | None = None

        self.helm_cmd = self._command(config.helm_bin, sudo=config.helm_sudo)
        self.kubectl_cmd = self._command(config.kubectl_bin, sudo=config.kubectl_sudo)

    @staticmethod
    def _command(raw: str, *, sudo: bool) -> list[str]:
        cmd = shlex.split(raw or "")
        if not cmd:
            raise ValueError("command cannot be empty")
        return (["sudo"] if sudo else []) + cmd

    @staticmethod
    def _helm_set_arg(key: str, value: Any) -> tuple[str, str]:
        if isinstance(value, (list, dict)):
            return "--set-json", f"{key}={json.dumps(value, separators=(',', ':'))}"
        if isinstance(value, bool):
            return "--set", f"{key}={str(value).lower()}"
        if value is None:
            return "--set-json", f"{key}=null"
        if isinstance(value, (int, float)):
            return "--set", f"{key}={value}"
        escaped = str(value).replace("\\", "\\\\").replace(",", "\\,")
        return "--set", f"{key}={escaped}"

    @classmethod
    def _redact_command_part(cls, part: str) -> str:
        if "=" not in part:
            return part
        key, _value = part.split("=", 1)
        normalized = key.lower().replace("-", "_").replace(".", "_")
        if any(marker in normalized for marker in cls.SENSITIVE_KEY_PARTS):
            return f"{key}=<redacted>"
        return part

    def format_command(self, cmd: list[str]) -> str:
        display_parts: list[str] = []
        redact_next = False
        for part in cmd:
            if redact_next:
                display_parts.append("<redacted>")
                redact_next = False
                continue
            display_part = self._redact_command_part(part)
            display_parts.append(shlex.quote(display_part))
            if part in self.SENSITIVE_FLAGS:
                redact_next = True
        return " ".join(display_parts)

    def build_upgrade_command(self) -> list[str]:
        cmd = self.helm_cmd + [
            "upgrade",
            "--install",
            self.release_name,
            self.chart_ref,
            "--namespace",
            self.namespace,
            "--create-namespace",
            "--wait",
            "--timeout",
            f"{int(self.config.helm_timeout)}s",
        ]

        if self.config.helm_chart_version:
            cmd += ["--version", self.config.helm_chart_version]

        if self.config.helm_values_file:
            cmd += ["-f", self.config.helm_values_file]

        effective_set = getattr(self.config, "effective_helm_set", None)
        helm_set = effective_set() if callable(effective_set) else self.config.helm_set
        for key in sorted(helm_set):
            flag, assignment = self._helm_set_arg(key, helm_set[key])
            cmd += [flag, assignment]

        return cmd

    def start(self) -> int:
        rc = self._run(self.build_upgrade_command())
        if rc != 0:
            return rc

        service_name = self.resolve_main_service_name(timeout_s=int(self.config.readiness_timeout))
        if not service_name:
            logger.warning("Readiness timeout. Main service/gateway was not created.")
            return 1

        try:
            self.start_port_forward(service_name)
        except RuntimeError as exc:
            logger.warning("%s", exc)
            return 1

        if not self.check_readiness(timeout_s=int(self.config.readiness_timeout)):
            return 1

        if not self.wait_for_chart_pods(timeout_s=int(self.config.readiness_timeout)):
            return 1

        if not self.wait_for_optional_resources(timeout_s=int(self.config.readiness_timeout)):
            return 1

        return 0

    def stop(self, *, uninstall: bool = True) -> int:
        self.stop_port_forwards()
        if not uninstall:
            return 0
        return self._run(self.helm_cmd + ["uninstall", self.release_name, "--namespace", self.namespace])

    def _run(self, cmd: list[str]) -> int:
        logger.info("$ %s", self.format_command(cmd))
        return subprocess.run(cmd).returncode

    def _selector_for_component(self, component: str) -> str:
        return f"app.kubernetes.io/instance={self.release_name},app.kubernetes.io/component={component}"

    def find_services_by_component(self, component: str) -> list[str]:
        selector = self._selector_for_component(component)
        cmd = self.kubectl_cmd + ["get", "services", "-n", self.namespace, "-l", selector, "-o", "name"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return []
        return [line.split("/", 1)[-1] for line in result.stdout.splitlines() if line.strip()]

    def wait_for_services_by_component(self, component: str, *, timeout_s: int, interval_s: int = 5) -> list[str]:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            services = self.find_services_by_component(component)
            if services:
                return services
            time.sleep(interval_s)
        return []

    def resolve_main_service_name(self, *, timeout_s: int, interval_s: int = 5) -> str | None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            for component in self.MAIN_SERVICE_COMPONENTS:
                services = self.find_services_by_component(component)
                if services:
                    self._forwarded_component = component
                    return services[0]
            time.sleep(interval_s)
        return None

    def start_port_forward(
        self, service_name: str, *, local_port: int | None = None, remote_port: int | None = None
    ) -> None:
        local = int(local_port or self.local_port)
        remote = int(remote_port or self.remote_port)
        self._forwarded_service_name = service_name
        cmd = self.kubectl_cmd + [
            "port-forward",
            "-n",
            self.namespace,
            f"service/{service_name}",
            f"{local}:{remote}",
        ]
        logger.info("$ %s (background)", self.format_command(cmd))
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
        )
        self.port_forward_processes.append(proc)
        time.sleep(2)
        if proc.poll() is not None:
            _, stderr = proc.communicate()
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"kubectl port-forward failed for {service_name}: {detail}")

    def stop_port_forwards(self) -> None:
        for proc in self.port_forward_processes:
            try:
                pgid = os.getpgid(proc.pid)
            except ProcessLookupError:
                continue
            try:
                os.killpg(pgid, signal.SIGTERM)
                proc.wait(timeout=5)
            except PermissionError as exc:
                logger.warning("Could not signal port-forward process group %s for pid %s: %s", pgid, proc.pid, exc)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except PermissionError as exc:
                    logger.warning(
                        "Could not force-kill port-forward process group %s for pid %s: %s", pgid, proc.pid, exc
                    )
                except ProcessLookupError:
                    pass
            except ProcessLookupError:
                pass
        self.port_forward_processes = []

    def get_service_url(self, service: str = "api") -> str:
        base = f"http://localhost:{self.local_port}"
        if service == "health":
            return f"{base}/v1/health"
        return base

    def _poll_http_200(self, url: str, *, timeout_s: int, interval_s: int = 3) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if resp.status == 200:
                        return True
            except OSError:
                pass
            time.sleep(interval_s)
        logger.warning("Readiness timeout. %s did not return HTTP 200.", url)
        return False

    def check_readiness(self, *, timeout_s: int, interval_s: int = 3) -> bool:
        return self._poll_http_200(self.get_service_url("health"), timeout_s=timeout_s, interval_s=interval_s)

    def wait_for_chart_pods(self, *, timeout_s: int) -> bool:
        selector = f"app.kubernetes.io/instance={self.release_name}"
        cmd = self.kubectl_cmd + [
            "wait",
            "--for=condition=Ready",
            "pod",
            "-n",
            self.namespace,
            "-l",
            selector,
            f"--timeout={int(timeout_s)}s",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s + 30)
        if result.returncode != 0:
            logger.warning("Chart pod readiness wait returned %s: %s", result.returncode, result.stderr.strip())
            return False
        return True

    def wait_for_optional_resources(self, *, timeout_s: int) -> bool:
        ok = True

        vectordb_services = self.find_services_by_component("vectordb")
        if vectordb_services:
            vdb_local_port = self.local_port + 1
            try:
                self.start_port_forward(vectordb_services[0], local_port=vdb_local_port, remote_port=7671)
                ok = self._poll_http_200(f"http://localhost:{vdb_local_port}/v1/health", timeout_s=timeout_s) and ok
            except RuntimeError as exc:
                logger.warning("VectorDB health check could not start: %s", exc)
                ok = False

        if not self._crd_exists("nimservices.apps.nvidia.com"):
            return ok

        nimcache_crd_exists = self._crd_exists("nimcaches.apps.nvidia.com")
        for name in self.NIM_OPERATOR_RESOURCES:
            if nimcache_crd_exists:
                ok = (
                    self._wait_for_optional_k8s_resource(
                        "nimcache",
                        name,
                        condition="NIM_CACHE_JOB_COMPLETED",
                        timeout_s=timeout_s,
                    )
                    and ok
                )
            ok = self._wait_for_optional_k8s_resource("nimservice", name, timeout_s=timeout_s) and ok
        return ok

    def _crd_exists(self, name: str) -> bool:
        result = subprocess.run(self.kubectl_cmd + ["get", "crd", name], capture_output=True, text=True, timeout=30)
        return result.returncode == 0

    def _wait_for_optional_k8s_resource(
        self,
        kind: str,
        name: str,
        *,
        condition: str = "Ready",
        timeout_s: int,
    ) -> bool:
        get_cmd = self.kubectl_cmd + ["get", kind, name, "-n", self.namespace]
        exists = subprocess.run(get_cmd, capture_output=True, text=True, timeout=30)
        if exists.returncode != 0:
            return True

        wait_cmd = self.kubectl_cmd + [
            "wait",
            f"--for=condition={condition}",
            kind,
            name,
            "-n",
            self.namespace,
            f"--timeout={int(timeout_s)}s",
        ]
        waited = subprocess.run(wait_cmd, capture_output=True, text=True, timeout=timeout_s + 30)
        if waited.returncode != 0:
            logger.warning("%s %s did not become Ready: %s", kind, name, waited.stderr.strip())
            return False
        return True

    def dump_logs(self, artifacts_dir: Path) -> int:
        logs_dir = Path(artifacts_dir) / "service_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        pod_cmd = self.kubectl_cmd + [
            "get",
            "pods",
            "-n",
            self.namespace,
            "-l",
            f"app.kubernetes.io/instance={self.release_name}",
            "-o",
            "name",
        ]
        pods = subprocess.run(pod_cmd, capture_output=True, text=True, timeout=30)
        if pods.returncode != 0:
            (logs_dir / "kubectl_get_pods.err").write_text(pods.stderr, encoding="utf-8")
            return pods.returncode

        for pod_ref in [line.strip() for line in pods.stdout.splitlines() if line.strip()]:
            pod = pod_ref.split("/", 1)[-1]
            log_cmd = self.kubectl_cmd + ["logs", pod, "-n", self.namespace, "--all-containers", "--tail=-1"]
            result = subprocess.run(log_cmd, capture_output=True, text=True, timeout=120)
            (logs_dir / f"{pod}.log").write_text(result.stdout, encoding="utf-8")
            if result.stderr:
                (logs_dir / f"{pod}.err").write_text(result.stderr, encoding="utf-8")
        return 0
