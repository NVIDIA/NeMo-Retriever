# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kubernetes cluster executor for dispatching harness jobs to remote K8s clusters."""

from __future__ import annotations

import base64
import json
import logging
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_JOB_IMAGE = "nvcr.io/nvidia/nemo-retriever:latest"
JOB_TTL_SECONDS = 3600
JOB_ACTIVE_DEADLINE_SECONDS = 7200
BACKOFF_LIMIT = 0


def _get_k8s_client(cluster: dict[str, Any]) -> tuple[Any, str]:
    """Build a Kubernetes API client from cluster connection details.

    Returns (ApiClient, namespace).
    """
    try:
        from kubernetes import client as k8s_client
        from kubernetes.client import Configuration
    except ImportError as exc:
        raise ImportError(
            "The 'kubernetes' package is required for cluster execution. "
            "Install it with: pip install kubernetes"
        ) from exc

    namespace = cluster.get("namespace", "default")
    auth_method = cluster.get("auth_method", "kubeconfig")

    if auth_method == "kubeconfig" and cluster.get("kubeconfig_data"):
        import yaml

        kubeconfig = yaml.safe_load(cluster["kubeconfig_data"])
        context_name = cluster.get("kubeconfig_context")

        from kubernetes.config.kube_config import KubeConfigLoader

        loader = KubeConfigLoader(
            config_dict=kubeconfig,
            active_context=context_name,
        )
        config = Configuration()
        loader.load_and_set(config)
        api_client = k8s_client.ApiClient(configuration=config)

    elif auth_method == "token" and cluster.get("service_account_token"):
        config = Configuration()
        config.host = cluster["api_server_url"]
        config.api_key = {"authorization": f"Bearer {cluster['service_account_token']}"}

        if cluster.get("ca_cert_data"):
            ca_file = tempfile.NamedTemporaryFile(delete=False, suffix=".crt")
            ca_file.write(cluster["ca_cert_data"].encode())
            ca_file.close()
            config.ssl_ca_cert = ca_file.name
        else:
            config.verify_ssl = False

        api_client = k8s_client.ApiClient(configuration=config)

    else:
        from kubernetes import config as k8s_config

        context = cluster.get("kubeconfig_context") or None
        k8s_config.load_kube_config(context=context)
        api_client = k8s_client.ApiClient()

    return api_client, namespace


def check_cluster_health(cluster: dict[str, Any]) -> tuple[str, str]:
    """Test connectivity to a K8s cluster. Returns (status, message)."""
    try:
        api_client, namespace = _get_k8s_client(cluster)
        from kubernetes import client as k8s_client

        v1 = k8s_client.CoreV1Api(api_client)
        version_api = k8s_client.VersionApi(api_client)
        info = version_api.get_code()
        v1.list_namespaced_pod(namespace=namespace, limit=1)
        return "online", f"Connected. Server version: {info.git_version}"
    except ImportError:
        return "unknown", "kubernetes Python package not installed"
    except Exception as exc:
        return "error", str(exc)


def _build_job_manifest(
    job: dict[str, Any],
    cluster: dict[str, Any],
    portal_url: str | None = None,
) -> dict[str, Any]:
    """Build a Kubernetes Job manifest for executing a harness run."""
    import os

    job_id = job["id"]
    dataset = job.get("dataset", "unknown")
    preset = job.get("preset", "")
    run_mode = job.get("run_mode") or cluster.get("default_run_mode", "batch")
    image = cluster.get("default_image") or DEFAULT_JOB_IMAGE

    overrides = job.get("dataset_overrides") or {}
    if isinstance(overrides, str):
        try:
            overrides = json.loads(overrides)
        except (json.JSONDecodeError, TypeError):
            overrides = {}

    container_command = ["python", "-m", "nemo_retriever.harness.portal_job"]

    env_vars = [
        {"name": "HARNESS_JOB_ID", "value": job_id},
        {"name": "HARNESS_JOB_JSON", "value": json.dumps(job)},
        {"name": "HARNESS_SKIP_LOCAL_HISTORY", "value": "1"},
    ]

    actual_portal_url = portal_url or os.environ.get("RETRIEVER_HARNESS_PORTAL_URL", "")
    if actual_portal_url:
        env_vars.append({"name": "HARNESS_PORTAL_URL", "value": actual_portal_url})

    resource_requests = {}
    resource_limits = {}
    gpu_count = cluster.get("gpu_count", 0)
    if gpu_count > 0:
        resource_limits["nvidia.com/gpu"] = str(gpu_count)
        resource_requests["nvidia.com/gpu"] = str(gpu_count)

    node_selector = cluster.get("node_selector")
    if isinstance(node_selector, str):
        try:
            node_selector = json.loads(node_selector)
        except (json.JSONDecodeError, TypeError):
            node_selector = None

    k8s_job_name = f"harness-{job_id[:8]}-{dataset[:20]}".lower().replace("_", "-").replace(" ", "-")
    k8s_job_name = k8s_job_name[:63]

    manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": k8s_job_name,
            "labels": {
                "app": "nemo-retriever-harness",
                "harness-job-id": job_id,
                "dataset": dataset[:63],
            },
        },
        "spec": {
            "ttlSecondsAfterFinished": JOB_TTL_SECONDS,
            "activeDeadlineSeconds": JOB_ACTIVE_DEADLINE_SECONDS,
            "backoffLimit": BACKOFF_LIMIT,
            "template": {
                "metadata": {
                    "labels": {
                        "app": "nemo-retriever-harness",
                        "harness-job-id": job_id,
                    },
                },
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "harness-runner",
                            "image": image,
                            "command": container_command,
                            "env": env_vars,
                            "resources": {
                                "requests": resource_requests,
                                "limits": resource_limits,
                            },
                        }
                    ],
                },
            },
        },
    }

    if node_selector:
        manifest["spec"]["template"]["spec"]["nodeSelector"] = node_selector

    return manifest


def dispatch_job_to_cluster(job: dict[str, Any], cluster: dict[str, Any]) -> str:
    """Create a Kubernetes Job on the target cluster for the given harness job.

    Returns the K8s Job name on success.
    """
    import os

    api_client, namespace = _get_k8s_client(cluster)

    from kubernetes import client as k8s_client

    batch_v1 = k8s_client.BatchV1Api(api_client)

    portal_url = os.environ.get("RETRIEVER_HARNESS_PORTAL_URL", "")
    manifest = _build_job_manifest(job, cluster, portal_url=portal_url)

    logger.info(
        "Dispatching job %s to cluster %s (ns=%s, image=%s)",
        job["id"],
        cluster["name"],
        namespace,
        cluster.get("default_image") or DEFAULT_JOB_IMAGE,
    )

    k8s_job = batch_v1.create_namespaced_job(
        namespace=namespace,
        body=manifest,
    )

    job_name = k8s_job.metadata.name
    logger.info("K8s Job created: %s in namespace %s", job_name, namespace)

    from nemo_retriever.harness import history

    history.update_job_status(job["id"], "running")

    return job_name


def get_job_status_on_cluster(
    job_id: str, cluster: dict[str, Any]
) -> dict[str, Any]:
    """Check the status of a K8s Job for the given harness job ID."""
    api_client, namespace = _get_k8s_client(cluster)

    from kubernetes import client as k8s_client

    batch_v1 = k8s_client.BatchV1Api(api_client)

    label_selector = f"harness-job-id={job_id}"
    jobs = batch_v1.list_namespaced_job(namespace=namespace, label_selector=label_selector)

    if not jobs.items:
        return {"found": False, "status": "unknown"}

    k8s_job = jobs.items[0]
    status = k8s_job.status

    if status.succeeded and status.succeeded > 0:
        return {"found": True, "status": "completed", "k8s_job_name": k8s_job.metadata.name}
    elif status.failed and status.failed > 0:
        return {"found": True, "status": "failed", "k8s_job_name": k8s_job.metadata.name}
    elif status.active and status.active > 0:
        return {"found": True, "status": "running", "k8s_job_name": k8s_job.metadata.name}
    else:
        return {"found": True, "status": "pending", "k8s_job_name": k8s_job.metadata.name}


def cancel_job_on_cluster(job_id: str, cluster: dict[str, Any]) -> bool:
    """Delete the K8s Job for a harness job, effectively cancelling it."""
    api_client, namespace = _get_k8s_client(cluster)

    from kubernetes import client as k8s_client

    batch_v1 = k8s_client.BatchV1Api(api_client)

    label_selector = f"harness-job-id={job_id}"
    jobs = batch_v1.list_namespaced_job(namespace=namespace, label_selector=label_selector)

    if not jobs.items:
        return False

    k8s_job_name = jobs.items[0].metadata.name
    batch_v1.delete_namespaced_job(
        name=k8s_job_name,
        namespace=namespace,
        propagation_policy="Background",
    )
    logger.info("Cancelled K8s Job %s for harness job %s", k8s_job_name, job_id)
    return True
