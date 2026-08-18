# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helm coverage for public and internal Secret-backed authentication."""

from __future__ import annotations

import subprocess

import pytest

from tests.test_helm_shared_results import _render


def _deployments(documents: list[dict]) -> dict[str, dict]:
    return {
        document["metadata"]["labels"]["app.kubernetes.io/component"]: document
        for document in documents
        if document.get("kind") == "Deployment"
    }


def _container_env(deployment: dict) -> dict[str, dict]:
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    return {item["name"]: item for item in container.get("env", [])}


def _auth_args(*, split: bool = False) -> tuple[str, ...]:
    args = (
        "--set",
        "serviceConfig.vectordb.enabled=true",
        "--set",
        "serviceConfig.nimEndpoints.embedInvokeUrl=http://embed.invalid/v1/embeddings",
        "--set",
        "serviceConfig.vectordb.internalAuth.enabled=true",
        "--set",
        "serviceConfig.vectordb.internalAuth.existingSecret.name=nrl-internal-auth",
        "--set",
        "serviceConfig.auth.enabled=true",
        "--set",
        "serviceConfig.auth.scopeTokenSecret.name=nrl-public-auth",
    )
    if split:
        return (
            *args,
            "--set",
            "topology.mode=split",
            "--set",
            "serviceMonitor.autoEnableInSplitMode=false",
        )
    return args


def test_auth_is_disabled_by_default_for_unprotected_standalone_deployments() -> None:
    documents = _render()
    deployment = next(iter(_deployments(documents).values()))
    assert "NRL_INTERNAL_VDB_TOKEN" not in _container_env(deployment)
    assert "NRL_SCOPE_TOKEN_FILE" not in _container_env(deployment)
    config = next(
        item["data"]["retriever-service.yaml"]
        for item in documents
        if item.get("kind") == "ConfigMap" and "retriever-service.yaml" in item.get("data", {})
    )
    assert "api_token: null" in config
    assert 'default_scope: "default"' in config
    assert "enabled: false" in config
    assert "allow_unscoped_dev: false" in config


def test_standalone_wires_separate_public_and_internal_secrets() -> None:
    documents = _render(*_auth_args())
    deployments = _deployments(documents)
    assert set(deployments) >= {"service", "vectordb"}

    for component in ("service", "vectordb"):
        internal = _container_env(deployments[component])["NRL_INTERNAL_VDB_TOKEN"]["valueFrom"]["secretKeyRef"]
        assert internal == {
            "name": "nrl-internal-auth",
            "key": "token",
            "optional": False,
        }

    service = deployments["service"]
    scope_env = _container_env(service)["NRL_SCOPE_TOKEN_FILE"]
    assert scope_env["value"] == "/var/run/secrets/nemo-retriever/auth/scope-tokens.json"
    pod_spec = service["spec"]["template"]["spec"]
    scope_volume = next(item for item in pod_spec["volumes"] if item["name"] == "scope-token")
    assert scope_volume["secret"] == {
        "secretName": "nrl-public-auth",
        "defaultMode": 288,
        "items": [{"key": "scope-tokens.json", "path": "scope-tokens.json"}],
    }
    config = next(
        item["data"]["retriever-service.yaml"]
        for item in documents
        if item.get("kind") == "ConfigMap" and "retriever-service.yaml" in item.get("data", {})
    )
    assert "api_token: null" in config
    assert "nrl-public-auth" not in config
    assert "nrl-internal-auth" not in config


def test_split_mounts_public_secret_only_on_gateway_and_internal_secret_everywhere() -> None:
    deployments = _deployments(_render(*_auth_args(split=True)))
    assert set(deployments) >= {"gateway", "realtime", "batch", "vectordb"}
    for component in ("gateway", "realtime", "batch", "vectordb"):
        assert (
            _container_env(deployments[component])["NRL_INTERNAL_VDB_TOKEN"]["valueFrom"]["secretKeyRef"]["name"]
            == "nrl-internal-auth"
        )

    assert "NRL_SCOPE_TOKEN_FILE" in _container_env(deployments["gateway"])
    for component in ("realtime", "batch", "vectordb"):
        assert "NRL_SCOPE_TOKEN_FILE" not in _container_env(deployments[component])
        pod_spec = deployments[component]["spec"]["template"]["spec"]
        assert all(item["name"] != "scope-token" for item in pod_spec.get("volumes", []))


def test_split_without_internal_auth_mounts_public_secret_on_workers() -> None:
    deployments = _deployments(
        _render(
            "--set",
            "topology.mode=split",
            "--set",
            "serviceConfig.auth.enabled=true",
            "--set",
            "serviceConfig.auth.scopeTokenSecret.name=nrl-public-auth",
        )
    )

    for component in ("gateway", "realtime", "batch"):
        deployment = deployments[component]
        assert "NRL_SCOPE_TOKEN_FILE" in _container_env(deployment)
        scope_token_volume = next(
            volume for volume in deployment["spec"]["template"]["spec"]["volumes"] if volume["name"] == "scope-token"
        )
        assert scope_token_volume["secret"]["secretName"] == "nrl-public-auth"


def test_internal_auth_requires_existing_secret_name() -> None:
    with pytest.raises(subprocess.CalledProcessError) as error:
        _render("--set", "serviceConfig.vectordb.internalAuth.enabled=true")
    assert "internalAuth.existingSecret.name" in error.value.stderr


def test_internal_auth_requires_existing_secret_key() -> None:
    with pytest.raises(subprocess.CalledProcessError) as error:
        _render(
            "--set",
            "serviceConfig.vectordb.internalAuth.enabled=true",
            "--set",
            "serviceConfig.vectordb.internalAuth.existingSecret.name=nrl-internal-auth",
            "--set-string",
            "serviceConfig.vectordb.internalAuth.existingSecret.key=",
        )
    assert "internalAuth.existingSecret.key" in error.value.stderr


def test_public_auth_requires_existing_secret_key() -> None:
    with pytest.raises(subprocess.CalledProcessError) as error:
        _render(
            "--set",
            "serviceConfig.auth.enabled=true",
            "--set",
            "serviceConfig.auth.scopeTokenSecret.name=nrl-public-auth",
            "--set-string",
            "serviceConfig.auth.scopeTokenSecret.key=",
        )
    assert "scopeTokenSecret.key" in error.value.stderr


def test_inline_public_token_requires_explicit_insecure_gate() -> None:
    with pytest.raises(subprocess.CalledProcessError) as error:
        _render(
            "--set",
            "serviceConfig.auth.enabled=true",
            "--set-string",
            "serviceConfig.auth.apiToken=sentinel-public-token",
        )
    assert "allowInsecureInlineApiToken=true" in error.value.stderr

    documents = _render(
        "--set",
        "serviceConfig.auth.enabled=true",
        "--set-string",
        "serviceConfig.auth.apiToken=sentinel-public-token",
        "--set",
        "serviceConfig.auth.allowInsecureInlineApiToken=true",
    )
    config = next(
        item["data"]["retriever-service.yaml"]
        for item in documents
        if item.get("kind") == "ConfigMap" and "retriever-service.yaml" in item.get("data", {})
    )
    assert "api_token: sentinel-public-token" in config


def test_inline_and_secret_public_auth_are_mutually_exclusive() -> None:
    with pytest.raises(subprocess.CalledProcessError) as error:
        _render(
            "--set",
            "serviceConfig.auth.enabled=true",
            "--set-string",
            "serviceConfig.auth.apiToken=sentinel-public-token",
            "--set",
            "serviceConfig.auth.allowInsecureInlineApiToken=true",
            "--set",
            "serviceConfig.auth.scopeTokenSecret.name=nrl-public-auth",
        )
    assert "mutually exclusive" in error.value.stderr


def test_enabled_public_auth_requires_a_credential_source() -> None:
    with pytest.raises(subprocess.CalledProcessError) as error:
        _render("--set", "serviceConfig.auth.enabled=true")
    assert "auth.enabled=true requires" in error.value.stderr


def test_public_credentials_require_auth_to_be_enabled() -> None:
    with pytest.raises(subprocess.CalledProcessError) as error:
        _render("--set", "serviceConfig.auth.scopeTokenSecret.name=nrl-public-auth")
    assert "require serviceConfig.auth.enabled=true" in error.value.stderr
