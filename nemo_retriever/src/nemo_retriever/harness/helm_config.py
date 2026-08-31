# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


_IMAGE_REPOSITORY_ENV = "HARNESS_HELM_SERVICE_IMAGE_REPOSITORY"
_IMAGE_TAG_ENV = "HARNESS_HELM_SERVICE_IMAGE_TAG"


@dataclass(frozen=True)
class HelmDeploymentConfig:
    """Non-secret configuration for one managed Helm harness deployment."""

    helm_chart: str
    service_image_repository: str
    service_image_tag: str
    helm_release: str = "nemo-retriever-harness"
    helm_namespace: str | None = None
    helm_values_file: str | None = None
    helm_set: dict[str, Any] = field(default_factory=dict)
    helm_timeout: int = 1800
    readiness_timeout: int = 1800
    helm_service_local_port: int = 17670
    helm_bin: str = "helm"
    kubectl_bin: str = "kubectl"
    helm_sudo: bool = False
    kubectl_sudo: bool = False
    helm_chart_version: str | None = None
    service_api_token_env: str = "HARNESS_SERVICE_API_TOKEN"

    def __post_init__(self) -> None:
        errors: list[str] = []
        for name in ("helm_chart", "service_image_repository", "service_image_tag", "helm_release"):
            if not str(getattr(self, name) or "").strip():
                errors.append(f"{name} must be a non-empty string")
        for name in ("helm_timeout", "readiness_timeout", "helm_service_local_port"):
            if int(getattr(self, name)) < 1:
                errors.append(f"{name} must be >= 1")
        if self.service_image_tag.strip().lower() in {"latest", "main", "nightly"}:
            errors.append("service_image_tag must identify an immutable image, not a moving channel")
        if not isinstance(self.helm_set, dict):
            errors.append("helm_set must be a mapping")
        if errors:
            raise ValueError("Invalid Helm harness config: " + "; ".join(errors))

    @property
    def service_api_token(self) -> str | None:
        return os.environ.get(self.service_api_token_env) or None

    def effective_helm_set(self) -> dict[str, Any]:
        values = dict(self.helm_set)
        values["service.image.repository"] = self.service_image_repository
        values["service.image.tag"] = self.service_image_tag
        return values

    def public_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["helm_namespace"] = self.helm_namespace or self.helm_release
        payload["helm_set"] = self.effective_helm_set()
        payload.pop("service_api_token_env", None)
        return payload


def _resolve_path(value: str | None, *, config_dir: Path) -> str | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    candidate = (config_dir / path).resolve()
    if candidate.exists() or value.startswith("."):
        return str(candidate)
    return value


def load_helm_config(path: Path) -> HelmDeploymentConfig:
    source = path.expanduser().resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read Helm harness config {source}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Could not parse Helm harness config {source}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(f"Helm harness config must contain a mapping: {source}")

    allowed = set(HelmDeploymentConfig.__dataclass_fields__)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown Helm harness config key: {unknown[0]}")
    data = dict(raw)
    repository = os.environ.get(_IMAGE_REPOSITORY_ENV) or data.get("service_image_repository")
    tag = os.environ.get(_IMAGE_TAG_ENV) or data.get("service_image_tag")
    if not repository or not tag:
        raise ValueError(
            "Helm deployment requires an explicit main/nightly service image via "
            f"{_IMAGE_REPOSITORY_ENV} and {_IMAGE_TAG_ENV} (or the matching config keys)"
        )
    data["service_image_repository"] = str(repository)
    data["service_image_tag"] = str(tag)
    data["helm_chart"] = _resolve_path(str(data.get("helm_chart") or ""), config_dir=source.parent)
    data["helm_values_file"] = _resolve_path(data.get("helm_values_file"), config_dir=source.parent)
    return HelmDeploymentConfig(**data)
