# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_release_helm_chart_module() -> ModuleType:
    script_path = REPO_ROOT / "ci" / "scripts" / "release_helm_chart.py"
    spec = importlib.util.spec_from_file_location("release_helm_chart", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_versions_update_chart_and_default_service_image(tmp_path: Path) -> None:
    chart_dir = tmp_path / "chart"
    chart_dir.mkdir()
    chart_path = chart_dir / "Chart.yaml"
    chart_path.write_text(
        """\
# Keep chart comments.
apiVersion: v2
name: source-name
version: "1.0.0"
appVersion: "1.0.0"
""",
        encoding="utf-8",
    )
    values_path = chart_dir / "values.yaml"
    values_path.write_text(
        """\
# Keep values comments.
service:
  image:
    repository: example.invalid/service
    tag: "1.0.0"
  gpuImage:
    tag: "unchanged"
""",
        encoding="utf-8",
    )
    release_helm_chart = _load_release_helm_chart_module()

    release_helm_chart._set_release_versions(chart_dir, "nemo-retriever", "26.05-RC8")

    chart = yaml.safe_load(chart_path.read_text(encoding="utf-8"))
    values = yaml.safe_load(values_path.read_text(encoding="utf-8"))
    assert chart["name"] == "nemo-retriever"
    assert chart["version"] == "26.05-RC8"
    assert chart["appVersion"] == "26.05-RC8"
    assert values["service"]["image"]["tag"] == "26.05-RC8"
    assert values["service"]["gpuImage"]["tag"] == "unchanged"
    assert "# Keep chart comments." in chart_path.read_text(encoding="utf-8")
    assert "# Keep values comments." in values_path.read_text(encoding="utf-8")


def test_image_inventory_covers_nested_nim_images_and_skips_unset_repositories() -> None:
    release_helm_chart = _load_release_helm_chart_module()
    values = {
        "service": {
            "image": {"repository": "example.invalid/service", "tag": "26.08-RC1"},
            "gpuImage": {"repository": "", "tag": ""},
        },
        "nimOperator": {"ocr": {"image": {"repository": "example.invalid/ocr", "tag": "2.0.1"}}},
    }

    references = release_helm_chart._collect_image_references(values)

    assert sorted(references) == [
        ("nimOperator.ocr.image", "example.invalid/ocr", "2.0.1"),
        ("service.image", "example.invalid/service", "26.08-RC1"),
    ]


@pytest.mark.parametrize("tag", ["", "latest", "main"])
def test_release_rejects_images_without_a_pinned_tag(tag: str) -> None:
    release_helm_chart = _load_release_helm_chart_module()

    with pytest.raises(ValueError, match="must pin an explicit, non-floating tag"):
        release_helm_chart._validate_pinned_image_tags([("service.image", "example.invalid/service", tag)])


def test_checked_in_chart_pins_every_deployed_image() -> None:
    values = yaml.safe_load((REPO_ROOT / "nemo_retriever" / "helm" / "values.yaml").read_text(encoding="utf-8"))
    release_helm_chart = _load_release_helm_chart_module()

    references = release_helm_chart._collect_image_references(values)

    assert references
    release_helm_chart._validate_pinned_image_tags(references)


def test_image_inventory_lists_release_version_and_every_image() -> None:
    release_helm_chart = _load_release_helm_chart_module()

    inventory = release_helm_chart._render_image_inventory(
        [("service.image", "example.invalid/service", "26.08-RC1")],
        "nemo-retriever",
        "26.08-RC1",
    )

    assert "`nemo-retriever:26.08-RC1`" in inventory
    assert "| `service.image` | `example.invalid/service:26.08-RC1` |" in inventory
