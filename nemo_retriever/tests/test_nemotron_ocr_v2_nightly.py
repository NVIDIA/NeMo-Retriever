# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent


def _install_ocr_import_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    torch_mod = ModuleType("torch")
    nn_mod = ModuleType("torch.nn")

    class _Module:
        pass

    nn_mod.Module = _Module
    torch_mod.nn = nn_mod
    torch_mod.Tensor = object
    torch_mod.float16 = "float16"

    pil_mod = ModuleType("PIL")
    image_mod = ModuleType("PIL.Image")
    pil_mod.Image = image_mod

    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.nn", nn_mod)
    monkeypatch.setitem(sys.modules, "PIL", pil_mod)
    monkeypatch.setitem(sys.modules, "PIL.Image", image_mod)


def _install_upstream_ocr_v2_stub(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    captured_kwargs: list[dict[str, object]] = []

    class _NemotronOCRV2:
        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.append(kwargs)

    nemotron_ocr_mod = ModuleType("nemotron_ocr")
    inference_mod = ModuleType("nemotron_ocr.inference")
    pipeline_v2_mod = ModuleType("nemotron_ocr.inference.pipeline_v2")
    pipeline_v2_mod.NemotronOCRV2 = _NemotronOCRV2
    inference_mod.pipeline_v2 = pipeline_v2_mod
    nemotron_ocr_mod.inference = inference_mod

    monkeypatch.setitem(sys.modules, "nemotron_ocr", nemotron_ocr_mod)
    monkeypatch.setitem(sys.modules, "nemotron_ocr.inference", inference_mod)
    monkeypatch.setitem(sys.modules, "nemotron_ocr.inference.pipeline_v2", pipeline_v2_mod)
    monkeypatch.delenv("RETRIEVER_ENABLE_TORCH_TRT", raising=False)

    return captured_kwargs


def test_local_extra_depends_on_ocr_2_nightly_only() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    local_deps = pyproject["project"]["optional-dependencies"]["local"]
    uv_tool = pyproject["tool"]["uv"]
    uv_sources = uv_tool["sources"]

    assert (
        "nemotron-ocr>=2.0.0.dev0,<2.0.0a0; sys_platform == 'linux' "
        "and (platform_machine == 'x86_64' or platform_machine == 'aarch64')"
    ) in local_deps
    assert not any(dep.startswith("nemotron-ocr-v2") for dep in local_deps)
    assert "nemotron-ocr" in uv_tool["no-build-package"]
    assert "nemotron-ocr-v2" not in uv_tool["no-build-package"]
    assert uv_sources["nemotron-ocr"] == {"index": "test-pypi"}
    assert "nemotron-ocr-v2" not in uv_sources


def test_local_ocr_v2_wrapper_uses_original_namespace_and_lang_selector() -> None:
    source = (PROJECT_ROOT / "src" / "nemo_retriever" / "model" / "local" / "nemotron_ocr_v2.py").read_text(
        encoding="utf-8"
    )

    assert "from nemotron_ocr.inference import pipeline_v2" in source
    assert 'lang: str = "v2_english"' in source
    assert "_NemotronOCRV2(model_dir=model_dir, lang=lang)" in source
    assert "_NemotronOCRV2(lang=lang)" in source
    assert "nemotron_ocr_v2" not in source
    assert "nemotron-ocr-v2` from TestPyPI" not in source


def test_local_ocr_v2_wrapper_rejects_invalid_lang_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_ocr_import_stubs(monkeypatch)

    from nemo_retriever.model.local.nemotron_ocr_v2 import NemotronOCRV2

    with pytest.raises(ValueError, match=r"Invalid lang selector 'v3'"):
        NemotronOCRV2(lang="v3")


@pytest.mark.parametrize("selector", ["v1", "v2_english", "v2_multi"])
def test_local_ocr_v2_wrapper_accepts_valid_lang_selectors(
    monkeypatch: pytest.MonkeyPatch,
    selector: str,
) -> None:
    _install_ocr_import_stubs(monkeypatch)
    captured_kwargs = _install_upstream_ocr_v2_stub(monkeypatch)

    from nemo_retriever.model.local.nemotron_ocr_v2 import NemotronOCRV2

    NemotronOCRV2(lang=selector)

    assert captured_kwargs == [{"lang": selector}]


def test_huggingface_ocr_nightly_does_not_carry_namespace_patch_knobs() -> None:
    workflow_path = REPO_ROOT / ".github" / "workflows" / "huggingface-nightly.yml"
    if not workflow_path.exists():
        pytest.skip("Hugging Face nightly workflow is not available in this checkout")

    workflow = workflow_path.read_text(encoding="utf-8")
    v2_stanza = workflow.split("- id: nemotron-ocr-v2", 1)[1].split("container:", 1)[0]

    assert 'nightly_base_version: "2.0.0"' in v2_stanza
    assert "project_name:" not in workflow
    assert "package_rename:" not in workflow
    assert "expected_project_name:" not in workflow
    assert "expected_package:" not in workflow
    assert "--project-name" not in workflow
    assert "--rename-python-package" not in workflow
