# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
import tomllib
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_local_extra_depends_on_versioned_ocr_v2_nightly() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    local_deps = pyproject["project"]["optional-dependencies"]["local"]
    uv_tool = pyproject["tool"]["uv"]
    uv_sources = uv_tool["sources"]

    assert "nemotron-ocr>=1.0.2.dev0,<1.0.2a0; sys_platform == 'linux'" in local_deps
    assert "nemotron-ocr-v2>=2.0.0.dev0,<2.0.0a0; sys_platform == 'linux'" in local_deps
    assert "nemotron-ocr" in uv_tool["no-build-package"]
    assert "nemotron-ocr-v2" in uv_tool["no-build-package"]
    assert uv_sources["nemotron-ocr"] == {"index": "test-pypi"}
    assert uv_sources["nemotron-ocr-v2"] == {"index": "test-pypi"}


def test_local_ocr_v2_wrapper_imports_versioned_module() -> None:
    source = (PROJECT_ROOT / "src" / "nemo_retriever" / "model" / "local" / "nemotron_ocr_v2.py").read_text(
        encoding="utf-8"
    )

    assert "from nemotron_ocr_v2.inference import pipeline_v2" in source
    assert "Local Nemotron OCR v2 requires the `nemotron_ocr_v2` package." in source


def test_local_ocr_v2_wrapper_initializes_from_versioned_namespace(monkeypatch) -> None:
    class FakeTensor:
        pass

    class FakeModule:
        pass

    torch_stub = types.ModuleType("torch")
    torch_stub.__path__ = []
    torch_nn_stub = types.ModuleType("torch.nn")
    torch_nn_stub.Module = FakeModule
    torch_stub.Tensor = FakeTensor
    torch_stub.nn = torch_nn_stub
    torch_stub.float16 = object()
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "torch.nn", torch_nn_stub)

    constructed_with: list[str | None] = []

    class FakeUpstreamNemotronOCRV2:
        def __init__(self, model_dir: str | None = None) -> None:
            constructed_with.append(model_dir)

    pipeline_v2 = types.ModuleType("nemotron_ocr_v2.inference.pipeline_v2")
    pipeline_v2.NemotronOCRV2 = FakeUpstreamNemotronOCRV2
    pipeline_v2.hf_hub_download = lambda *args, **kwargs: ""

    inference_pkg = types.ModuleType("nemotron_ocr_v2.inference")
    inference_pkg.pipeline_v2 = pipeline_v2

    ocr_v2_pkg = types.ModuleType("nemotron_ocr_v2")
    ocr_v2_pkg.inference = inference_pkg

    monkeypatch.setitem(sys.modules, "nemotron_ocr_v2", ocr_v2_pkg)
    monkeypatch.setitem(sys.modules, "nemotron_ocr_v2.inference", inference_pkg)
    monkeypatch.setitem(sys.modules, "nemotron_ocr_v2.inference.pipeline_v2", pipeline_v2)
    monkeypatch.delitem(sys.modules, "nemo_retriever.model.local.nemotron_ocr_v2", raising=False)

    wrapper_module = importlib.import_module("nemo_retriever.model.local.nemotron_ocr_v2")
    wrapper = wrapper_module.NemotronOCRV2(model_dir="/models/ocr-v2")

    assert isinstance(wrapper._model, FakeUpstreamNemotronOCRV2)
    assert constructed_with == ["/models/ocr-v2"]
