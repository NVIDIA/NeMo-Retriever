# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose contracts for the Object Detection and OCR 2.0 NIMs."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "dev/compose/service-mode.compose.yaml"
CORE_PRESET = ROOT / "dev/compose/presets/nims-core.env"
LOCAL_PRESET = ROOT / "dev/compose/presets/local-models.env"


def _compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


def test_extraction_nims_use_distinct_native_models_and_supported_model_paths() -> None:
    services = _compose()["services"]
    expected = {
        "nim-page-elements": (
            "nvidia/nemotron-page-elements-v3",
            "page-elements",
            "NIM_PAGE_ELEMENTS_CACHE_PATH",
        ),
        "nim-table-structure": (
            "nvidia/nemotron-table-structure-v1",
            "table-structure",
            "NIM_TABLE_STRUCTURE_CACHE_PATH",
        ),
        "nim-ocr": ("nvidia/nemotron-ocr-v2", "ocr", "NIM_OCR_CACHE_PATH"),
    }

    for service_name, (model_name, model_dir, path_variable) in expected.items():
        service = services[service_name]
        env = service["environment"]
        assert env["NIM_ENGINE_MODEL_DOWNLOAD_PROVIDER"] == "ngc"
        assert env["NIM_ENGINE_MODEL_NAME"] == model_name
        assert env["NIM_ENGINE_MODEL_PATH"] == f"${{{path_variable}:-/model}}/{model_dir}"
        assert service["volumes"] == [
            f"nim_{service_name.removeprefix('nim-').replace('-', '_')}_cache:${{{path_variable}:-/model}}"
        ]
        assert service["healthcheck"]["test"] == [
            "CMD",
            "curl",
            "--fail",
            "--silent",
            "http://localhost:8000/v1/health/ready",
        ]
        assert not any(name.startswith("NIM_TRITON_") for name in env)

    assert services["nim-ocr"]["environment"]["NIM_ENGINE_MODEL_VARIANT"] == "multilingual"
    assert "/model-store" not in COMPOSE.read_text(encoding="utf-8")


def test_compose_and_presets_select_expected_inference_modes() -> None:
    compose = _compose()
    config = compose["configs"]["retriever_service_config"]["content"]
    core_preset = set(CORE_PRESET.read_text(encoding="utf-8").splitlines())
    local_preset = set(LOCAL_PRESET.read_text(encoding="utf-8").splitlines())

    hosted_endpoints = (
        "${NIM_PAGE_ELEMENTS_URL-https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-page-elements-v3}",
        "${NIM_TABLE_STRUCTURE_URL-https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-table-structure-v1}",
        "${NIM_OCR_URL-https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v2}",
    )
    for endpoint in hosted_endpoints:
        assert endpoint in config

    internal_endpoints = {
        "NIM_PAGE_ELEMENTS_URL=http://nim-page-elements:8000/v1/page-elements",
        "NIM_TABLE_STRUCTURE_URL=http://nim-table-structure:8000/v1/table-structure",
        "NIM_OCR_URL=http://nim-ocr:8000/v1/ocr",
        "NIM_EMBED_URL=http://nim-embedding:8000/v1/embeddings",
    }
    assert internal_endpoints <= core_preset
    assert "/v1/infer" not in CORE_PRESET.read_text(encoding="utf-8")

    disabled_endpoints = {
        "NIM_PAGE_ELEMENTS_URL=",
        "NIM_TABLE_STRUCTURE_URL=",
        "NIM_OCR_URL=",
        "NIM_EMBED_URL=",
    }
    assert disabled_endpoints <= local_preset
