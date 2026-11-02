# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Permanent Helm runtime and API contracts for extraction NIM 2.0."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest import SkipTest

import yaml

CHART = Path(__file__).resolve().parents[1] / "helm"
EXPECTED_MODELS = {
    "nemotron-page-elements-v3": (
        "nvidia/nemotron-page-elements-v3",
        "/model-store/page-elements",
    ),
    "nemotron-table-structure-v1": (
        "nvidia/nemotron-table-structure-v1",
        "/model-store/table-structure",
    ),
    "nemotron-ocr-v2": ("nvidia/nemotron-ocr-v2", "/model-store/ocr"),
}


def _render(extra_args: tuple[str, ...] = ()) -> list[dict]:
    helm = shutil.which("helm")
    if helm is None:
        raise SkipTest("`helm` binary not available")
    command = [
        helm,
        "template",
        "nrl-nim-2-contracts",
        str(CHART),
        "--set",
        "ngcImagePullSecret.create=false",
        "--set",
        "ngcApiSecret.create=false",
        "--api-versions",
        "apps.nvidia.com/v1alpha1",
        *extra_args,
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return [document for document in yaml.safe_load_all(completed.stdout) if isinstance(document, dict)]


def _find(documents: list[dict], kind: str, name: str) -> dict:
    return next(
        document
        for document in documents
        if document.get("kind") == kind and document.get("metadata", {}).get("name") == name
    )


def test_extraction_nims_select_distinct_native_models() -> None:
    documents = _render()

    for name, (model_name, model_path) in EXPECTED_MODELS.items():
        service = _find(documents, "NIMService", name)
        env = {item["name"]: item.get("value") for item in service["spec"]["env"]}
        assert env["NIM_ENGINE_MODEL_DOWNLOAD_PROVIDER"] == "ngc"
        assert env["NIM_ENGINE_MODEL_NAME"] == model_name
        assert env["NIM_ENGINE_MODEL_PATH"] == model_path
        assert env["NIM_PERFORMANCE_MODE"] == "1"
        assert env["NIM_ENGINE_COUNT"] == "1"
        assert "NIM_PIPELINE_MAX_BATCH_SIZE" not in env
        assert "NIM_TRITON_MAX_BATCH_SIZE" not in env

    ocr_env = {
        item["name"]: item.get("value") for item in _find(documents, "NIMService", "nemotron-ocr-v2")["spec"]["env"]
    }
    assert ocr_env["NIM_ENGINE_MODEL_VARIANT"] == "multilingual"


def test_default_embed_nim_uses_native_nemotron_35_auto_precision() -> None:
    documents = _render()
    service = _find(documents, "NIMService", "nemotron-3-embed-1b")
    assert service["spec"]["image"] == {
        "repository": "nvcr.io/nim/nvidia/nemotron-embed",
        "tag": "latest",
        "pullPolicy": "Always",
        "pullSecrets": ["ngc-secret"],
    }
    env = {item["name"]: item.get("value") for item in service["spec"]["env"]}
    assert len(env) == len(service["spec"]["env"])
    assert env["NIM_SERVER_BIND_ADDR"] == "0.0.0.0:8000"
    assert env["NIM_ENGINE_MODEL_DOWNLOAD_PROVIDER"] == "ngc"
    assert env["NIM_SERVER_SERVED_MODEL_NAME"] == "nvidia/nemotron-3.5-embed-1b"
    assert env["NIM_ENGINE_COUNT"] == "1"
    assert env["NIM_ENGINE_STREAM_COUNT"] == "1"
    assert env["NIM_PERFORMANCE_MODE"] == "1"
    for setting in (
        "NIM_ENGINE_MODEL_NAME",
        "NIM_ENGINE_PRECISION",
        "NIM_ENGINE_MODEL_PATH",
        "NIM_PIPELINE_MAX_SEQ_LEN",
        "NIM_PIPELINE_MAX_BATCH_SIZE",
        "NIM_HTTP_API_PORT",
    ):
        assert setting not in env
    assert not any(name.startswith(("NIM_TRITON_", "TRITON_")) for name in env)
    assert service["spec"]["expose"]["service"] == {"type": "ClusterIP", "port": 8000}
    assert service["spec"]["storage"]["nimCache"]["name"] == "nemotron-3-embed-1b"
    cache = _find(documents, "NIMCache", "nemotron-3-embed-1b")
    assert cache["spec"]["source"]["ngc"]["modelPuller"] == "nvcr.io/nim/nvidia/nemotron-embed:latest"
    assert cache["spec"]["env"] == [{"name": "NIM_ENGINE_MODEL_DOWNLOAD_PROVIDER", "value": "ngc"}]

    for mode in ("standalone", "split"):
        rendered = _render(("--set", f"topology.mode={mode}"))
        for document in rendered:
            config_text = document.get("data", {}).get("retriever-service.yaml")
            if config_text:
                config = yaml.safe_load(config_text)
                assert config["nim_endpoints"]["embed_model_name"] == env["NIM_SERVER_SERVED_MODEL_NAME"]
                assert config["vectordb"]["embed_model"] == env["NIM_SERVER_SERVED_MODEL_NAME"]
        vectordb = next(
            document
            for document in rendered
            if document.get("kind") == "Deployment" and document["metadata"]["name"].endswith("-vectordb")
        )
        args = vectordb["spec"]["template"]["spec"]["containers"][0]["args"]
        assert args[args.index("--embed-model") + 1] == env["NIM_SERVER_SERVED_MODEL_NAME"]


def test_explicit_embedding_alias_download_provider_and_grpc_overrides(tmp_path: Path) -> None:
    values = yaml.safe_load((CHART / "values.yaml").read_text())
    embedding = values["nimOperator"]["vlm_embed"]
    embedding["image"].update(tag="custom", pullPolicy="IfNotPresent")
    for item in embedding["env"]:
        if item["name"] == "NIM_SERVER_SERVED_MODEL_NAME":
            item["value"] = "company/embedding"
        if item["name"] == "NIM_ENGINE_MODEL_DOWNLOAD_PROVIDER":
            item["value"] = "hf"
    embedding["env"].append({"name": "NIM_SERVER_GRPC_BIND_ADDR", "value": "0.0.0.0:8001"})
    embedding["expose"]["service"]["grpcPort"] = 8001
    values["serviceConfig"]["vectordb"]["embedModel"] = "company/embedding"
    path = tmp_path / "values.yaml"
    path.write_text(yaml.safe_dump(values))
    documents = _render(("--values", str(path)))
    service = _find(documents, "NIMService", "nemotron-3-embed-1b")
    env = {item["name"]: item.get("value") for item in service["spec"]["env"]}
    assert env["NIM_SERVER_SERVED_MODEL_NAME"] == "company/embedding"
    assert env["NIM_SERVER_GRPC_BIND_ADDR"] == "0.0.0.0:8001"
    assert service["spec"]["expose"]["service"]["grpcPort"] == 8001
    assert service["spec"]["image"]["tag"] == "custom"
    assert service["spec"]["image"]["pullPolicy"] == "IfNotPresent"
    cache = _find(documents, "NIMCache", "nemotron-3-embed-1b")
    assert cache["spec"]["env"] == [{"name": "NIM_ENGINE_MODEL_DOWNLOAD_PROVIDER", "value": "hf"}]


def test_operator_managed_urls_use_nim_2_contracts() -> None:
    documents = _render()
    rendered_config = "\n".join(
        document.get("data", {}).get("retriever-service.yaml", "")
        for document in documents
        if document.get("kind") == "ConfigMap"
    )

    assert 'page_elements_invoke_url: "http://nemotron-page-elements-v3:8000/v1/page-elements"' in rendered_config
    assert 'table_structure_invoke_url: "http://nemotron-table-structure-v1:8000/v1/table-structure"' in rendered_config
    assert 'ocr_invoke_url: "http://nemotron-ocr-v2:8000/v1/ocr"' in rendered_config
