# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

from nemo_retriever.models.embed_model_spec import EmbedModelSpec
from nemo_retriever.models.local_embedder_spec import LocalEmbedderSpec
from nemo_retriever.models.warmup_registry import (
    build_warmup_spec,
    clear_warmed_models,
    get_warmed_model,
    is_warmup_active,
    warm_local_models,
)
from nemo_retriever.service.config import LocalModelsConfig
from nemo_retriever.service.services.pipeline_executor import (
    _build_pool_warmup_spec_json,
    _resolve_max_tasks_per_child,
    get_service_warmup_status,
    warmup_process_pool_workers,
)


def _checkpoint(model_id: str, revision: str | None = None, **_kwargs) -> EmbedModelSpec:
    return EmbedModelSpec(
        model_id=model_id,
        revision=revision,
        family="text",
        output_dimension=2048,
        query_prefix="query: ",
        document_prefix="passage: ",
        max_input_tokens=8192,
        query_prefix_declared=True,
        document_prefix_declared=True,
    )


def test_build_warmup_spec_full_local_stack() -> None:
    extract = {
        "use_table_structure": True,
        "ocr_version": "v2",
    }
    embed = {
        "model_name": "nvidia/llama-nemotron-embed-vl-1b-v2",
        "local_ingest_embed_backend": "hf",
        "runtime": {"device": "cuda:0"},
    }
    spec = build_warmup_spec(extract, embed, {})
    assert spec is not None
    assert "page_elements" in spec["stages"]
    assert "ocr" in spec["stages"]
    assert "table_structure" in spec["stages"]
    assert spec["embed"]["backend"] == "hf"
    assert spec["asr"] is True


def test_build_warmup_spec_skips_remote_stages() -> None:
    extract = {
        "page_elements_invoke_url": "http://pe-nim/v1/infer",
        "ocr_invoke_url": "http://ocr-nim/v1/infer",
        "use_table_structure": True,
        "table_structure_invoke_url": "http://table-nim/v1/infer",
    }
    embed = {"embed_invoke_url": "http://embed-nim/v1/embeddings"}
    asr = {"audio_endpoints": ["grpc://parakeet:50051", None]}
    assert build_warmup_spec(extract, embed, asr) is None


def test_build_warmup_spec_preserves_embed_model_revision() -> None:
    revision = "a" * 40

    with patch("nemo_retriever.models.embed_model_spec.resolve_embed_model_spec", side_effect=_checkpoint):
        spec = build_warmup_spec(
            {},
            {
                "model_name": "acme/embed-model",
                "embed_model_revision": revision,
                "local_ingest_embed_backend": "hf",
            },
            None,
        )

    assert spec is not None
    assert spec["embed"]["revision"] == revision


def test_warmed_identity_uses_resolved_sha_for_a_symbolic_revision() -> None:
    resolved_sha = "c" * 40

    def resolve_symbolic(model_id: str, revision: str | None = None, **_kwargs) -> EmbedModelSpec:
        return _checkpoint(model_id, resolved_sha)

    with patch("nemo_retriever.models.embed_model_spec.resolve_embed_model_spec", side_effect=resolve_symbolic):
        identity = LocalEmbedderSpec.from_config(
            {
                "model_name": "acme/embed-model",
                "revision": "main",
                "backend": "hf",
            }
        )

    assert identity.revision == resolved_sha


def test_warm_local_models_registers_mock_instances() -> None:
    clear_warmed_models()
    mock_page = MagicMock(name="page_elements")
    mock_ocr = MagicMock(name="ocr")
    mock_embed = MagicMock(name="embed")

    with (
        patch("nemo_retriever.models.local.NemotronPageElementsV3", return_value=mock_page),
        patch("nemo_retriever.models.local.NemotronOCRV2", return_value=mock_ocr),
        patch("nemo_retriever.models.local.NemotronTableStructureV1"),
        patch("nemo_retriever.models.embed_model_spec.resolve_embed_model_spec", side_effect=_checkpoint),
        patch("nemo_retriever.models.create_local_embedder", return_value=mock_embed),
    ):
        warm_local_models(
            {
                "stages": ["page_elements", "ocr", "table_structure"],
                "ocr_version": "v2",
                "embed": {"model_name": "m", "revision": "a" * 40, "backend": "hf"},
            }
        )

    assert is_warmup_active()
    assert get_warmed_model("page_elements") is mock_page
    assert get_warmed_model("ocr") is mock_ocr
    assert get_warmed_model("embed") is mock_embed
    clear_warmed_models()


def test_warm_local_models_forwards_embed_model_revision() -> None:
    clear_warmed_models()
    revision = "b" * 40

    with (
        patch("nemo_retriever.models.embed_model_spec.resolve_embed_model_spec", side_effect=_checkpoint),
        patch("nemo_retriever.models.create_local_embedder") as create_local_embedder,
    ):
        warm_local_models(
            {
                "embed": {
                    "model_name": "acme/embed-model",
                    "revision": revision,
                    "backend": "hf",
                }
            }
        )

    assert create_local_embedder.call_args.kwargs["revision"] == revision
    clear_warmed_models()


def test_gpu_actor_loads_requested_embedder_when_warmed_identity_differs() -> None:
    from nemo_retriever.common.params import EmbedParams
    from nemo_retriever.operators.embed import gpu_operator

    clear_warmed_models()
    warmed = MagicMock(name="warmed_revision_a")
    with (
        patch("nemo_retriever.models.embed_model_spec.resolve_embed_model_spec", side_effect=_checkpoint),
        patch("nemo_retriever.models.create_local_embedder", return_value=warmed),
    ):
        warm_local_models(
            {
                "embed": {
                    "model_name": "acme/embed-model",
                    "revision": "a" * 40,
                    "backend": "hf",
                }
            }
        )

    requested = MagicMock(name="requested_revision_b")
    with (
        patch("nemo_retriever.models.embed_model_spec.resolve_embed_model_spec", side_effect=_checkpoint),
        patch("nemo_retriever.models.create_local_embedder", return_value=requested) as create_local_embedder,
    ):
        actor = gpu_operator._BatchEmbedActor(
            params=EmbedParams(
                model_name="acme/embed-model",
                embed_model_revision="b" * 40,
                local_ingest_embed_backend="hf",
            )
        )

    assert actor._model is requested
    assert actor._model is not warmed
    create_local_embedder.assert_called_once()
    assert create_local_embedder.call_args.kwargs["revision"] == "b" * 40

    clear_warmed_models()


def test_resolve_max_tasks_per_child_warmup_defaults() -> None:
    local = LocalModelsConfig(enabled=True, warmup_on_startup=True)
    assert _resolve_max_tasks_per_child(local) == 10_000

    local_custom = LocalModelsConfig(enabled=True, warmup_on_startup=True, max_tasks_per_child=500)
    assert _resolve_max_tasks_per_child(local_custom) == 500

    local_off = LocalModelsConfig(enabled=True, warmup_on_startup=False)
    assert _resolve_max_tasks_per_child(local_off) == 100


def test_build_pool_warmup_spec_json_disabled() -> None:
    local = LocalModelsConfig(enabled=True, warmup_on_startup=False)
    assert _build_pool_warmup_spec_json(local, {"use_table_structure": True}, None, None) == ""


def test_warmup_process_pool_workers_no_targets() -> None:
    status = warmup_process_pool_workers()
    assert status["complete"] is True


def test_get_service_warmup_status_returns_copy() -> None:
    from nemo_retriever.service.services import pipeline_executor

    pipeline_executor._service_warmup_state["complete"] = False
    status = get_service_warmup_status()
    status["complete"] = True
    assert get_service_warmup_status()["complete"] is False
