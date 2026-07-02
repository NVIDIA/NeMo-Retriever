# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_retriever.service.config import (
    LocalEmbedConfig,
    LocalModelsConfig,
    NimEndpointsConfig,
    PipelinePoolConfig,
    ServiceConfig,
)
from nemo_retriever.service.services.pipeline_executor import (
    _embed_params_enabled,
    _resolve_embed_params,
    build_asr_params,
    build_embed_params,
    build_extract_params,
)


def test_build_embed_params_returns_none_without_endpoint_or_local() -> None:
    nim = NimEndpointsConfig()
    local = LocalModelsConfig()
    assert build_embed_params(nim, local) is None


def test_build_embed_params_from_nim_config() -> None:
    nim = NimEndpointsConfig(
        embed_invoke_url="http://embed-nim/v1/embeddings",
        embed_model_name="nvidia/llama-nemotron-embed-vl-1b-v2",
        embed_model_name_prefix="nvidia",
        api_key="k",
    )
    ep = build_embed_params(nim, LocalModelsConfig(enabled=True))
    assert ep is not None
    assert ep.embed_invoke_url == "http://embed-nim/v1/embeddings"
    assert ep.model_name == "nvidia/llama-nemotron-embed-vl-1b-v2"
    assert ep.embed_model_name == "nvidia/llama-nemotron-embed-vl-1b-v2"
    assert ep.embed_model_name_prefix == "nvidia"
    assert ep.api_key == "k"


def test_build_embed_params_local_when_enabled() -> None:
    local = LocalModelsConfig(
        enabled=True,
        hf_cache_dir="/models/hf",
        device="cuda:0",
        embed=LocalEmbedConfig(
            model_name="nvidia/llama-nemotron-embed-1b-v2",
            local_ingest_embed_backend="hf",
            gpu_memory_utilization=0.6,
        ),
    )
    ep = build_embed_params(NimEndpointsConfig(), local)
    assert ep is not None
    assert ep.embed_invoke_url is None
    assert ep.model_name == "nvidia/llama-nemotron-embed-1b-v2"
    assert ep.local_ingest_embed_backend == "hf"
    assert ep.runtime.device == "cuda:0"
    assert ep.runtime.hf_cache_dir == "/models/hf"
    assert ep.runtime.gpu_memory_utilization == 0.6


def test_build_embed_params_nim_url_wins_over_local() -> None:
    nim = NimEndpointsConfig(embed_invoke_url="http://embed-nim/v1/embeddings")
    local = LocalModelsConfig(enabled=True)
    ep = build_embed_params(nim, local)
    assert ep is not None
    assert ep.embed_invoke_url == "http://embed-nim/v1/embeddings"


def test_build_extract_params_local_enables_structure_stages() -> None:
    local = LocalModelsConfig(enabled=True)
    ep = build_extract_params(NimEndpointsConfig(), local)
    assert ep.use_table_structure is True
    assert ep.use_graphic_elements is True
    assert ep.ocr_version == "v2"
    assert ep.page_elements_invoke_url is None


def test_build_extract_params_nim_url_wins_over_local_flags() -> None:
    nim = NimEndpointsConfig(table_structure_invoke_url="http://table-nim/v1/infer")
    local = LocalModelsConfig(enabled=True)
    ep = build_extract_params(nim, local)
    assert ep.table_structure_invoke_url == "http://table-nim/v1/infer"
    assert ep.use_table_structure is True


def test_build_asr_params_local_when_enabled() -> None:
    local = LocalModelsConfig(enabled=True)
    asr = build_asr_params(NimEndpointsConfig(), local)
    assert asr is not None
    assert asr.audio_endpoints == (None, None)


def test_build_asr_params_remote_grpc_wins() -> None:
    nim = NimEndpointsConfig(audio_grpc_endpoint="parakeet-nim:50051", api_key="k")
    asr = build_asr_params(nim, LocalModelsConfig(enabled=True))
    assert asr is not None
    assert asr.audio_endpoints == ("parakeet-nim:50051", None)
    assert asr.auth_token == "k"


def test_embed_params_enabled_detects_local_model_name() -> None:
    assert _embed_params_enabled({"model_name": "nvidia/llama-nemotron-embed-1b-v2"}) is True
    assert _embed_params_enabled({}) is False
    assert _embed_params_enabled({"embed_invoke_url": "http://embed/v1/embeddings"}) is True


def test_resolve_embed_params_accepts_local_without_url() -> None:
    base = {
        "model_name": "nvidia/llama-nemotron-embed-vl-1b-v2",
        "embed_model_name": "nvidia/llama-nemotron-embed-vl-1b-v2",
        "local_ingest_embed_backend": "hf",
    }
    ep = _resolve_embed_params(base, None)
    assert ep is not None
    assert ep.embed_invoke_url is None
    assert ep.model_name == "nvidia/llama-nemotron-embed-vl-1b-v2"


def test_service_config_loads_local_models_from_dict() -> None:
    cfg = ServiceConfig(
        local_models=LocalModelsConfig(
            enabled=True,
            embed=LocalEmbedConfig(local_ingest_embed_backend="vllm"),
        )
    )
    assert cfg.local_models.enabled is True
    assert cfg.local_models.embed.local_ingest_embed_backend == "vllm"


def test_local_models_caps_pipeline_workers() -> None:
    cfg = ServiceConfig(
        local_models=LocalModelsConfig(enabled=True, max_process_pool_workers=1),
        pipeline=PipelinePoolConfig(realtime_workers=8, batch_workers=16),
    )
    assert cfg.pipeline.realtime_workers == 1
    assert cfg.pipeline.batch_workers == 1


def test_local_models_respects_higher_process_pool_cap() -> None:
    cfg = ServiceConfig(
        local_models=LocalModelsConfig(enabled=True, max_process_pool_workers=2),
        pipeline=PipelinePoolConfig(realtime_workers=8, batch_workers=16),
    )
    assert cfg.pipeline.realtime_workers == 2
    assert cfg.pipeline.batch_workers == 2
