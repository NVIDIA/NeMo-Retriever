# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for agentic (ReAct) retrieval wiring in the revamped harness.

Covers the three integration points that re-port the old ``harness/run.py``
agentic logic onto the in-process harness:

1. ``build_query_request`` populates ``QueryRequest.agentic`` from the benchmark
   spec ``query`` section (and the agentic override paths are accepted).
2. ``build_agentic_config`` maps a ``QueryRequest`` onto an
   ``AgenticRetrievalConfig`` (including the harness ``top_k`` override).
3. ``run_beir_queries`` routes to the agentic batch path when enabled and writes
   the expected BEIR artifacts.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from nemo_retriever.harness.artifact_writer import ArtifactWriter
from nemo_retriever.harness.beir_runner import run_beir_queries
from nemo_retriever.harness.resolution import QUERY_OVERRIDE_PATHS, build_query_request
from nemo_retriever.query.options import QueryAgenticOptions, QueryRequest
from nemo_retriever.query.workflow import build_agentic_config
from nemo_retriever.tools.recall.beir import BeirDataset


def _resolved(query: dict) -> dict:
    return {
        "query": query,
        "ingest": {"storage": {"lancedb_uri": "lancedb", "table_name": "bench"}},
    }


def test_query_override_paths_include_agentic_fields() -> None:
    for key in (
        "query.agentic",
        "query.agentic_llm_model",
        "query.agentic_llm_backend",
        "query.agentic_invoke_url",
        "query.agentic_local_llm_backend",
        "query.agentic_local_hf_cache_dir",
        "query.agentic_local_device",
        "query.agentic_local_gpu_memory_utilization",
        "query.agentic_local_tensor_parallel_size",
        "query.agentic_local_max_model_len",
        "query.agentic_local_max_num_seqs",
        "query.agentic_reasoning_effort",
        "query.agentic_backend_top_k",
        "query.agentic_react_max_steps",
        "query.agentic_text_truncation",
        "query.agentic_num_concurrent",
        "query.agentic_temperature",
    ):
        assert key in QUERY_OVERRIDE_PATHS


def test_build_query_request_populates_agentic() -> None:
    request = build_query_request(
        _resolved(
            {
                "top_k": 10,
                "agentic": True,
                "agentic_llm_model": "test-model",
                "agentic_llm_backend": "openai_compatible",
                "agentic_invoke_url": "https://example.invalid/v1",
                "agentic_local_llm_backend": "vllm",
                "agentic_local_hf_cache_dir": "/tmp/hf",
                "agentic_local_device": "0",
                "agentic_local_gpu_memory_utilization": 0.6,
                "agentic_local_tensor_parallel_size": 2,
                "agentic_local_max_model_len": 8192,
                "agentic_local_max_num_seqs": 4,
                "agentic_reasoning_effort": "high",
                "agentic_backend_top_k": 25,
                "agentic_react_max_steps": 12,
                "agentic_text_truncation": 4000,
                "agentic_num_concurrent": 4,
                "agentic_temperature": 0.5,
            }
        ),
        "",
    )
    agentic = request.agentic
    assert agentic.enabled is True
    assert agentic.llm_model == "test-model"
    assert agentic.llm_backend == "openai_compatible"
    assert agentic.invoke_url == "https://example.invalid/v1"
    assert agentic.local_llm_backend == "vllm"
    assert agentic.local_hf_cache_dir == "/tmp/hf"
    assert agentic.local_device == "0"
    assert agentic.local_gpu_memory_utilization == pytest.approx(0.6)
    assert agentic.local_tensor_parallel_size == 2
    assert agentic.local_max_model_len == 8192
    assert agentic.local_max_num_seqs == 4
    assert agentic.reasoning_effort == "high"
    assert agentic.backend_top_k == 25
    assert agentic.react_max_steps == 12
    assert agentic.text_truncation == 4000
    assert agentic.num_concurrent == 4
    assert agentic.temperature == pytest.approx(0.5)


def test_build_query_request_agentic_defaults_when_absent() -> None:
    request = build_query_request(_resolved({"top_k": 10}), "")
    assert request.agentic == QueryAgenticOptions()
    assert request.agentic.enabled is False


def test_build_agentic_config_maps_request_and_top_k_override() -> None:
    request = QueryRequest(
        query="q",
        agentic=QueryAgenticOptions(
            enabled=True,
            llm_model="test-model",
            llm_backend="openai_compatible",
            backend_top_k=20,
            num_concurrent=4,
            temperature=0.0,
        ),
    )
    cfg = build_agentic_config(request, top_k=10)
    assert cfg.llm_model == "test-model"
    assert cfg.top_k == 10  # harness sets this to the deepest BEIR k
    assert cfg.backend_top_k == 20
    assert cfg.num_concurrent == 4


def test_build_agentic_config_defaults_to_local_vllm_nemotron_8b() -> None:
    request = QueryRequest(query="q", agentic=QueryAgenticOptions(enabled=True))

    cfg = build_agentic_config(request, top_k=10)

    assert cfg.llm_backend == "in_process"
    assert cfg.local_llm_backend == "vllm"
    assert cfg.llm_model == "nemotron-8b"


def test_run_beir_queries_routes_to_agentic(tmp_path) -> None:
    # Stage 1 (shared) load and stage 3 (shared) scoring/artifacts run for real;
    # only stage 2 (the agent retrieve+rank) is stubbed at the agentic_beir_retrieve
    # seam. This proves the agentic path reuses the SAME load + run schema + metrics
    # + artifact writing as dense, while forking only at retrieval.
    writer = ArtifactWriter(artifact_dir=tmp_path, run_id="run-1", benchmark="bench")
    resolved = {
        "evaluation": {
            "mode": "beir",
            "loader": "vidore",
            "dataset_name": "demo",
            "doc_id_field": "pdf_basename",
            "ks": [1, 5, 10],
        },
        "query": {},
    }
    request = build_query_request(
        _resolved(
            {
                "top_k": 10,
                "agentic": True,
                "agentic_llm_model": "test-model",
                "agentic_llm_backend": "openai_compatible",
            }
        ),
        "",
    )

    dataset = BeirDataset(
        dataset_name="demo",
        query_ids=["q1", "q2"],
        queries=["text1", "text2"],
        qrels={"q1": {"d1": 1}, "q2": {"d2": 1}},
    )
    result = pd.DataFrame(
        [
            {"query_id": "q1", "doc_id": "d1", "rank": 1, "result_source": "rrf"},
            {"query_id": "q2", "doc_id": "d2", "rank": 1, "result_source": "rrf"},
        ]
    )
    ranked_doc_ids = [["d1"], ["d2"]]  # perfect retrieval -> recall@k == 1.0

    with patch("nemo_retriever.harness.beir_runner.load_beir_dataset", return_value=dataset) as mocked_load, patch(
        "nemo_retriever.query.agentic.agentic_beir_retrieve",
        return_value=(result, ranked_doc_ids),
    ) as mocked_agentic:
        latencies, metrics, count = run_beir_queries(writer, resolved, None, request)

    # Stage 1: shared loader called with the resolved evaluation args.
    assert mocked_load.call_args.kwargs["doc_id_field"] == "pdf_basename"
    # Stage 2: forked into the agentic batch (dense query_plan unused -> None here),
    # with cfg.top_k overridden to the deepest BEIR k.
    assert mocked_agentic.call_count == 1
    assert mocked_agentic.call_args.kwargs["doc_id_field"] == "pdf_basename"
    assert mocked_agentic.call_args.args[1].top_k == 10  # cfg, top_k = max(ks)
    # Stage 3: shared real scoring over the agentic run + artifacts.
    assert metrics["recall@10"] == pytest.approx(1.0)
    assert count == 2
    assert len(latencies) == 2
    assert (tmp_path / "beir_metrics.json").exists()
    assert (tmp_path / "beir_run.trec").exists()
    assert (tmp_path / "query_results.jsonl").exists()


def test_run_beir_queries_invalid_agentic_config_is_structured_failure(tmp_path) -> None:
    # An invalid agentic config (backend_top_k below the target top_k = max(ks))
    # must surface as a structured HarnessRunError, not a raw ValueError.
    from nemo_retriever.harness.contracts import EXIT_INVALID, HarnessRunError

    writer = ArtifactWriter(artifact_dir=tmp_path, run_id="run-bad", benchmark="bench")
    resolved = {
        "evaluation": {"mode": "beir", "loader": "vidore", "dataset_name": "demo", "ks": [1, 5, 10]},
        "query": {},
    }
    request = build_query_request(
        _resolved({"top_k": 10, "agentic": True, "agentic_llm_model": "nemotron-8b", "agentic_backend_top_k": 5}),
        "",
    )
    dataset = BeirDataset(dataset_name="demo", query_ids=["q1"], queries=["t1"], qrels={"q1": {"d1": 1}})
    with patch("nemo_retriever.harness.beir_runner.load_beir_dataset", return_value=dataset):
        with pytest.raises(HarnessRunError) as excinfo:
            run_beir_queries(writer, resolved, None, request)
    assert excinfo.value.exit_code == EXIT_INVALID
    assert excinfo.value.failure.failure_reason == "invalid_agentic_config"
