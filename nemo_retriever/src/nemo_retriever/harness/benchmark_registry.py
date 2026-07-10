# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from nemo_retriever.harness.benchmark_specs import BenchmarkSpec, DatasetSpec, RunSet

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_EMBED_MODEL = "nvidia/llama-nemotron-embed-1b-v2"
VIDORE_V3_EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"
DEFAULT_TABLE_NAME = "nemo-retriever"
VIDORE_V3_PUBLIC_DATASETS: dict[str, str] = {
    "vidore_v3_computer_science": "computer science",
    "vidore_v3_energy": "energy",
    "vidore_v3_finance_en": "English finance",
    "vidore_v3_finance_fr": "French finance",
    "vidore_v3_hr": "human resources",
    "vidore_v3_industrial": "industrial",
    "vidore_v3_pharmaceuticals": "pharmaceuticals",
    "vidore_v3_physics": "physics",
}
DEFAULT_SUMMARY_KEYS: tuple[str, ...] = (
    "files",
    "pages",
    "rows_processed",
    "ingest_secs",
    "pages_per_sec_ingest",
    "query_count",
    "query_latency_p50_ms",
    "query_latency_p95_ms",
    "ndcg_10",
    "recall_5",
    "recall_10",
)


def _data_path(name: str) -> str:
    return str(Path("data") / name)


def _vidore_v3_dataset(name: str, domain: str) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        path=f"/datasets/nv-ingest/vidore_v3/{name}",
        query_file=None,
        input_type="pdf",
        beir_loader="vidore_hf",
        beir_doc_id_field="pdf_page",
        description=f"ViDoRe v3 {domain} benchmark slice.",
    )


DATASETS: dict[str, DatasetSpec] = {
    "jp20": DatasetSpec(
        name="jp20",
        path="/datasets/nv-ingest/jp20",
        query_file=_data_path("jp20_query_gt.csv"),
        input_type="pdf",
        beir_loader="jp20_csv",
        beir_doc_id_field="pdf_page",
        description="20-PDF JP Morgan benchmark corpus.",
    ),
    "bo20": DatasetSpec(
        name="bo20",
        path="/datasets/nv-ingest/bo20",
        query_file=None,
        input_type="pdf",
        description="20-PDF benchmark smoke corpus without BEIR qrels.",
    ),
    "bo767": DatasetSpec(
        name="bo767",
        path="/datasets/nv-ingest/bo767",
        query_file=_data_path("bo767_query_gt.csv"),
        input_type="pdf",
        beir_loader="bo767_csv",
        beir_doc_id_field="pdf_page",
        description="767-PDF Business Objects benchmark corpus.",
    ),
    "financebench": DatasetSpec(
        name="financebench",
        path="/datasets/nv-ingest/foundation_rag/financebench",
        query_file=_data_path("financebench_train.json"),
        input_type="pdf",
        beir_loader="financebench_json",
        beir_doc_id_field="pdf_basename",
        description="FinanceBench PDF retrieval benchmark.",
    ),
    "bo10k": DatasetSpec(
        name="bo10k",
        path="/datasets/nv-ingest/bo10k",
        query_file=_data_path("digital_corpora_10k_annotations.csv"),
        input_type="pdf",
        beir_loader="bo10k_csv",
        beir_doc_id_field="pdf_page",
        description="10k-PDF Business Objects benchmark corpus.",
    ),
    "earnings_consulting": DatasetSpec(
        name="earnings_consulting",
        path="/datasets/nv-ingest/earnings_consulting_flattened",
        query_file=_data_path("earnings_consulting_multimodal.csv"),
        input_type="pdf",
        beir_loader="earnings_csv",
        beir_doc_id_field="pdf_page",
        description="Earnings consulting multimodal benchmark corpus.",
    ),
    **{name: _vidore_v3_dataset(name, domain) for name, domain in VIDORE_V3_PUBLIC_DATASETS.items()},
}


def _base_ingest(*, profile: str = "auto") -> dict[str, Any]:
    return {
        "profile": profile,
        "input_type": "pdf",
        "run_mode": "inprocess",
        "extract": {},
        "media": {},
        "caption": {"enabled": False},
        "dedup": {"enabled": False},
        "chunk": {"enabled": False},
        "embed": {"embed_model_name": DEFAULT_EMBED_MODEL},
        "image_store": {},
        "storage": {
            "table_name": DEFAULT_TABLE_NAME,
            "overwrite": True,
            "index_mode": "dense",
        },
    }


def _base_query(*, top_k: int = 10, embed_model_name: str = DEFAULT_EMBED_MODEL) -> dict[str, Any]:
    return {
        "top_k": top_k,
        "candidate_k": None,
        "page_dedup": False,
        "content_types": None,
        "retrieval_mode": "auto",
        "embed_model_name": embed_model_name,
        "embed_invoke_url": None,
        "rerank": False,
        "reranker_invoke_url": None,
        "reranker_model_name": None,
        "reranker_backend": None,
    }


def _beir_eval(dataset: DatasetSpec) -> dict[str, Any]:
    return {
        "mode": "beir",
        "loader": dataset.beir_loader,
        "dataset_name": dataset.query_file,
        "split": "test",
        "query_language": None,
        "doc_id_field": dataset.beir_doc_id_field,
        "ks": dataset.beir_ks,
    }


def _vidore_v3_ingest() -> dict[str, Any]:
    ingest = _base_ingest(profile="auto")
    ingest["extract"] = {
        "extract_infographics": True,
        "extract_page_as_image": True,
    }
    ingest["embed"] = {
        "embed_model_name": VIDORE_V3_EMBED_MODEL,
        "embed_modality": "text_image",
        "embed_granularity": "page",
    }
    return ingest


def _vidore_v3_benchmark(dataset_name: str) -> BenchmarkSpec:
    dataset = DATASETS[dataset_name]
    domain = VIDORE_V3_PUBLIC_DATASETS[dataset_name]
    domain_tags = ("finance",) if "finance" in domain.lower() else ()
    return BenchmarkSpec(
        name=f"{dataset_name}_beir",
        dataset=dataset_name,
        ingest=_vidore_v3_ingest(),
        query=_base_query(top_k=10, embed_model_name=VIDORE_V3_EMBED_MODEL),
        evaluation={
            **_beir_eval(dataset),
            "dataset_name": dataset_name,
        },
        summary_keys=DEFAULT_SUMMARY_KEYS,
        tags=("beir", "vidore", *domain_tags, "pdf"),
        description=f"ViDoRe v3 {domain} BEIR retrieval benchmark.",
    )


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "jp20_smoke": BenchmarkSpec(
        name="jp20_smoke",
        dataset="jp20",
        ingest=_base_ingest(profile="fast-text"),
        query=_base_query(top_k=5),
        evaluation={"mode": "none"},
        summary_keys=DEFAULT_SUMMARY_KEYS,
        tags=("smoke", "local", "pdf"),
        description="Fast-text ingest smoke run over jp20 without BEIR evaluation.",
    ),
    "jp20_beir": BenchmarkSpec(
        name="jp20_beir",
        dataset="jp20",
        ingest=_base_ingest(profile="auto"),
        query=_base_query(top_k=10),
        evaluation=_beir_eval(DATASETS["jp20"]),
        summary_keys=DEFAULT_SUMMARY_KEYS,
        tags=("beir", "local", "pdf"),
        description="JP20 end-to-end BEIR retrieval benchmark.",
    ),
    "bo767_beir": BenchmarkSpec(
        name="bo767_beir",
        dataset="bo767",
        ingest=_base_ingest(profile="auto"),
        query=_base_query(top_k=10),
        evaluation=_beir_eval(DATASETS["bo767"]),
        summary_keys=DEFAULT_SUMMARY_KEYS,
        tags=("beir", "pdf"),
        description="BO767 end-to-end BEIR retrieval benchmark.",
    ),
    "financebench_beir": BenchmarkSpec(
        name="financebench_beir",
        dataset="financebench",
        ingest=_base_ingest(profile="auto"),
        query=_base_query(top_k=10),
        evaluation=_beir_eval(DATASETS["financebench"]),
        summary_keys=DEFAULT_SUMMARY_KEYS,
        tags=("beir", "finance", "pdf"),
        description="FinanceBench end-to-end BEIR retrieval benchmark.",
    ),
    "earnings_beir": BenchmarkSpec(
        name="earnings_beir",
        dataset="earnings_consulting",
        ingest=_base_ingest(profile="auto"),
        query=_base_query(top_k=10),
        evaluation=_beir_eval(DATASETS["earnings_consulting"]),
        summary_keys=DEFAULT_SUMMARY_KEYS,
        tags=("beir", "earnings", "pdf"),
        description="Earnings consulting end-to-end BEIR retrieval benchmark.",
    ),
    **{f"{dataset_name}_beir": _vidore_v3_benchmark(dataset_name) for dataset_name in VIDORE_V3_PUBLIC_DATASETS},
    "bo10k_beir_fast_text": BenchmarkSpec(
        name="bo10k_beir_fast_text",
        dataset="bo10k",
        ingest=_base_ingest(profile="fast-text"),
        query=_base_query(top_k=10),
        evaluation=_beir_eval(DATASETS["bo10k"]),
        summary_keys=DEFAULT_SUMMARY_KEYS,
        tags=("beir", "fast-text", "large", "pdf"),
        description="BO10K fast-text BEIR benchmark for large-corpus validation.",
    ),
    "bo20_smoke": BenchmarkSpec(
        name="bo20_smoke",
        dataset="bo20",
        ingest=_base_ingest(profile="fast-text"),
        query=_base_query(top_k=5),
        evaluation={"mode": "none"},
        summary_keys=DEFAULT_SUMMARY_KEYS,
        tags=("smoke", "pdf"),
        description="BO20 fast-text smoke run without BEIR qrels.",
    ),
}

RUNSETS: dict[str, RunSet] = {
    "jp20_core": RunSet(
        name="jp20_core",
        runs=("jp20_smoke", "jp20_beir"),
        tags=("jp20", "core"),
        description="Small jp20 smoke plus BEIR validation set.",
    ),
    "vidore_v3_all": RunSet(
        name="vidore_v3_all",
        runs=tuple(f"{dataset_name}_beir" for dataset_name in VIDORE_V3_PUBLIC_DATASETS),
        tags=("vidore", "beir", "all"),
        description="All eight public ViDoRe v3 BEIR retrieval benchmarks.",
    ),
}


def list_benchmarks() -> list[BenchmarkSpec]:
    return [BENCHMARKS[name] for name in sorted(BENCHMARKS)]


def list_runsets() -> list[RunSet]:
    return [RUNSETS[name] for name in sorted(RUNSETS)]


def get_dataset(name: str) -> DatasetSpec:
    return DATASETS[name]


def get_benchmark(name: str) -> BenchmarkSpec:
    return BENCHMARKS[name]


def get_runset(name: str) -> RunSet:
    return RUNSETS[name]


def benchmark_names() -> tuple[str, ...]:
    return tuple(sorted(BENCHMARKS))


def runset_names() -> tuple[str, ...]:
    return tuple(sorted(RUNSETS))


def benchmark_payload(spec: BenchmarkSpec) -> dict[str, Any]:
    dataset = get_dataset(spec.dataset)
    payload = spec.to_dict()
    payload["dataset_spec"] = dataset.to_dict()
    return deepcopy(payload)
