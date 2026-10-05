#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load staged embedding Parquet through the NRL Elasticsearch/cuVS operator."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from nemo_retriever.common.ray_runtime import ensure_local_ray_runtime

DEFAULT_INPUT = "/data/fineweb/embeddings_parquet"
DEFAULT_OUTPUT = "fineweb_embeddings_gate"
DEFAULT_URLS = [f"http://127.0.0.1:{port}" for port in range(9200, 9208)]
DEFAULT_TEMP_DIRECTORY = "/data/fineweb/elasticsearch_tmp"
DEFAULT_RAY_TEMP_DIRECTORY = "/tmp/nrl-ray"
DEFAULT_MODEL = "nvidia/nemotron-3.5-embed-1b-bf16-EA"
DEFAULT_MODEL_REVISION = "643532494f5aa0e5fb8541db602b364b8055af50"
PROJECTED_COLUMNS = ["id", "document_id", "text", "url", "file_path", "dump", "source_id", "vector"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finish_phase(phase_seconds: dict[str, float], name: str, started: float) -> None:
    elapsed = time.perf_counter() - started
    phase_seconds[name] = elapsed
    print(f"PHASE_END name={name} seconds={elapsed:.6f} timestamp={_utc_now()}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--index-name", default=DEFAULT_OUTPUT)
    parser.add_argument("--urls", nargs="+", default=DEFAULT_URLS)
    parser.add_argument("--index-type", choices=("int8_hnsw", "hnsw"), default="int8_hnsw")
    parser.add_argument("--num-candidates", type=int, default=10_000)
    parser.add_argument("--number-of-shards", type=int, default=8)
    parser.add_argument("--number-of-replicas", type=int, default=0)
    parser.add_argument("--bulk-workers", type=int, default=32)
    parser.add_argument("--bulk-queue-size", type=int, default=32)
    parser.add_argument("--bulk-batch-size", type=int, default=500)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--transport-max-retries", type=int, default=3)
    parser.add_argument("--bulk-max-retries", type=int, default=3)
    parser.add_argument("--bulk-initial-backoff", type=float, default=2.0)
    parser.add_argument("--bulk-max-backoff", type=float, default=600.0)
    parser.add_argument("--preserve-index-on-failure", action="store_true")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Upsert deterministic document IDs into an existing partial index.",
    )
    parser.add_argument("--temp-directory", default=DEFAULT_TEMP_DIRECTORY)
    parser.add_argument("--ray-temp-directory", default=DEFAULT_RAY_TEMP_DIRECTORY)
    parser.add_argument("--limit", type=int, default=10_000_000)
    parser.add_argument("--full-corpus", action="store_true")
    parser.add_argument("--vector-dim", type=int, default=2048)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--embedding-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--metric", choices=("l2_norm", "cosine", "dot_product", "max_inner_product"), default="cosine")
    partitions = parser.add_mutually_exclusive_group()
    partitions.add_argument("--num-partitions", type=int, default=None)
    partitions.add_argument("--target-partition-size", type=int, default=1_048_576)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--sample-rate", type=int, default=256)
    parser.add_argument("--hnsw-m", type=int, default=64)
    parser.add_argument("--hnsw-ef-construction", type=int, default=2000)
    parser.add_argument("--index-accelerator", default=None)
    parser.add_argument("--query-index-cache-gib", type=int, default=32)
    parser.add_argument("--query-metadata-cache-mib", type=int, default=1024)
    parser.add_argument("--validation-nprobes", type=int, default=1)
    parser.add_argument("--validation-refine-factor", type=int, default=10)
    parser.add_argument("--read-workers", type=int, default=32)
    parser.add_argument("--read-blocks", type=int, default=512)
    parser.add_argument("--prepare-workers", type=int, default=16)
    parser.add_argument("--stream-prefetch-batches", type=int, default=8)
    parser.add_argument("--stream-queue-size", type=int, default=64)
    parser.add_argument("--stream-spool-directory", default=None)
    parser.add_argument("--stream-spool-max-gib", type=int, default=None)
    parser.add_argument("--target-block-mib", type=int, default=128)
    parser.add_argument("--ray-data-execution-memory-gib", type=int, default=250)
    parser.add_argument("--ray-object-store-memory-gib", type=int, default=320)
    parser.add_argument("--no-build-index", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "limit",
        "vector_dim",
        "target_partition_size",
        "num_partitions",
        "max_iterations",
        "sample_rate",
        "hnsw_m",
        "hnsw_ef_construction",
        "num_candidates",
        "number_of_shards",
        "bulk_workers",
        "bulk_queue_size",
        "bulk_batch_size",
        "request_timeout",
        "transport_max_retries",
        "bulk_max_retries",
        "bulk_initial_backoff",
        "bulk_max_backoff",
        "query_index_cache_gib",
        "query_metadata_cache_mib",
        "validation_nprobes",
        "validation_refine_factor",
        "read_workers",
        "read_blocks",
        "prepare_workers",
        "stream_prefetch_batches",
        "stream_queue_size",
        "stream_spool_max_gib",
        "target_block_mib",
        "ray_data_execution_memory_gib",
        "ray_object_store_memory_gib",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.ray_object_store_memory_gib < args.ray_data_execution_memory_gib:
        raise ValueError("Ray object-store memory must be at least the Ray Data execution budget")
    if not Path(args.input).exists():
        raise FileNotFoundError(args.input)
    Path(args.temp_directory).mkdir(parents=True, exist_ok=True)
    Path(args.ray_temp_directory).mkdir(parents=True, exist_ok=True)
    if args.stream_spool_max_gib is not None and args.stream_spool_directory is None:
        raise ValueError("--stream-spool-max-gib requires --stream-spool-directory")
    if args.stream_spool_directory is not None:
        Path(args.stream_spool_directory).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    validate_args(args)
    config = {
        **vars(args),
        "projected_columns": PROJECTED_COLUMNS,
        "index_type": args.index_type,
        "input_format": "embedding_parquet",
        "started_at": _utc_now(),
    }
    print("GATE_CONFIG " + json.dumps(config, sort_keys=True), flush=True)
    if args.dry_run:
        return

    timing_path = Path(args.temp_directory) / f"{args.index_name}_timings.jsonl"
    os.environ["TMPDIR"] = str(Path(args.temp_directory).resolve())
    total_started = time.perf_counter()
    phase_seconds: dict[str, float] = {}

    phase_started = time.perf_counter()
    print(f"PHASE_START name=ray_startup timestamp={_utc_now()}", flush=True)
    ray = ensure_local_ray_runtime(
        None,
        log_to_driver=True,
        object_store_memory=args.ray_object_store_memory_gib << 30,
        temp_dir=str(Path(args.ray_temp_directory).resolve()),
    )
    _finish_phase(phase_seconds, "ray_startup", phase_started)

    import ray.data as rd
    from ray.data import DataContext

    from nemo_retriever.graph.executor import RayDataExecutor
    from nemo_retriever.graph.pipeline_graph import Graph
    from nemo_retriever.operators.vdb import IngestVdbOperator
    from ray.data._internal.execution.interfaces.execution_options import ExecutionResources

    context = DataContext.get_current()
    context.override_object_store_memory_limit_fraction = 1.0
    context.execution_options.resource_limits = ExecutionResources.for_limits(
        object_store_memory=args.ray_data_execution_memory_gib << 30
    )
    context.target_max_block_size = args.target_block_mib << 20
    context.parquet_chunker_target_chunk_size = args.target_block_mib << 20

    phase_started = time.perf_counter()
    print(f"PHASE_START name=dataset_preflight timestamp={_utc_now()}", flush=True)
    dataset = rd.read_parquet(
        args.input,
        columns=PROJECTED_COLUMNS,
        concurrency=args.read_workers,
        override_num_blocks=args.read_blocks,
    )
    if not args.full_corpus:
        dataset = dataset.limit(args.limit)
    import pyarrow.dataset as pa_dataset

    source_rows = int(pa_dataset.dataset(args.input, format="parquet").count_rows())
    expected_rows = source_rows if args.full_corpus else min(args.limit, source_rows)
    samples = dataset.take(8)
    print(f"GATE_DATASET_READY timestamp={_utc_now()} sample_rows={len(samples)} source_rows={source_rows}", flush=True)
    _finish_phase(phase_seconds, "dataset_preflight", phase_started)

    sink = IngestVdbOperator(
        vdb_op="elasticsearch",
        vdb_kwargs={
            "url": args.urls,
            "index_name": args.index_name,
            "overwrite": not args.resume_existing,
            "input_format": "embedding_parquet",
            "vector_dim": args.vector_dim,
            "index_type": args.index_type,
            "metric": args.metric,
            "hnsw_m": args.hnsw_m,
            "hnsw_ef_construction": args.hnsw_ef_construction,
            "num_candidates": args.num_candidates,
            "number_of_shards": args.number_of_shards,
            "number_of_replicas": args.number_of_replicas,
            "refresh_interval": "-1",
            "bulk_workers": args.bulk_workers,
            "bulk_queue_size": args.bulk_queue_size,
            "batch_size": args.bulk_batch_size,
            "request_timeout": args.request_timeout,
            "transport_max_retries": args.transport_max_retries,
            "retry_on_timeout": True,
            "bulk_max_retries": args.bulk_max_retries,
            "bulk_initial_backoff": args.bulk_initial_backoff,
            "bulk_max_backoff": args.bulk_max_backoff,
            "delete_index_on_failure": not args.preserve_index_on_failure,
            "encode_vectors_base64": True,
            "require_gpu": True,
            "index_mode": "vectordb_document",
            "use_auto_ids": False,
        },
    )
    graph = Graph() >> sink
    executor = RayDataExecutor(
        graph,
        retain_stream_ingest_output=False,
        stream_ingest_prepare_concurrency=args.prepare_workers,
        stream_ingest_prefetch_batches=args.stream_prefetch_batches,
        stream_ingest_queue_size=args.stream_queue_size,
        stream_ingest_spool_directory=args.stream_spool_directory,
        stream_ingest_spool_max_bytes=(
            args.stream_spool_max_gib << 30 if args.stream_spool_max_gib is not None else None
        ),
        source_cpu_reservation=1,
    )
    phase_started = time.perf_counter()
    print(
        f"PHASE_START name=stream_ingest_and_cuvs_build timestamp={_utc_now()} expected_rows={expected_rows}",
        flush=True,
    )
    executor.ingest(dataset)
    _finish_phase(phase_seconds, "stream_ingest_and_cuvs_build", phase_started)
    elapsed = time.perf_counter() - total_started

    phase_started = time.perf_counter()
    print(f"PHASE_START name=refresh_and_validation timestamp={_utc_now()}", flush=True)
    from elasticsearch import Elasticsearch as ElasticsearchClient

    client = ElasticsearchClient(
        args.urls,
        request_timeout=args.request_timeout,
        max_retries=args.transport_max_retries,
        retry_on_timeout=True,
    )
    client.indices.refresh(index=args.index_name)
    count_response = client.count(index=args.index_name)
    count_body = getattr(count_response, "body", count_response)
    rows = int(count_body["count"])
    if rows != expected_rows:
        raise RuntimeError(f"Index row count mismatch: expected {expected_rows}, got {rows}")
    mapping_response = client.indices.get_mapping(index=args.index_name)
    mapping_body = getattr(mapping_response, "body", mapping_response)
    vector_mapping = mapping_body[args.index_name]["mappings"]["properties"]["vector"]
    expected_options = {
        "type": args.index_type,
        "m": args.hnsw_m,
        "ef_construction": args.hnsw_ef_construction,
    }
    if vector_mapping.get("index_options") != expected_options:
        raise RuntimeError(
            f"Vector mapping mismatch: expected {expected_options}, got {vector_mapping.get("index_options")}"
        )

    self_rank_one = 0
    self_top_ten = 0
    cold_query_seconds = []
    warm_query_seconds = []
    for sample in samples:
        sample_rank = None
        for attempt in range(2):
            query_started = time.perf_counter()
            hits = sink._vdb.retrieval([sample["vector"]], top_k=10, num_candidates=args.num_candidates)[0]
            query_seconds = time.perf_counter() - query_started
            if attempt == 0:
                cold_query_seconds.append(query_seconds)
                sample_rank = next(
                    (rank for rank, hit in enumerate(hits, start=1) if hit.get("id") == sample["id"]),
                    None,
                )
            else:
                warm_query_seconds.append(query_seconds)
        self_rank_one += int(sample_rank == 1)
        self_top_ten += int(sample_rank is not None)
    if self_top_ten != len(samples):
        raise RuntimeError(f"Self-retrieval top-10 validation failed: {self_top_ten}/{len(samples)}")

    client.indices.put_settings(index=args.index_name, settings={"index": {"refresh_interval": "30s"}})
    health_response = client.cluster.health(index=args.index_name)
    health_body = getattr(health_response, "body", health_response)
    gpu_usage = sink._vdb.health()["gpu_vector_indexing"]
    _finish_phase(phase_seconds, "refresh_and_validation", phase_started)
    total_wall_seconds = time.perf_counter() - total_started
    result = {
        "completed_at": _utc_now(),
        "elapsed_seconds": elapsed,
        "total_wall_seconds": total_wall_seconds,
        "phase_seconds": phase_seconds,
        "rows": rows,
        "rows_per_second": rows / elapsed,
        "self_retrieval_rank_one": f"{self_rank_one}/{len(samples)}",
        "self_retrieval_top_ten": f"{self_top_ten}/{len(samples)}",
        "cold_query_latency_ms": [round(value * 1000, 3) for value in cold_query_seconds],
        "warm_query_latency_ms": [round(value * 1000, 3) for value in warm_query_seconds],
        "index_name": args.index_name,
        "vector_mapping": vector_mapping,
        "cluster_health": health_body,
        "gpu_vector_indexing": gpu_usage,
        "sink_stats": executor._last_stream_ingest_stats,
        "timing_path": str(timing_path),
    }
    print("GATE_RESULT " + json.dumps(result, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
