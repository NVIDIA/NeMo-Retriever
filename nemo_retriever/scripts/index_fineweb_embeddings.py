#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load staged embedding Parquet through the NRL VDB operator and build LanceDB."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from nemo_retriever.common.ray_runtime import ensure_local_ray_runtime

DEFAULT_INPUT = "/data/fineweb/embeddings_parquet"
DEFAULT_OUTPUT = "/data/fineweb/lancedb_gate_10m"
DEFAULT_TEMP_DIRECTORY = "/data/fineweb/lancedb_tmp"
DEFAULT_RAY_TEMP_DIRECTORY = "/tmp/nrl-ray"
DEFAULT_MODEL = "nvidia/llama-nemotron-embed-1b-v2"
DEFAULT_MODEL_REVISION = "113abe4acafa848e77ead9c0623205e511932348"
PROJECTED_COLUMNS = ["id", "document_id", "text", "url", "file_path", "dump", "source_id", "vector"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--table-name", default="fineweb_edu_fortified")
    parser.add_argument("--temp-directory", default=DEFAULT_TEMP_DIRECTORY)
    parser.add_argument("--ray-temp-directory", default=DEFAULT_RAY_TEMP_DIRECTORY)
    parser.add_argument("--limit", type=int, default=10_000_000)
    parser.add_argument("--full-corpus", action="store_true")
    parser.add_argument("--vector-dim", type=int, default=2048)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--embedding-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--metric", choices=("l2", "cosine", "dot"), default="cosine")
    partitions = parser.add_mutually_exclusive_group()
    partitions.add_argument("--num-partitions", type=int, default=None)
    partitions.add_argument("--target-partition-size", type=int, default=1_048_576)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--sample-rate", type=int, default=256)
    parser.add_argument("--hnsw-m", type=int, default=20)
    parser.add_argument("--hnsw-ef-construction", type=int, default=300)
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
        "query_index_cache_gib",
        "query_metadata_cache_mib",
        "validation_nprobes",
        "validation_refine_factor",
        "read_workers",
        "read_blocks",
        "prepare_workers",
        "stream_prefetch_batches",
        "stream_queue_size",
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
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    Path(args.temp_directory).mkdir(parents=True, exist_ok=True)
    Path(args.ray_temp_directory).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    validate_args(args)
    config = {
        **vars(args),
        "projected_columns": PROJECTED_COLUMNS,
        "index_type": "IVF_HNSW_SQ",
        "input_format": "embedding_parquet",
        "started_at": _utc_now(),
    }
    print("GATE_CONFIG " + json.dumps(config, sort_keys=True), flush=True)
    if args.dry_run:
        return

    timing_path = Path(args.output).parent / f"{Path(args.output).name}_timings.jsonl"
    os.environ["NV_INGEST_LANCEDB_TIMING_PATH"] = str(timing_path)
    os.environ["TMPDIR"] = str(Path(args.temp_directory).resolve())
    started = time.perf_counter()

    ray = ensure_local_ray_runtime(
        None,
        log_to_driver=True,
        object_store_memory=args.ray_object_store_memory_gib << 30,
        temp_dir=str(Path(args.ray_temp_directory).resolve()),
    )
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
    print(f"GATE_DATASET_READY timestamp={_utc_now()} sample_rows={len(samples)}", flush=True)

    sink = IngestVdbOperator(
        vdb_op="lancedb",
        vdb_kwargs={
            "uri": args.output,
            "table_name": args.table_name,
            "overwrite": True,
            "hybrid": False,
            "sparse": False,
            "input_format": "embedding_parquet",
            "embedding_model_name": args.embedding_model,
            "embedding_model_revision": args.embedding_revision,
            "vector_dim": args.vector_dim,
            "index_type": "IVF_HNSW_SQ",
            "metric": args.metric,
            "num_partitions": args.num_partitions,
            "target_partition_size": args.target_partition_size,
            "max_iterations": args.max_iterations,
            "sample_rate": args.sample_rate,
            "hnsw_m": args.hnsw_m,
            "hnsw_ef_construction": args.hnsw_ef_construction,
            "index_accelerator": args.index_accelerator,
            "build_index": not args.no_build_index,
            "stream_optimize": False,
            "stream_operation_id": None,
        },
    )
    graph = Graph() >> sink
    executor = RayDataExecutor(
        graph,
        retain_stream_ingest_output=False,
        stream_ingest_prepare_concurrency=args.prepare_workers,
        stream_ingest_prefetch_batches=args.stream_prefetch_batches,
        stream_ingest_queue_size=args.stream_queue_size,
        source_cpu_reservation=1,
    )
    executor.ingest(dataset)
    elapsed = time.perf_counter() - started

    import lancedb

    query_session = lancedb.Session(
        index_cache_size_bytes=args.query_index_cache_gib << 30,
        metadata_cache_size_bytes=args.query_metadata_cache_mib << 20,
    )
    table = lancedb.connect(args.output, session=query_session).open_table(args.table_name)
    rows = int(table.count_rows())
    if rows != expected_rows:
        raise RuntimeError(f"Index row count mismatch: expected {expected_rows}, got {rows}")
    indices = table.list_indices()
    if not args.no_build_index and not indices:
        raise RuntimeError("Gate completed without a vector index")

    self_rank_one = 0
    self_top_ten = 0
    cold_query_seconds = []
    warm_query_seconds = []
    for sample in samples:
        sample_rank = None
        for attempt in range(2):
            query_started = time.perf_counter()
            result = (
                table.search(sample["vector"], vector_column_name="vector")
                .distance_type(args.metric)
                .nprobes(args.validation_nprobes)
                .refine_factor(args.validation_refine_factor)
                .select(["id", "_distance"])
                .limit(10)
                .to_list()
            )
            query_seconds = time.perf_counter() - query_started
            if attempt == 0:
                cold_query_seconds.append(query_seconds)
                sample_rank = next(
                    (rank for rank, hit in enumerate(result, start=1) if hit.get("id") == sample["id"]),
                    None,
                )
            else:
                warm_query_seconds.append(query_seconds)
        self_rank_one += int(sample_rank == 1)
        self_top_ten += int(sample_rank is not None)
    if self_top_ten != len(samples):
        raise RuntimeError(f"Self-retrieval top-10 validation failed: {self_top_ten}/{len(samples)}")

    index_summaries = []
    for item in indices:
        index_summaries.append(
            {
                "name": item.name,
                "type": item.index_type,
                "rows": item.num_indexed_rows,
                "unindexed_rows": item.num_unindexed_rows,
                "size_bytes": item.size_bytes,
                "details": item.index_details,
            }
        )
    result = {
        "completed_at": _utc_now(),
        "elapsed_seconds": elapsed,
        "rows": rows,
        "rows_per_second": rows / elapsed,
        "self_retrieval_rank_one": f"{self_rank_one}/{len(samples)}",
        "self_retrieval_top_ten": f"{self_top_ten}/{len(samples)}",
        "cold_query_latency_ms": [round(value * 1000, 3) for value in cold_query_seconds],
        "warm_query_latency_ms": [round(value * 1000, 3) for value in warm_query_seconds],
        "query_session": str(query_session),
        "index": index_summaries,
        "sink_stats": executor._last_stream_ingest_stats,
        "timing_path": str(timing_path),
    }
    print("GATE_RESULT " + json.dumps(result, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
