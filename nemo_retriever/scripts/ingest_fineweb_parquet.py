#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embed projected FineWeb Parquet rows into Parquet staging or a dense LanceDB index."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from nemo_retriever.common.params import EmbedParams, ModelRuntimeParams
from nemo_retriever.common.ray_runtime import ensure_local_ray_runtime
from nemo_retriever.graph.executor import RayDataExecutor
from nemo_retriever.graph.pipeline_graph import Graph
from nemo_retriever.operators.embed.continuous_batch import (
    AsyncVllmEmbeddingEngine,
    ContinuousBatchEmbedActor,
    ContinuousBatchEngineRouter,
)
from nemo_retriever.operators.graph_ops.embedding_parquet_operator import EmbeddingParquetActor
from nemo_retriever.operators.graph_ops.parquet_text_operator import ParquetTextActor
from nemo_retriever.operators.vdb import IngestVdbOperator

DEFAULT_INPUT = "/data/fineweb/deduplicated"
DEFAULT_OUTPUT = "/data/fineweb/lancedb"
DEFAULT_MODEL = "nvidia/llama-nemotron-embed-1b-v2"
DEFAULT_MODEL_REVISION = "113abe4acafa848e77ead9c0623205e511932348"
DEFAULT_OPERATION_ID = "fineweb-dedup-dense-20260924-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--embedding-parquet-output", default=None)
    parser.add_argument(
        "--append-embedding-parquet",
        action="store_true",
        help="Append to an existing embedding Parquet dataset instead of requiring a new directory.",
    )
    parser.add_argument("--parquet-write-workers", type=int, default=32)
    parser.add_argument("--parquet-min-rows-per-file", type=int, default=32768)
    parser.add_argument("--parquet-max-rows-per-file", type=int, default=65536)
    parser.add_argument("--table-name", default="fineweb_edu_fortified")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--vector-dim", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--engine-max-model-len", type=int, default=None)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--engines-per-gpu", type=int, default=2)
    parser.add_argument("--embed-cpu-workers", type=int, default=64)
    parser.add_argument("--vdb-prepare-workers", type=int, default=32)
    parser.add_argument("--embed-batch-size", type=int, default=512)
    parser.add_argument("--inference-batch-size", type=int, default=512)
    parser.add_argument("--engine-cpus", type=int, default=2)
    parser.add_argument("--engine-max-concurrency", type=int, default=16)
    parser.add_argument("--engine-max-num-seqs", type=int, default=512)
    parser.add_argument("--pooler-use-activation", action="store_true")
    parser.add_argument("--source-workers", type=int, default=8)
    parser.add_argument("--read-workers", type=int, default=2)
    parser.add_argument("--source-batch-size", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--num-partitions", type=int, default=None)
    parser.add_argument("--target-partition-size", type=int, default=None)
    parser.add_argument("--num-sub-vectors", type=int, default=256)
    parser.add_argument("--stream-batch-mib", type=int, default=256)
    parser.add_argument("--stream-prefetch-batches", type=int, default=8)
    parser.add_argument("--stream-queue-size", type=int, default=65536)
    parser.add_argument("--stream-spool-directory", default="/data/fineweb/vdb_spool")
    parser.add_argument("--stream-spool-max-gib", type=int, default=8192)
    parser.add_argument("--operation-id", default=DEFAULT_OPERATION_ID)
    parser.add_argument("--no-durable-operation", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--ray-data-execution-memory-gib", type=int, default=250)
    parser.add_argument("--ray-object-store-memory-gib", type=int, default=320)
    parser.add_argument("--downstream-capacity-ratio", type=float, default=1.0)
    parser.add_argument("--downstream-backpressure-threshold", type=float, default=0.0)
    parser.add_argument("--target-block-mib", type=int, default=32)
    parser.add_argument("--no-build-index", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpus <= 0:
        raise ValueError("--gpus must be positive")
    if args.engines_per_gpu <= 0:
        raise ValueError("--engines-per-gpu must be positive")
    if args.embed_cpu_workers <= 0:
        raise ValueError("--embed-cpu-workers must be positive")
    if args.vdb_prepare_workers <= 0:
        raise ValueError("--vdb-prepare-workers must be positive")
    if args.engine_cpus <= 0:
        raise ValueError("--engine-cpus must be positive")
    if args.engine_max_concurrency <= 0:
        raise ValueError("--engine-max-concurrency must be positive")
    if args.engine_max_num_seqs <= 0:
        raise ValueError("--engine-max-num-seqs must be positive")
    if args.engine_max_model_len is not None and args.engine_max_model_len < args.max_length:
        raise ValueError("--engine-max-model-len must be at least --max-length")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("--gpu-memory-utilization must be between 0 and 1")
    if args.ray_data_execution_memory_gib <= 0:
        raise ValueError("--ray-data-execution-memory-gib must be positive")
    if args.ray_object_store_memory_gib < args.ray_data_execution_memory_gib:
        raise ValueError("--ray-object-store-memory-gib must be at least the Ray Data execution budget")
    if args.downstream_capacity_ratio <= 0:
        raise ValueError("--downstream-capacity-ratio must be positive")
    if not 0 <= args.downstream_backpressure_threshold <= 1:
        raise ValueError("--downstream-backpressure-threshold must be between 0 and 1")
    if args.target_block_mib <= 0:
        raise ValueError("--target-block-mib must be positive")
    if args.parquet_write_workers <= 0:
        raise ValueError("--parquet-write-workers must be positive")
    if args.parquet_min_rows_per_file <= 0:
        raise ValueError("--parquet-min-rows-per-file must be positive")
    if args.parquet_max_rows_per_file < args.parquet_min_rows_per_file:
        raise ValueError("--parquet-max-rows-per-file must be at least --parquet-min-rows-per-file")
    if args.stream_prefetch_batches <= 0:
        raise ValueError("--stream-prefetch-batches must be positive")
    if args.stream_queue_size <= 0:
        raise ValueError("--stream-queue-size must be positive")
    if args.stream_spool_max_gib <= 0:
        raise ValueError("--stream-spool-max-gib must be positive")
    if args.num_partitions is not None and args.target_partition_size is not None:
        raise ValueError("Pass only one of --num-partitions or --target-partition-size")
    if args.target_partition_size is not None and args.target_partition_size <= 0:
        raise ValueError("--target-partition-size must be positive")
    if args.no_durable_operation and args.append:
        raise ValueError("--no-durable-operation is only supported for overwrite ingestion")
    if args.append and args.operation_id == DEFAULT_OPERATION_ID:
        raise ValueError("--append requires a caller-managed unique --operation-id")
    stream_operation_id = None if args.no_durable_operation else args.operation_id

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    embedding_parquet_output = Path(args.embedding_parquet_output) if args.embedding_parquet_output else None
    if embedding_parquet_output is not None and embedding_parquet_output.exists() and not args.append_embedding_parquet:
        raise FileExistsError(
            f"Embedding Parquet output already exists: {embedding_parquet_output}. "
            "Use a new path or explicitly remove the incomplete output before retrying."
        )

    parquet = ParquetTextActor()
    columns = parquet.projected_columns()
    engine_count = args.gpus * args.engines_per_gpu
    per_engine_gpu_memory_utilization = args.gpu_memory_utilization / args.engines_per_gpu
    configuration = {
        "input": str(input_path),
        "columns": columns,
        "output": args.output,
        "embedding_parquet_output": str(embedding_parquet_output) if embedding_parquet_output else None,
        "append_embedding_parquet": args.append_embedding_parquet,
        "parquet_write_workers": args.parquet_write_workers,
        "parquet_min_rows_per_file": args.parquet_min_rows_per_file,
        "parquet_max_rows_per_file": args.parquet_max_rows_per_file,
        "table_name": args.table_name,
        "retrieval_mode": "dense",
        "model": args.model,
        "model_revision": args.model_revision,
        "vector_dim": args.vector_dim,
        "max_length": args.max_length,
        "engine_max_model_len": args.engine_max_model_len or args.max_length,
        "gpus": args.gpus,
        "embedding_architecture": "continuous_async_vllm",
        "gpu_engine_actors": engine_count,
        "engines_per_gpu": args.engines_per_gpu,
        "per_engine_gpu_memory_utilization": per_engine_gpu_memory_utilization,
        "embed_cpu_workers": args.embed_cpu_workers,
        "vdb_prepare_workers": args.vdb_prepare_workers,
        "embed_batch_size": args.embed_batch_size,
        "inference_batch_size": args.inference_batch_size,
        "engine_cpus": args.engine_cpus,
        "engine_max_concurrency": args.engine_max_concurrency,
        "engine_max_num_seqs": args.engine_max_num_seqs,
        "pooler_use_activation": args.pooler_use_activation,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "parquet_readers": args.read_workers,
        "ray_data_execution_memory_gib": args.ray_data_execution_memory_gib,
        "ray_object_store_memory_gib": args.ray_object_store_memory_gib,
        "downstream_capacity_ratio": args.downstream_capacity_ratio,
        "downstream_backpressure_threshold": args.downstream_backpressure_threshold,
        "target_block_mib": args.target_block_mib,
        "stream_prefetch_batches": args.stream_prefetch_batches,
        "stream_queue_size": args.stream_queue_size,
        "stream_spool_directory": args.stream_spool_directory,
        "stream_spool_max_gib": args.stream_spool_max_gib,
        "build_index": not args.no_build_index,
        "overwrite": not args.append,
        "operation_id": stream_operation_id,
        "durable_operation": stream_operation_id is not None,
        "limit": args.limit,
    }
    print(json.dumps(configuration, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return

    ray = ensure_local_ray_runtime(
        args.ray_address,
        log_to_driver=True,
        object_store_memory=args.ray_object_store_memory_gib << 30,
    )
    import ray.data as rd
    from ray.data import DataContext
    from ray.data._internal.execution.backpressure_policy.downstream_capacity_backpressure_policy import (
        DownstreamCapacityBackpressurePolicy,
    )
    from ray.data._internal.execution.interfaces.execution_options import ExecutionResources

    data_context = DataContext.get_current()
    data_context.override_object_store_memory_limit_fraction = 1.0
    data_context.execution_options.resource_limits = ExecutionResources.for_limits(
        object_store_memory=args.ray_data_execution_memory_gib << 30
    )
    data_context.downstream_capacity_backpressure_ratio = args.downstream_capacity_ratio
    DownstreamCapacityBackpressurePolicy.OBJECT_STORE_BUDGET_UTIL_THRESHOLD = args.downstream_backpressure_threshold
    data_context.target_max_block_size = args.target_block_mib << 20
    data_context.parquet_chunker_target_chunk_size = args.target_block_mib << 20

    available_gpus = int(ray.cluster_resources().get("GPU", 0))
    if available_gpus < args.gpus:
        raise RuntimeError(f"Requested {args.gpus} GPU actors, but Ray reports {available_gpus} GPUs")

    dataset = rd.read_parquet(
        str(input_path),
        columns=columns,
        concurrency=min(args.read_workers, args.gpus) if args.limit is not None else args.read_workers,
        override_num_blocks=max(32, args.gpus * 4) if args.limit is not None else None,
    )
    if args.limit is not None:
        dataset = dataset.limit(args.limit).repartition(num_blocks=min(max(32, args.embed_cpu_workers * 2), args.limit))

    engine_type = ray.remote(
        num_cpus=args.engine_cpus,
        num_gpus=1 / args.engines_per_gpu,
        max_concurrency=args.engine_max_concurrency,
    )(AsyncVllmEmbeddingEngine)
    engines = [
        engine_type.remote(
            engine_index=index,
            model=args.model,
            revision=args.model_revision,
            max_model_len=args.engine_max_model_len or args.max_length,
            gpu_memory_utilization=per_engine_gpu_memory_utilization,
            dimensions=args.vector_dim,
            max_num_seqs=args.engine_max_num_seqs,
            pooler_use_activation=True if args.pooler_use_activation else None,
        )
        for index in range(engine_count)
    ]
    router_type = ray.remote(num_cpus=0)(ContinuousBatchEngineRouter)
    router = router_type.remote(engine_count)
    ray.get([engine.ready.remote() for engine in engines])

    embed_params = EmbedParams(
        model_name=args.model,
        embed_model_name=args.model,
        embed_model_revision=args.model_revision,
        input_type="passage",
        text_column="text",
        inference_batch_size=args.inference_batch_size,
        embed_inference_batch_size=args.inference_batch_size,
        runtime=ModelRuntimeParams(
            hf_cache_dir=args.hf_cache_dir,
            max_length=args.max_length,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=False,
        ),
    )
    embed = ContinuousBatchEmbedActor(embed_params, engines, router)
    graph = Graph()
    if embedding_parquet_output is not None:
        parquet_output = EmbeddingParquetActor(
            vector_dim=args.vector_dim,
            embedding_model=args.model,
            embedding_revision=args.model_revision,
        )
        graph.add_chain(parquet, embed, parquet_output)
    else:
        sink = IngestVdbOperator(
            vdb_op="lancedb",
            vdb_kwargs={
                "uri": args.output,
                "table_name": args.table_name,
                "overwrite": not args.append,
                "hybrid": False,
                "sparse": False,
                "embedding_model_name": args.model,
                "embedding_model_revision": args.model_revision,
                "vector_dim": args.vector_dim,
                "index_type": "IVF_HNSW_SQ",
                "metric": "cosine",
                "num_partitions": args.num_partitions,
                "target_partition_size": args.target_partition_size,
                "num_sub_vectors": args.num_sub_vectors,
                "build_index": not args.no_build_index,
                "stream_batch_bytes": args.stream_batch_mib << 20,
                "stream_optimize": False,
                "stream_operation_id": stream_operation_id,
            },
        )
        graph.add_chain(parquet, embed, sink)

    node_overrides = {
        "ParquetTextActor": {
            "batch_size": args.source_batch_size,
            "batch_format": "pyarrow",
            "concurrency": args.source_workers,
            "num_cpus": 1,
            "num_gpus": 0,
        },
        "ContinuousBatchEmbedActor": {
            "batch_size": args.embed_batch_size,
            "concurrency": args.embed_cpu_workers,
            "num_cpus": 1,
            "num_gpus": 0,
        },
        "EmbeddingParquetActor": {
            "batch_size": 2048,
            "batch_format": "pandas",
            "concurrency": args.parquet_write_workers,
            "num_cpus": 1,
            "num_gpus": 0,
        },
    }
    executor_kwargs = {
        "ray_address": args.ray_address,
        "batch_size": args.source_batch_size,
        "batch_format": "pandas",
        "num_cpus": 1,
        "node_overrides": node_overrides,
        "source_cpu_reservation": 1,
    }
    if embedding_parquet_output is None:
        executor_kwargs.update(
            retain_stream_ingest_output=False,
            stream_ingest_prepare_concurrency=args.vdb_prepare_workers,
            stream_ingest_prefetch_batches=args.stream_prefetch_batches,
            stream_ingest_queue_size=args.stream_queue_size,
            stream_ingest_spool_directory=args.stream_spool_directory,
            stream_ingest_spool_max_bytes=args.stream_spool_max_gib << 30,
        )
    executor = RayDataExecutor(graph, **executor_kwargs)
    try:
        if embedding_parquet_output is not None:
            embedded_dataset = executor.build_dataset(dataset)
            embedded_dataset.write_parquet(
                str(embedding_parquet_output),
                concurrency=args.parquet_write_workers,
                min_rows_per_file=args.parquet_min_rows_per_file,
                max_rows_per_file=args.parquet_max_rows_per_file,
                mode=rd.SaveMode.APPEND if args.append_embedding_parquet else rd.SaveMode.ERROR,
                compression="zstd",
                compression_level=1,
                use_dictionary=["dump", "embedding_model", "embedding_revision"],
                write_statistics=True,
            )
            logging.getLogger(__name__).info(
                "Embedding Parquet generation finished: output=%s", embedding_parquet_output
            )
        else:
            result = executor.ingest(dataset)
            logging.getLogger(__name__).info(
                "Dense ingest finished: result_rows=%d sink_stats=%s",
                len(result),
                getattr(executor, "_last_stream_ingest_stats", None),
            )
    finally:
        for engine in engines:
            ray.kill(engine, no_restart=True)
        ray.kill(router, no_restart=True)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
