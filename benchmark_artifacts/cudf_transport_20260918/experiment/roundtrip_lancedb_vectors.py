"""Round-trip an NRL LanceDB vector column through a narrow cuDF GPU island."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import ray

from cudf_cuda_ipc import register_cudf_cuda_ipc


TABLE_NAME = "nemo-retriever"


def load_tables(uri: str):
    import lancedb
    import pyarrow as pa

    table = lancedb.connect(uri).open_table(TABLE_NAME).to_arrow()
    ordinals = pa.array(range(table.num_rows), type=pa.int64())
    fixed_vectors = table["vector"].combine_chunks()
    dimensions = fixed_vectors.type.list_size
    offsets = pa.array(
        range(0, (table.num_rows + 1) * dimensions, dimensions),
        type=pa.int64(),
    )
    # cuDF 26.08 does not import Arrow's fixed-size-list logical type, so use
    # an equivalent variable-list view while the values are on the GPU.
    gpu_vectors = pa.ListArray.from_arrays(offsets, fixed_vectors.values)
    vectors = pa.table({"row_id": ordinals, "vector": gpu_vectors})
    return table, vectors


def replace_vectors(full_table, gpu_frame):
    import pyarrow as pa

    vector_table = gpu_frame.to_arrow()
    if vector_table.num_rows != full_table.num_rows:
        raise ValueError(f"vector row count changed: {vector_table.num_rows} != {full_table.num_rows}")
    vector_index = full_table.schema.get_field_index("vector")
    fixed_type = full_table.schema.field(vector_index).type
    vectors = vector_table["vector"].combine_chunks().cast(fixed_type)
    return full_table.set_column(vector_index, "vector", pa.chunked_array([vectors], type=fixed_type))


def write_index(target_uri: str, table) -> dict[str, Any]:
    import lancedb
    from nemo_retriever.common.vdb.lancedb import LanceDB

    target = Path(target_uri)
    target.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(target_uri)
    written = db.create_table(
        TABLE_NAME,
        data=table,
        schema=table.schema,
        mode="overwrite",
    )
    backend = LanceDB(uri=target_uri, overwrite=False, table_name=TABLE_NAME)
    backend.write_to_index(
        records=None,
        table=written,
        index_type="IVF_HNSW_SQ",
        metric="l2",
        num_partitions=16,
        num_sub_vectors=256,
        hybrid=False,
    )
    return {
        "rows": int(written.count_rows()),
        "schema": str(written.schema),
        "indexes": [index.name for index in written.list_indices()],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--target-uri", required=True)
    parser.add_argument("--mode", choices=("narrow", "custom"), required=True)
    args = parser.parse_args()

    import cudf
    import cupy as cp

    register_cudf_cuda_ipc()
    ray.init()

    @ray.remote(num_gpus=0.5, enable_tensor_transport=True)
    class Source:
        def __init__(self, source_uri: str) -> None:
            self.full_table, vector_table = load_tables(source_uri)
            self.vectors = cudf.DataFrame.from_arrow(vector_table)
            cp.cuda.get_current_stream().synchronize()

        def describe(self) -> dict[str, Any]:
            return {
                "rows": len(self.vectors),
                "columns": list(self.vectors.columns),
                "device": int(cp.cuda.runtime.getDevice()),
            }

        def write_narrow(self, target_uri: str) -> dict[str, Any]:
            started = time.perf_counter()
            reconstructed = replace_vectors(self.full_table, self.vectors)
            cp.cuda.get_current_stream().synchronize()
            gpu_roundtrip_seconds = time.perf_counter() - started
            result = write_index(target_uri, reconstructed)
            result.update(
                {
                    "mode": "narrow",
                    "gpu_roundtrip_seconds": gpu_roundtrip_seconds,
                    "total_seconds": time.perf_counter() - started,
                }
            )
            return result

        @ray.method(tensor_transport="CUDF_CUDA_IPC")
        def get_vectors_custom(self):
            return self.vectors

    @ray.remote(num_gpus=0.5, enable_tensor_transport=True)
    class Sink:
        def __init__(self, source_uri: str) -> None:
            self.full_table, _ = load_tables(source_uri)

        def ready(self) -> int:
            return self.full_table.num_rows

        def receive_only(self, vectors) -> int:
            cp.cuda.get_current_stream().synchronize()
            return len(vectors)

        def write_custom(self, vectors, target_uri: str, started: float):
            received_at = time.perf_counter()
            reconstructed = replace_vectors(self.full_table, vectors)
            cp.cuda.get_current_stream().synchronize()
            result = write_index(target_uri, reconstructed)
            result.update(
                {
                    "mode": "custom",
                    "transport_seconds": received_at - started,
                    "total_seconds": time.perf_counter() - started,
                    "received_type": type(vectors).__name__,
                }
            )
            return result

    source = Source.remote(args.source_uri)
    source_description = ray.get(source.describe.remote())
    if args.mode == "narrow":
        result = ray.get(source.write_narrow.remote(args.target_uri))
    else:
        sink = Sink.remote(args.source_uri)
        ray.get(sink.ready.remote())
        ray.get(sink.receive_only.remote(source.get_vectors_custom.remote()))
        started = time.perf_counter()
        vectors = source.get_vectors_custom.remote()
        result = ray.get(sink.write_custom.remote(vectors, args.target_uri, started))
    result["source"] = source_description
    print(json.dumps(result, indent=2, sort_keys=True))
    ray.shutdown()


if __name__ == "__main__":
    main()
