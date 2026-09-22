"""Losslessness smoke test for the experimental cuDF CUDA IPC transport."""

from __future__ import annotations

import json
import time

import ray

from cudf_cuda_ipc import register_cudf_cuda_ipc


def main() -> None:
    import cudf
    import cupy as cp

    register_cudf_cuda_ipc()
    ray.init()

    @ray.remote(num_gpus=0.5, enable_tensor_transport=True)
    class Source:
        @ray.method(tensor_transport="CUDF_CUDA_IPC")
        def make(self, rows: int):
            return cudf.DataFrame(
                {
                    "row_id": cp.arange(rows, dtype=cp.int64),
                    "score": cp.arange(rows, dtype=cp.float32) / 7,
                    "label": cudf.Series([f"row-{i % 17}" for i in range(rows)]),
                }
            )

    @ray.remote(num_gpus=0.5, enable_tensor_transport=True)
    class Sink:
        def verify(self, dataframe):
            expected = int(dataframe["row_id"].sum())
            checksum = float(dataframe["score"].sum())
            return {
                "type": type(dataframe).__name__,
                "rows": len(dataframe),
                "row_id_sum": expected,
                "score_sum": checksum,
                "device": int(cp.cuda.runtime.getDevice()),
            }

    source = Source.remote()
    sink = Sink.remote()
    started = time.perf_counter()
    frame_ref = source.make.remote(100_000)
    result = ray.get(sink.verify.remote(frame_ref))
    result["elapsed_seconds"] = time.perf_counter() - started
    result["expected_row_id_sum"] = 100_000 * 99_999 // 2
    result["lossless"] = result["row_id_sum"] == result["expected_row_id_sum"]
    print(json.dumps(result, sort_keys=True))
    ray.shutdown()


if __name__ == "__main__":
    main()
