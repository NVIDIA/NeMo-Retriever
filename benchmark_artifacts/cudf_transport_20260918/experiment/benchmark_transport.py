# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare a narrow GPU island, Ray's default path, and custom cuDF RDT."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any

import ray

from cudf_cuda_ipc import register_cudf_cuda_ipc


def summarize(samples: list[float], logical_bytes: int) -> dict[str, float]:
    ordered = sorted(samples)
    p50 = statistics.median(ordered)
    p95_index = min(len(ordered) - 1, max(0, int(len(ordered) * 0.95) - 1))
    return {
        "latency_p50_ms": p50 * 1000,
        "latency_p95_ms": ordered[p95_index] * 1000,
        "logical_gbps_p50": logical_bytes / p50 / 1e9,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--dimensions", type=int, default=2_048)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    import cudf
    import cupy as cp
    import numpy as np
    import pyarrow as pa

    register_cudf_cuda_ipc()
    ray.init()

    @ray.remote(num_gpus=0.5, enable_tensor_transport=True)
    class Source:
        def __init__(self) -> None:
            self.frame = None

        def prepare(self, rows: int, dimensions: int) -> dict[str, Any]:
            # Preparation is intentionally excluded from transport timings.
            values = (np.arange(rows * dimensions, dtype=np.float32) % 251) / 251
            # cuDF 26.08 imports Arrow variable-size lists but not Arrow's
            # fixed-size-list logical type. The offsets still describe the
            # exact same dense rows x dimensions float payload.
            embedding = pa.ListArray.from_arrays(
                pa.array(np.arange(0, (rows + 1) * dimensions, dimensions)),
                pa.array(values),
            )
            table = pa.table(
                {
                    "row_id": pa.array(np.arange(rows, dtype=np.int64)),
                    "embedding": embedding,
                }
            )
            self.frame = cudf.DataFrame.from_arrow(table)
            cp.cuda.get_current_stream().synchronize()
            return {
                "rows": len(self.frame),
                "logical_bytes": int(rows * (dimensions * 4 + 8)),
                "checksum": float(self.frame["embedding"].list.leaves.sum()),
            }

        def local_checksum(self) -> tuple[int, float]:
            value = float(self.frame["embedding"].list.leaves.sum())
            cp.cuda.get_current_stream().synchronize()
            return len(self.frame), value

        def get_default(self):
            return self.frame

        @ray.method(tensor_transport="CUDF_CUDA_IPC")
        def get_custom(self):
            return self.frame

    @ray.remote(num_gpus=0.5, enable_tensor_transport=True)
    class Sink:
        def checksum(self, dataframe) -> tuple[int, float, str]:
            value = float(dataframe["embedding"].list.leaves.sum())
            cp.cuda.get_current_stream().synchronize()
            return len(dataframe), value, type(dataframe).__name__

    source = Source.remote()
    sink = Sink.remote()
    prepared = ray.get(source.prepare.remote(args.rows, args.dimensions))

    results: dict[str, Any] = {
        "rows": args.rows,
        "dimensions": args.dimensions,
        "iterations": args.iterations,
        **prepared,
        "variants": {},
    }

    def validate(value: tuple[int, float, str] | tuple[int, float]) -> None:
        if value[0] != args.rows:
            raise AssertionError(f"row mismatch: {value[0]} != {args.rows}")
        if not np.isclose(value[1], prepared["checksum"], rtol=1e-6):
            raise AssertionError(f"checksum mismatch: {value[1]} != {prepared['checksum']}")

    # Warm each path once, then retain end-to-end actor call + GPU checksum time.
    validate(ray.get(source.local_checksum.remote()))
    default_warm = ray.get(sink.checksum.remote(source.get_default.remote()))
    validate(default_warm)
    custom_warm = ray.get(sink.checksum.remote(source.get_custom.remote()))
    validate(custom_warm)

    for name in ("narrow_gpu_island", "ray_default", "cudf_cuda_ipc"):
        samples: list[float] = []
        types: set[str] = set()
        for _ in range(args.iterations):
            started = time.perf_counter()
            if name == "narrow_gpu_island":
                observed = ray.get(source.local_checksum.remote())
            elif name == "ray_default":
                observed = ray.get(sink.checksum.remote(source.get_default.remote()))
            else:
                observed = ray.get(sink.checksum.remote(source.get_custom.remote()))
            samples.append(time.perf_counter() - started)
            validate(observed)
            if len(observed) == 3:
                types.add(observed[2])
        results["variants"][name] = {
            **summarize(samples, prepared["logical_bytes"]),
            "received_types": sorted(types),
            "lossless": True,
            "samples_seconds": samples,
        }

    default_p50 = results["variants"]["ray_default"]["latency_p50_ms"]
    for name, variant in results["variants"].items():
        variant["speedup_vs_ray_default"] = default_p50 / variant["latency_p50_ms"]

    print(json.dumps(results, indent=2, sort_keys=True))
    ray.shutdown()


if __name__ == "__main__":
    main()
