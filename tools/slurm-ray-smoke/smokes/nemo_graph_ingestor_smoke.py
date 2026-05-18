from __future__ import annotations

import argparse
from typing import Any

import pandas as pd
import ray

from _common import assert_cluster_ready, dataframe_sample, ray_state, write_json
from nemo_retriever.graph_ingestor import GraphIngestor
from nemo_retriever.params import TextChunkParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a GraphIngestor NeMo-Retriever text smoke.")
    parser.add_argument("--ray-address", required=True)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    params = TextChunkParams(
        tokenizer_model_id=args.tokenizer_dir,
        max_tokens=6,
        overlap_tokens=0,
    )
    result = (
        GraphIngestor(
            run_mode="batch",
            documents=[args.input_glob],
            ray_address=args.ray_address,
            allow_no_gpu=True,
            batch_size=1,
            num_cpus=0.25,
            num_gpus=0,
            node_overrides={
                "MultiTypeExtractOperator": {
                    "resources": {"nemo_worker": 0.01},
                    "num_cpus": 0.25,
                    "num_gpus": 0,
                    "concurrency": 1,
                }
            },
        )
        .extract_txt(params)
        .ingest()
    )
    state = ray_state(ray)

    if not isinstance(result, pd.DataFrame):
        raise AssertionError(f"expected pandas DataFrame result, got {type(result).__name__}")
    if result.empty:
        raise AssertionError("expected non-empty result DataFrame")
    assert_cluster_ready(state, expected_nodes=2)

    return {
        "status": "succeeded",
        "ray_address": args.ray_address,
        "rows": int(len(result)),
        "columns": list(result.columns),
        "ray": state,
        "sample": dataframe_sample(result),
    }


def main() -> None:
    args = parse_args()
    try:
        payload = run(args)
    except Exception as exc:
        write_json(args.output_json, {"status": "failed", "error": repr(exc)})
        raise
    write_json(args.output_json, payload)


if __name__ == "__main__":
    main()
