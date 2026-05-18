from __future__ import annotations

import argparse
import socket
from typing import Any

import pandas as pd
import ray

from _common import assert_cluster_ready, dataframe_sample, ray_state, write_json
from nemo_retriever.graph import Graph, Node, RayDataExecutor
from nemo_retriever.params import TextChunkParams
from nemo_retriever.txt.ray_data import TextChunkCPUActor, TxtSplitCPUActor


class HostAnnotatedTxtSplitCPUActor(TxtSplitCPUActor):
    def process(self, data: Any, **kwargs: Any) -> Any:
        output = super().process(data, **kwargs)
        if isinstance(output, pd.DataFrame):
            output = output.copy()
            output["split_host"] = socket.gethostname()
        return output


class HostAnnotatedTextChunkCPUActor(TextChunkCPUActor):
    def process(self, data: Any, **kwargs: Any) -> Any:
        output = super().process(data, **kwargs)
        if isinstance(output, pd.DataFrame):
            output = output.copy()
            output["chunk_host"] = socket.gethostname()
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a RayDataExecutor NeMo-Retriever text smoke.")
    parser.add_argument("--ray-address", required=True)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _build_graph(params: TextChunkParams) -> Graph:
    graph = Graph()
    graph.add_chain(
        Node(HostAnnotatedTxtSplitCPUActor(params=params), name="TxtSplitActor"),
        Node(HostAnnotatedTextChunkCPUActor(params=params), name="TextChunkActor"),
    )
    return graph


def _node_overrides() -> dict[str, dict[str, Any]]:
    return {
        "TxtSplitActor": {
            "resources": {"nemo_worker": 0.01},
            "num_cpus": 0.25,
            "num_gpus": 0,
            "concurrency": 1,
        },
        "TextChunkActor": {
            "resources": {"nemo_head": 0.01},
            "num_cpus": 0.25,
            "num_gpus": 0,
            "concurrency": 1,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    params = TextChunkParams(
        tokenizer_model_id=args.tokenizer_dir,
        max_tokens=6,
        overlap_tokens=0,
    )
    executor = RayDataExecutor(
        _build_graph(params),
        ray_address=args.ray_address,
        batch_size=1,
        num_cpus=0.25,
        num_gpus=0,
        node_overrides=_node_overrides(),
    )
    result = executor.ingest(args.input_glob)
    state = ray_state(ray)

    if not isinstance(result, pd.DataFrame):
        raise AssertionError(f"expected pandas DataFrame result, got {type(result).__name__}")
    if result.empty:
        raise AssertionError("expected non-empty result DataFrame")
    assert_cluster_ready(state, expected_nodes=2)

    for column in ("split_host", "chunk_host"):
        if column not in result.columns:
            raise AssertionError(f"expected result column {column!r}")

    split_hosts = sorted({str(host) for host in result["split_host"].dropna().unique() if str(host)})
    chunk_hosts = sorted({str(host) for host in result["chunk_host"].dropna().unique() if str(host)})
    if not split_hosts:
        raise AssertionError("expected at least one split_host value")
    if not chunk_hosts:
        raise AssertionError("expected at least one chunk_host value")
    if set(split_hosts) & set(chunk_hosts):
        raise AssertionError(f"expected split_hosts and chunk_hosts to differ, got {split_hosts!r} and {chunk_hosts!r}")

    return {
        "status": "succeeded",
        "ray_address": args.ray_address,
        "rows": int(len(result)),
        "columns": list(result.columns),
        "split_hosts": split_hosts,
        "chunk_hosts": chunk_hosts,
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
