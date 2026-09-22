# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run NRL's unchanged BEIR query/evaluation phase against a supplied index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nemo_retriever.harness.artifact_writer import ArtifactWriter
from nemo_retriever.harness.beir_runner import run_beir_queries
from nemo_retriever.harness.metrics import build_summary_metrics
from nemo_retriever.harness.resolution import build_query_request
from nemo_retriever.query.workflow import resolve_query_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved", required=True)
    parser.add_argument("--lancedb-uri", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    resolved = json.loads(Path(args.resolved).read_text(encoding="utf-8"))
    resolved["ingest"]["storage"]["lancedb_uri"] = args.lancedb_uri
    query_request = build_query_request(resolved, "")
    query_plan = resolve_query_plan(query_request)
    writer = ArtifactWriter(
        artifact_dir=Path(args.output_dir),
        run_id=args.run_id,
        benchmark=str(resolved["name"]),
    )
    latencies, metrics, query_count = run_beir_queries(writer, resolved, query_plan, query_request)
    summary = build_summary_metrics(
        resolved,
        documents=[],
        query_latencies_ms=latencies,
        beir_metrics=metrics,
    )
    summary["query_count"] = query_count
    Path(args.output_dir, "query_only_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
