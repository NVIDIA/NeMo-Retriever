# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from nemo_retriever.harness.artifact_writer import ArtifactWriter
from nemo_retriever.harness.media_recall_runner import (
    load_media_recall_dataset,
    run_media_recall_queries,
)


class _Retriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def query(self, query: str, **kwargs: object) -> list[dict[str, object]]:
        self.calls.append((query, kwargs))
        return [
            {
                "source_id": "/registered-datasets/audio_retrieval/corpus/clip.with.dots.mp3",
                "metadata": {"segment_start_seconds": 10.5, "segment_end_seconds": 20.5},
            }
        ]


class _QueryPlan:
    def __init__(self, retriever: _Retriever) -> None:
        self.retriever = retriever

    def create_retriever(self) -> _Retriever:
        return self.retriever

    def query_kwargs(self) -> dict[str, object]:
        return {"top_k": 10}


def test_media_recall_loads_audio_ground_truth_and_records_segment_metrics(tmp_path: Path) -> None:
    queries = tmp_path / "audio.csv"
    queries.write_text(
        "query,expected_media_id,expected_start_time,expected_end_time\n"
        "which clip,clip.with.dots,10.0,20.0\n",
        encoding="utf-8",
    )
    dataset = load_media_recall_dataset(queries)
    writer = ArtifactWriter(artifact_dir=tmp_path / "artifacts", run_id="run-1", benchmark="audio_retrieval_recall")
    retriever = _Retriever()

    latencies, metrics, query_count = run_media_recall_queries(
        writer,
        {
            "dataset": {"query_file": str(queries)},
            "evaluation": {"mode": "media_recall", "dataset_name": str(queries), "ks": (1, 5)},
        },
        _QueryPlan(retriever),  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
    )

    assert dataset.golden_answers == ["clip\t10.000000\t20.000000"]
    assert query_count == 1
    assert len(latencies) == 1
    assert metrics == {"recall@1": 1.0, "recall@5": 1.0}
    assert retriever.calls == [("which clip", {"top_k": 10})]
    assert json.loads(writer.path("media_recall_metrics.json").read_text(encoding="utf-8")) == metrics
    result = json.loads(writer.path("query_results.jsonl").read_text(encoding="utf-8"))
    assert result["hits"][0]["rank"] == 1


def test_media_recall_loads_video_ground_truth_shape(tmp_path: Path) -> None:
    queries = tmp_path / "video.csv"
    queries.write_text(
        "question,name,start_time,end_time\n"
        "which clip,clip.with.dots.mp4,10.0,20.0\n",
        encoding="utf-8",
    )

    dataset = load_media_recall_dataset(queries)

    assert dataset.queries == ["which clip"]
    assert dataset.golden_answers == ["clip\t10.000000\t20.000000"]
