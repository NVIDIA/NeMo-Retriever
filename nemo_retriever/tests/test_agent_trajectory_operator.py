# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pandas as pd
import pytest

from nemo_retriever.common.policy import (
    PipelineOverridesPolicy,
    validate_pipeline_spec,
)
from nemo_retriever.common.schemas.pipeline_spec import PipelineSpec
from nemo_retriever.graph import AgentTrajectoryProjectionOperator, Graph
from nemo_retriever.graph.graph_pipeline_registry import (
    deserialize_graph,
    serialize_graph,
)
from nemo_retriever.graph.ingestor_runtime import build_graph
from nemo_retriever.operators.cpu_operator import CPUOperator
from nemo_retriever.operators.extract.txt.ray_data import TextChunkActor
from nemo_retriever.operators.graph_ops.agent_trajectory_operator import (
    project_agent_trajectory,
)


def _trajectory() -> dict:
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": "session/with spaces",
        "agent": {"name": "example"},
        "steps": [
            {
                "step_id": "user/1",
                "timestamp": "2026-01-01T00:00:00Z",
                "source": "user",
                "message": "Question",
            },
            {
                "step_id": 2,
                "timestamp": "2026-01-01T00:00:01Z",
                "source": "agent",
                "message": "Answer",
                "reasoning_content": "Reason",
                "tool_calls": [
                    {
                        "tool_call_id": "call/search",
                        "function_name": "search",
                        "arguments": {"query": "q"},
                    },
                    {
                        "tool_call_id": "call/memory",
                        "function_name": "memory_query",
                        "arguments": {},
                    },
                ],
                "observation": {
                    "results": [
                        {"source_call_id": "call/search", "content": "search result"},
                        {"source_call_id": "call/memory", "content": "memory result"},
                    ]
                },
                "metrics": {"completion_tokens": 3},
                "extra": {"stage": "main"},
            },
        ],
    }


def test_projects_messages_reasoning_and_tool_observations() -> None:
    result = project_agent_trajectory(_trajectory())

    assert result["text"].tolist() == [
        "Question",
        "Answer",
        "Reason",
        "search result",
        "memory result",
    ]
    assert result["bytes"].tolist() == [text.encode() for text in result["text"]]
    assert [metadata["content_metadata"]["event_type"] for metadata in result["metadata"]] == [
        "message",
        "message",
        "reasoning",
        "observation",
        "observation",
    ]
    observation_metadata = result.iloc[3]["metadata"]["content_metadata"]
    assert observation_metadata["tool_name"] == "search"
    assert observation_metadata["tool_call_id"] == "call/search"
    assert result.iloc[1]["metadata"]["agent_trajectory"]["metrics"] == {"completion_tokens": 3}


def test_source_ids_are_stable_unique_and_url_encoded() -> None:
    first = project_agent_trajectory(_trajectory())
    second = project_agent_trajectory(_trajectory())

    assert first["path"].tolist() == second["path"].tolist()
    assert first["path"].is_unique
    assert all(" " not in path for path in first["path"])
    assert "%2F" in first.iloc[0]["path"]
    assert first.iloc[0]["metadata"]["source_path"] == first.iloc[0]["path"]


def test_excludes_named_tools_and_optionally_truncates_outputs() -> None:
    result = project_agent_trajectory(
        _trajectory(),
        exclude_tool_names=("memory_query",),
        tool_output_char_limit=6,
    )

    assert result["text"].tolist() == ["Question", "Answer", "Reason", "search"]
    metadata = result.iloc[-1]["metadata"]["content_metadata"]
    assert metadata["truncated"] is True
    assert metadata["original_chars"] == len("search result")


def test_default_tool_output_limit_does_not_truncate() -> None:
    trajectory = _trajectory()
    trajectory["steps"][1]["observation"]["results"][0]["content"] = "x" * 25_000

    result = project_agent_trajectory(trajectory)

    assert result.iloc[3]["text"] == "x" * 25_000
    assert result.iloc[3]["metadata"]["content_metadata"]["truncated"] is False


@pytest.mark.parametrize(
    ("trajectory", "error", "match"),
    [
        ({"steps": []}, ValueError, "session_id"),
        ({"session_id": 123, "steps": []}, ValueError, "non-empty string"),
        ({"session_id": "s"}, TypeError, "steps"),
        ({"session_id": "s", "steps": [None]}, TypeError, r"steps\[0\]"),
        (
            {
                "session_id": "s",
                "steps": [{"step_id": 1, "source": "user", "message": {"text": "bad"}}],
            },
            TypeError,
            "message must be a string",
        ),
    ],
)
def test_rejects_malformed_input(trajectory: dict, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        project_agent_trajectory(trajectory)


def test_omits_empty_fields_and_returns_stable_empty_schema() -> None:
    result = project_agent_trajectory(
        {
            "session_id": "empty",
            "steps": [
                {"step_id": 1, "source": "user", "message": ""},
                {
                    "step_id": 2,
                    "source": "agent",
                    "message": None,
                    "reasoning_content": "   ",
                    "observation": {"results": [{"source_call_id": "x", "content": ""}]},
                },
            ],
        }
    )

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["path", "bytes", "text", "metadata"]
    assert result.empty


def test_operator_is_cpu_designer_component_and_graph_serializable() -> None:
    operator = AgentTrajectoryProjectionOperator(
        exclude_tool_names=("memory_query",),
        tool_output_char_limit=128,
    )
    graph = Graph() >> operator

    restored = deserialize_graph(serialize_graph(graph)).roots[0].operator

    assert isinstance(operator, CPUOperator)
    assert operator._designer_meta["compute"] == "cpu"
    assert isinstance(restored, AgentTrajectoryProjectionOperator)
    assert restored.exclude_tool_names == ("memory_query",)
    assert restored.tool_output_char_limit == 128


def test_operator_projects_uploaded_json_dataframe() -> None:
    result = AgentTrajectoryProjectionOperator().run(
        pd.DataFrame(
            [
                {
                    "path": "trajectory.json",
                    "bytes": json.dumps(_trajectory()).encode(),
                }
            ]
        )
    )

    assert result["text"].tolist() == [
        "Question",
        "Answer",
        "Reason",
        "search result",
        "memory result",
    ]


def test_trajectory_extraction_mode_uses_metadata_safe_chunking() -> None:
    graph = build_graph(
        extraction_mode="trajectory",
        trajectory_params={
            "exclude_tool_names": ["memory_query"],
            "tool_output_char_limit": 128,
        },
    )

    root = graph.roots[0]
    assert isinstance(root.operator, AgentTrajectoryProjectionOperator)
    assert root.operator.exclude_tool_names == ("memory_query",)
    assert isinstance(root.children[0].operator, TextChunkActor)


def test_service_pipeline_spec_allows_bounded_trajectory_options() -> None:
    spec = PipelineSpec(
        extraction_mode="trajectory",
        trajectory_params={
            "exclude_tool_names": ["nrl-memory_query"],
            "tool_output_char_limit": 4096,
        },
        stage_order=["extract"],
    )

    validated = validate_pipeline_spec(spec, PipelineOverridesPolicy())

    assert validated == spec
