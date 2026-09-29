# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from nemo_retriever.service.trajectory_adapter import project_agent_trajectory


def _trajectory() -> dict:
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": "session-1",
        "agent": {"name": "example"},
        "steps": [
            {
                "step_id": "user-1",
                "timestamp": "2026-01-01T00:00:00Z",
                "source": "user",
                "message": "Question",
            },
            {
                "step_id": "agent-1",
                "timestamp": "2026-01-01T00:00:01Z",
                "source": "agent",
                "message": "Answer",
                "reasoning_content": "Reason",
                "tool_calls": [
                    {
                        "tool_call_id": "call-1",
                        "function_name": "set_document_permission",
                        "arguments": {
                            "document": "report-1",
                            "person": "Priya",
                            "role": "writer",
                        },
                    },
                    {
                        "tool_call_id": "call-memory",
                        "function_name": "memory_query",
                        "arguments": {},
                    },
                ],
                "observation": {
                    "results": [
                        {"source_call_id": "call-1", "content": '{"success":true}'},
                        {
                            "source_call_id": "call-memory",
                            "content": "memory result",
                        },
                    ]
                },
            },
        ],
    }


def test_projects_events_to_text_documents_with_metadata() -> None:
    documents = project_agent_trajectory(_trajectory())

    assert [document.text for document in documents] == [
        "Question",
        "Answer",
        "Reason",
        (
            'set_document_permission({"document":"report-1","person":"Priya",'
            '"role":"writer"}) -> {"success":true}'
        ),
        "memory_query({}) -> memory result",
    ]
    assert [document.filename for document in documents] == [
        f"trajectory-{index:06d}.txt" for index in range(5)
    ]
    assert documents[3].metadata["session_id"] == "session-1"
    assert documents[3].metadata["timestamp"] == "2026-01-01T00:00:01Z"
    assert documents[3].metadata["tool_name"] == "set_document_permission"


def test_accepts_text_part_arrays_and_ignores_nontext_parts() -> None:
    trajectory = _trajectory()
    trajectory["steps"][0]["message"] = [
        {"type": "text", "text": "first"},
        {
            "type": "image",
            "source": {
                "media_type": "image/png",
                "path": "https://example.invalid/image.png",
            },
        },
        {"type": "text", "text": "second"},
    ]
    trajectory["steps"][1]["observation"]["results"][0]["content"] = [
        {
            "type": "image",
            "source": {
                "media_type": "image/png",
                "path": "https://example.invalid/result.png",
            },
        },
        {"type": "text", "text": '{"success":true}'},
    ]

    documents = project_agent_trajectory(trajectory)

    assert documents[0].text == "first\nsecond"
    assert documents[3].text.endswith('-> {"success":true}')


def test_omits_events_with_only_nontext_parts() -> None:
    trajectory = _trajectory()
    trajectory["steps"][0]["message"] = [
        {
            "type": "image",
            "source": {
                "media_type": "image/png",
                "path": "https://example.invalid/image.png",
            },
        }
    ]

    documents = project_agent_trajectory(trajectory)

    assert "Question" not in [document.text for document in documents]


def test_excludes_tools_and_truncates_outputs() -> None:
    documents = project_agent_trajectory(
        _trajectory(),
        exclude_tool_names=("memory_query",),
        tool_output_char_limit=6,
    )

    assert documents[-1].text.endswith("-> {\"succ")
    assert documents[-1].metadata["truncated"] is True
    assert documents[-1].metadata["original_chars"] == len('{"success":true}')


@pytest.mark.parametrize(
    ("trajectory", "error", "match"),
    [
        ({"steps": []}, ValueError, "session_id"),
        ({"session_id": "s"}, TypeError, "steps"),
        (
            {
                "session_id": "s",
                "steps": [
                    {
                        "step_id": 1,
                        "source": "user",
                        "message": [{"type": "text", "text": 1}],
                    }
                ],
            },
            TypeError,
            r"message\[0\]\.text",
        ),
    ],
)
def test_rejects_malformed_input(
    trajectory: dict,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        project_agent_trajectory(trajectory)
