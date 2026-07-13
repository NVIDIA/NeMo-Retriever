# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from nemo_retriever.query.evidence import build_evidence_result


def test_visual_only_match_is_omitted_with_specific_coverage() -> None:
    result = build_evidence_result(
        [
            {
                "text": "",
                "source": "scanned.pdf",
                "page_number": 7,
                "content_type": "text",
                "metadata": json.dumps({"type": "image"}),
            }
        ],
        ["dense"],
    )

    assert result == {
        "evidence": [],
        "coverage": {
            "strategies_used": ["dense"],
            "n_docs_seen": 1,
            "thin_spots": ["only visual matches — no answer-ready text"],
        },
    }


def test_nonvisual_match_without_text_is_omitted_with_generic_coverage() -> None:
    result = build_evidence_result(
        [
            {
                "text": " \n\t ",
                "source": "report.pdf",
                "page_number": 3,
                "metadata": {"type": "table"},
            }
        ],
        ["dense"],
    )

    assert result == {
        "evidence": [],
        "coverage": {
            "strategies_used": ["dense"],
            "n_docs_seen": 1,
            "thin_spots": ["matches found — no answer-ready text"],
        },
    }


def test_visual_only_match_is_reported_when_text_evidence_remains() -> None:
    result = build_evidence_result(
        [
            {
                "text": "",
                "source": "scanned.pdf",
                "metadata": {"type": "image"},
            },
            {
                "text": "answer-ready text",
                "source": "text.pdf",
                "metadata": {"type": "text"},
            },
        ],
        ["dense"],
    )

    assert [item["text"] for item in result["evidence"]] == ["answer-ready text"]
    assert result["coverage"] == {
        "strategies_used": ["dense"],
        "n_docs_seen": 2,
        "thin_spots": ["single source", "visual-only matches omitted"],
    }
