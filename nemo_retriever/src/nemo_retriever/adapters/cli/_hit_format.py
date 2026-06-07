# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-sourced query-hit shaping, shared by the CLI and the MCP surface."""

from __future__ import annotations

from nemo_retriever.vdb.records import RetrievalHit


def _query_cli_hit(hit: RetrievalHit, max_text_chars: int | None = None) -> dict[str, object]:
    metadata = hit.get("metadata") or {}
    modality = hit.get("content_type") or metadata.get("type") or "text"
    # Relevance the engine ranked by: rerank/hybrid score if present, else the
    # vector distance, else null. Hit ORDER is authoritative; score is informational.
    if "_score" in hit:
        score: object = hit["_score"]
    elif "_distance" in hit:
        score = hit["_distance"]
    else:
        score = None
    text = hit.get("text", "")
    # Compact output: truncate text to max_text_chars (0 = omit -> metadata-only
    # summary). None/negative = full text (default, backward-compatible).
    if max_text_chars is not None and max_text_chars >= 0 and len(text) > max_text_chars:
        text = text[:max_text_chars] + ("…" if max_text_chars > 0 else "")
    return {
        "source": hit.get("source", ""),
        "page_number": hit.get("page_number"),
        "text": text,
        "modality": modality,
        "score": score,
    }
