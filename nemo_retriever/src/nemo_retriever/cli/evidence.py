# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Answer-ready ``{evidence, coverage}`` shaping for ``retriever query --format evidence``.

The skill reasons over this shape: each evidence item is fidelity-tagged and
citation-ready, and ``coverage`` summarizes what was searched and flags thin spots.
``--format evidence`` is opt-in; ``query``'s default output stays the flat hit list.
"""

from __future__ import annotations

import os
from typing import Any

from nemo_retriever.common.vdb.records import _derive_fidelity

_KNOWN_MODALITIES = {"text", "table", "chart", "image", "audio", "video_frame"}


def _normalize_modality(value: Any) -> str:
    m = str(value or "text").lower()
    if m in _KNOWN_MODALITIES:
        return m
    if m.startswith("table"):
        return "table"
    if m.startswith("chart"):
        return "chart"
    if m.startswith(("image", "infographic")):
        return "image"
    if m.startswith("video"):
        return "video_frame"
    if m.startswith("audio"):
        return "audio"
    return "text"


def _evidence_item(hit: dict[str, Any]) -> dict[str, Any]:
    meta = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    src_raw = hit.get("pdf_basename") or hit.get("source") or ""
    source = os.path.basename(str(src_raw))
    if source.lower().endswith(".pdf"):
        source = source[:-4]
    raw_modality = hit.get("content_type") or meta.get("type") or "text"
    modality = _normalize_modality(raw_modality)

    page = hit.get("page_number")
    if page is not None:
        locator = {"kind": "page", "value": page}
        citation = f"{source} p.{page}"
    elif meta.get("segment_start_seconds") is not None:
        locator = {"kind": "segment", "value": meta["segment_start_seconds"]}
        citation = f"{source} @{meta['segment_start_seconds']}"
    elif meta.get("frame_timestamp_seconds") is not None:
        locator = {"kind": "timestamp", "value": meta["frame_timestamp_seconds"]}
        citation = f"{source} @{meta['frame_timestamp_seconds']}"
    elif meta.get("bbox_xyxy_norm") is not None:
        locator = {"kind": "bbox", "value": meta["bbox_xyxy_norm"]}
        citation = source
    else:
        locator = {"kind": "page", "value": None}
        citation = source

    fidelity = meta.get("fidelity") or _derive_fidelity(raw_modality, meta, meta) or "verbatim"

    if "_score" in hit and hit["_score"] is not None:
        score: float = hit["_score"]
    elif "_distance" in hit and hit["_distance"] is not None:
        score = hit["_distance"]
    else:
        score = 0.0

    return {
        "text": hit.get("text", ""),
        "source": source,
        "locator": locator,
        "modality": modality,
        "fidelity": fidelity,
        "score": score,
        "citation": citation,
    }


def build_evidence_result(hits: list, strategies_used: list[str]) -> dict[str, Any]:
    """Assemble the answer-ready ``{evidence, coverage}`` contract shape from raw hits.

    ``evidence`` items are fidelity-tagged and citation-ready; ``coverage`` summarizes
    what was searched (``strategies_used``, ``n_docs_seen``) and flags thin spots
    (single source, low-fidelity-only, out-of-corpus). This is the shape the skill
    reasons over — emitted by ``retriever query --format evidence``.
    """
    evidence = [_evidence_item(h) for h in (hits or [])]
    sources = {e["source"] for e in evidence if e.get("source")}
    thin: list[str] = []
    if not evidence:
        thin.append("no matches — likely out of corpus")
    else:
        if len(sources) == 1:
            thin.append("single source")
        if all(e["fidelity"] == "vlm_caption" for e in evidence):
            thin.append("only low-fidelity (chart/image) evidence")
    return {
        "evidence": evidence,
        "coverage": {"strategies_used": strategies_used, "n_docs_seen": len(sources), "thin_spots": thin},
    }
