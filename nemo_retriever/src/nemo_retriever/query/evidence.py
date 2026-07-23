# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Answer-ready ``{evidence, coverage}`` shaping shared by the CLI and the service.

The skill reasons over this shape: each evidence item is fidelity-tagged and
citation-ready, and ``coverage`` summarizes what was searched and flags thin spots.
Both ``retriever query --format evidence`` (CLI) and ``POST /v1/query`` with
``format=evidence`` (service) project raw retrieval hits through this module, so
the two surfaces stay in lockstep.
"""

from __future__ import annotations

import os
from typing import Any

from nemo_retriever.common.vdb.records import _derive_fidelity
from nemo_retriever.common.vdb.sidecar_metadata import parse_hit_content_metadata
from nemo_retriever.query.shaping import resolve_hit_content_type

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
    meta = parse_hit_content_metadata(hit)
    src_raw = hit.get("pdf_basename") or hit.get("source") or ""
    source = os.path.basename(str(src_raw))
    if source.lower().endswith(".pdf"):
        source = source[:-4]
    raw_modality = resolve_hit_content_type(hit) or "text"
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
    elif bbox := meta.get("bbox_xyxy_norm") or hit.get("bbox_xyxy_norm"):
        locator = {"kind": "bbox", "value": bbox}
        citation = source
    else:
        locator = {"kind": "page", "value": None}
        citation = source

    fidelity = meta.get("fidelity") or _derive_fidelity(raw_modality, meta, meta) or "verbatim"

    if "score" in hit and hit["score"] is not None:
        score: float = hit["score"]
    elif "_score" in hit and hit["_score"] is not None:
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


def _is_answer_ready_evidence(item: dict[str, Any]) -> bool:
    text = item.get("text")
    return isinstance(text, str) and bool(text.strip())


def build_evidence_result(hits: list, strategies_used: list[str]) -> dict[str, Any]:
    """Assemble the answer-ready ``{evidence, coverage}`` contract shape from raw hits.

    ``evidence`` items are fidelity-tagged and citation-ready; ``coverage`` summarizes
    what was searched (``strategies_used``, ``n_docs_seen``) and flags thin spots
    (single source, low-fidelity-only, out-of-corpus). This is the shape the skill
    reasons over — emitted by ``retriever query --format evidence`` and by the
    service ``/v1/query`` endpoint with ``format=evidence``.
    """
    projected = [_evidence_item(h) for h in (hits or [])]
    omitted: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for item in projected:
        if _is_answer_ready_evidence(item):
            evidence.append(item)
        else:
            omitted.append(item)
    sources = {item["source"] for item in projected if item.get("source")}
    evidence_sources = {item["source"] for item in evidence if item.get("source")}
    only_visual_omissions = bool(omitted) and all(item.get("modality") == "image" for item in omitted)
    thin: list[str] = []
    if not evidence:
        if only_visual_omissions:
            thin.append("only visual matches — no answer-ready text")
        elif omitted:
            thin.append("matches found — no answer-ready text")
        else:
            thin.append("no matches — likely out of corpus")
    else:
        if len(evidence_sources) == 1:
            thin.append("single source")
        if all(e["fidelity"] == "vlm_caption" for e in evidence):
            thin.append("only low-fidelity (chart/image) evidence")
        if omitted:
            thin.append(
                "visual-only matches omitted" if only_visual_omissions else "matches without answer-ready text omitted"
            )
    return {
        "evidence": evidence,
        "coverage": {
            "strategies_used": strategies_used,
            "n_docs_seen": len(sources),
            "thin_spots": thin,
        },
    }
