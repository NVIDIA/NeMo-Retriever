# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction

import pytest
from pydantic import ValidationError

from nemo_retriever.common.api.util.converters.pagetools import (
    normalize_one_based_page_number,
)
from nemo_retriever.common.schemas.collections import QueryHit
from nemo_retriever.common.vdb.records import (
    RetrievalContractError,
    normalize_retrieval_results,
    to_public_collection_hit,
)


def _normalize_one(hit: dict) -> dict:
    return normalize_retrieval_results([[hit]])[0][0]


def test_legacy_entity_is_flattened_once_with_top_level_precedence() -> None:
    hit = _normalize_one(
        {
            "entity": {
                "text": "nested text",
                "source": {"source_id": "nested.pdf"},
                "content_metadata": {"page_number": 2},
                "chunk_id": "nested-chunk",
                "document_id": "nested-document",
            },
            "text": "flat text",
            "source": {"source_id": "flat.pdf"},
            "content_metadata": {"page_number": 3},
            "chunk_id": "flat-chunk",
            "document_id": "flat-document",
            "filename": "flat.pdf",
        }
    )

    assert hit["text"] == "flat text"
    assert hit["source_id"] == "flat.pdf"
    assert hit["page_number"] == 3
    assert hit["chunk_id"] == "flat-chunk"
    assert hit["document_id"] == "flat-document"
    assert "entity" not in hit


def test_legacy_entity_accepts_only_pre_collection_fields() -> None:
    hit = _normalize_one(
        {
            "entity": {
                "text": "legacy text",
                "source": {"source_id": "legacy.pdf"},
                "content_metadata": {"page_number": 4},
                "chunk_id": "nested-chunk",
                "document_id": "nested-document",
                "document_version": "nested-version",
            }
        }
    )

    assert hit["text"] == "legacy text"
    assert hit["source_id"] == "legacy.pdf"
    assert hit["page_number"] == 4
    assert "chunk_id" not in hit
    assert "document_id" not in hit
    assert "document_version" not in hit


def test_flat_hit_is_canonicalized_without_entity() -> None:
    hit = _normalize_one(
        {
            "text": "flat",
            "source": {"source_id": "flat.pdf"},
            "content_metadata": {"page_number": "5"},
        }
    )

    assert hit["text"] == "flat"
    assert hit["source_id"] == "flat.pdf"
    assert hit["page_number"] == 5
    assert hit["pdf_page"] == "flat_5"


@pytest.mark.parametrize("value", [1, "2", " 3 ", 4.0])
def test_normalize_one_based_page_number_accepts_positive_integral_values(
    value,
) -> None:
    assert normalize_one_based_page_number(value) == int(value)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        " ",
        True,
        False,
        0,
        -1,
        -7,
        1.5,
        "1.5",
        "bad",
        Decimal("1.5"),
        Fraction(3, 2),
        math.nan,
        math.inf,
    ],
)
def test_normalize_one_based_page_number_rejects_unknown_or_invalid_values(
    value,
) -> None:
    assert normalize_one_based_page_number(value) is None


def test_invalid_page_is_not_repaired_and_modality_metadata_is_preserved() -> None:
    metadata = {
        "page_number": -7,
        "segment_start_seconds": 2.5,
        "frame_timestamp_seconds": 9.0,
    }

    hit = _normalize_one({"text": "media", "content_metadata": metadata})

    assert hit["page_number"] is None
    assert hit["pdf_page"] == ""
    assert hit["metadata"] == metadata


@pytest.mark.parametrize(
    ("mode", "native_field", "value", "expected"),
    [
        (
            "dense",
            "_distance",
            4.2,
            {
                "rank": 2,
                "value": 4.2,
                "kind": "vector_distance",
                "higher_is_better": False,
            },
        ),
        (
            "hybrid",
            "_relevance_score",
            1.7,
            {
                "rank": 2,
                "value": 1.7,
                "kind": "hybrid_relevance",
                "higher_is_better": True,
            },
        ),
    ],
)
def test_public_collection_hit_describes_native_ranking_without_normalizing(
    mode, native_field, value, expected
) -> None:
    hit = _normalize_one({"text": "hit", native_field: value, "_score": 0.3, "_distance": value})

    public = to_public_collection_hit(hit, retrieval_mode=mode, rank=2)

    assert public["ranking"] == expected
    assert "score" not in public
    assert not {"_distance", "_score", "_relevance_score"}.intersection(public)


@pytest.mark.parametrize("mode,field", [("dense", "_distance"), ("hybrid", "_relevance_score")])
@pytest.mark.parametrize("value", [None, True, math.nan, math.inf, -math.inf])
def test_public_collection_ranking_rejects_missing_or_non_finite_values(mode, field, value) -> None:
    raw = {"text": "bad"}
    if value is not None:
        raw[field] = value

    with pytest.raises(RetrievalContractError, match=rf"{mode} hit 0.*{field}"):
        to_public_collection_hit(_normalize_one(raw), retrieval_mode=mode, rank=1)


def test_hybrid_ranking_does_not_substitute_native_fts_score() -> None:
    with pytest.raises(RetrievalContractError, match="hybrid hit 0.*_relevance_score"):
        to_public_collection_hit(
            _normalize_one({"text": "bad", "_score": 0.8}),
            retrieval_mode="hybrid",
            rank=1,
        )


def test_query_hit_validates_canonical_page_instead_of_repairing_it() -> None:
    payload = {
        "chunk_id": "chunk",
        "document_id": "document",
        "text": "text",
        "ranking": {
            "rank": 1,
            "value": 0.2,
            "kind": "vector_distance",
            "higher_is_better": False,
        },
        "filename": "document.pdf",
    }

    assert QueryHit(**payload, page_number=None).page_number is None
    with pytest.raises(ValidationError):
        QueryHit(**payload, page_number=0)
