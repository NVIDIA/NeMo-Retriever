# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, order-independent row IDs for graph-ingested records."""

from __future__ import annotations

import copy
import random

import pandas as pd
import pytest

from nemo_retriever.common.modality.content_transforms import explode_content_to_rows
from nemo_retriever.common.modality.txt.split import split_df
from nemo_retriever.common.schemas.embedding import embedding_split_metadata
from nemo_retriever.common.vdb.lancedb import LanceDB, _create_lancedb_result, _create_sparse_lancedb_result
from nemo_retriever.common.vdb.records import (
    _client_record_from_graph_row,
    _iter_client_vdb_records,
    graph_row_id,
    graph_row_identity,
)

_EMBEDDING = {"embedding": [0.1, 0.2]}


def _embedded(row: dict) -> dict:
    return {**row, "text_embeddings_1b_v2": _EMBEDDING}


def _pdf_page() -> dict:
    """One PDF page before per-element reshaping, with every element family."""
    return {
        "path": "/docs/report.pdf",
        "page_number": 3,
        "text": "Quarterly revenue grew.",
        "metadata": {"source_path": "/docs/report.pdf", "needs_ocr_for_text": False},
        "table": [
            {"text": "a | b", "bbox_xyxy_norm": [0.1, 0.1, 0.5, 0.5], "caption": "Revenue table"},
            # Identical text in a second detection is still a distinct element.
            {"text": "a | b", "bbox_xyxy_norm": [0.1, 0.6, 0.5, 0.9]},
        ],
        "chart": [{"text": "bars", "bbox_xyxy_norm": [0.5, 0.1, 0.9, 0.5]}],
        "infographic": [{"text": "ocr words", "caption": "ocr words", "bbox_xyxy_norm": [0.2, 0.2, 0.6, 0.6]}],
    }


def _audio_row(**overrides: object) -> dict:
    row = {
        "path": "/tmp/retriever_audio_chunk_x/call_chunk_0001.mp3",
        "source_path": "/media/call.wav",
        "page_number": 1,
        "text": "hello from audio",
        "_content_type": "audio",
        "metadata": {
            "source_path": "/media/call.wav",
            "_content_type": "audio",
            "chunk_index": 1,
            "segment_start_seconds": 30.0,
            "segment_end_seconds": 60.0,
        },
    }
    row.update(overrides)
    return _embedded(row)


def test_row_id_derivation_is_pinned() -> None:
    """Stored IDs are persistent; changing their derivation must be deliberate."""
    row = _embedded(
        {
            "path": "/docs/report.pdf",
            "page_number": 3,
            "text": "Quarterly revenue grew.",
            "_content_type": "table",
            "_bbox_xyxy_norm": [0.1, 0.2, 0.8, 0.9],
            "metadata": {"source_path": "/docs/report.pdf", "chunk_index": 1, "chunk_count": 2},
        }
    )

    assert graph_row_identity(row) == {
        "source": "/docs/report.pdf",
        "page": 3,
        "kind": "table",
        "bbox": [0.1, 0.2, 0.8, 0.9],
        "chunk": {"chunk_index": 1, "chunk_count": 2},
        "text_sha256": "cc6ebcd23f68777636e354d32a44b9ca124a07a84813c97089644d5f1fbafd95",
    }
    assert graph_row_id(row) == "44fee34faab07301e0f8f8b049b9f86bae270c904787e415bda834bf56b462f8"


def test_explicit_ids_win_and_records_otherwise_carry_the_derived_id() -> None:
    derived = _embedded({"text": "chunk", "path": "/docs/a.pdf", "page_number": 1})
    explicit = _embedded({**derived, "metadata": {"content_metadata": {"id": "row-1"}}})
    blank = _embedded({**derived, "metadata": {"id": "", "content_metadata": {"id": " "}}})

    derived_record = _client_record_from_graph_row(derived)
    explicit_record = _client_record_from_graph_row(explicit)
    blank_record = _client_record_from_graph_row(blank)

    assert derived_record["metadata"]["id"] == graph_row_id(derived)
    assert "id" not in explicit_record["metadata"]
    assert blank_record["metadata"]["id"] == graph_row_id(blank)
    # The ID is not copied into the stored metadata JSON column.
    assert "id" not in derived_record["metadata"]["content_metadata"]

    assert _create_lancedb_result(derived_record, expected_dim=2)[0]["id"] == graph_row_id(derived)
    assert _create_lancedb_result(explicit_record, expected_dim=2)[0]["id"] == "row-1"
    assert _create_lancedb_result(blank_record, expected_dim=2)[0]["id"] == graph_row_id(blank)
    assert _create_sparse_lancedb_result(blank_record)["id"] == graph_row_id(blank)


def test_page_elements_get_unique_ids_independent_of_batch_composition() -> None:
    page = _pdf_page()
    other_page = {**copy.deepcopy(page), "page_number": 4}

    alone = explode_content_to_rows(pd.DataFrame([page]), content_columns=("table", "chart", "infographic"))
    batched = explode_content_to_rows(
        pd.DataFrame([other_page, page]), content_columns=("table", "chart", "infographic")
    )

    alone_ids = [graph_row_id(_embedded(row)) for row in alone.to_dict(orient="records")]
    batched_rows = [row for row in batched.to_dict(orient="records") if row["page_number"] == 3]
    batched_ids = [graph_row_id(_embedded(row)) for row in batched_rows]

    # Page text, two table texts, one table caption, one chart, and an
    # infographic text plus a caption that repeats it: seven distinct rows.
    assert len(alone_ids) == 7
    assert len(set(alone_ids)) == 7
    assert sorted(alone_ids) == sorted(batched_ids)


def test_full_page_fallback_elements_differ_only_by_kind() -> None:
    """Nemotron Parse fallback copies identical text and bbox into each structured type."""
    base = {"path": "/docs/a.pdf", "page_number": 2, "text": "same", "_bbox_xyxy_norm": [0.0, 0.0, 1.0, 1.0]}
    ids = {graph_row_id(_embedded({**base, "_content_type": kind})) for kind in ("table", "chart", "infographic")}

    assert len(ids) == 3


def test_text_chunks_and_embedding_splits_get_unique_ids() -> None:
    page = pd.DataFrame([{"text": "one two three four five six", "page_number": 2, "path": "/d/a.txt", "metadata": {}}])
    chunks = split_df(page, max_tokens=2, overlap_tokens=0).to_dict(orient="records")
    chunk_ids = [graph_row_id(_embedded(row)) for row in chunks]

    split_rows = [
        _embedded(
            {
                "text": "whole parent text",
                "path": "/d/a.pdf",
                "page_number": 2,
                "metadata": embedding_split_metadata(
                    content=content,
                    parent_id="parent",
                    chunk_id=f"child-{index}",
                    chunk_index=index,
                    chunk_count=3,
                    start_token=index * 100,
                    end_token=index * 100 + 100,
                ),
            }
        )
        # Two children with identical text remain distinct by token span.
        for index, content in enumerate(["alpha", "alpha", "omega"])
    ]
    split_ids = [graph_row_id(row) for row in split_rows]

    assert len(chunks) > 1
    assert len(set(chunk_ids)) == len(chunks)
    assert len(set(split_ids)) == 3


def test_media_rows_use_source_path_time_window_and_segment() -> None:
    first_run = _audio_row()
    rerun = _audio_row(path="/tmp/retriever_audio_chunk_y/call_chunk_0001.mp3")
    # Invalid ASR ranges are clamped to the whole chunk; segment_index keeps them apart.
    segments = [
        _audio_row(text="same words", metadata={**first_run["metadata"], "segment_index": index}) for index in range(3)
    ]
    frames = [
        _embedded(
            {
                "path": "/media/talk.mp4",
                "page_number": index,
                "text": "slide title",
                "_content_type": "video_frame",
                "metadata": {
                    "source_path": "/media/talk.mp4",
                    "_content_type": "video_frame",
                    "frame_timestamp_seconds": index + 0.5,
                    "segment_start_seconds": float(index),
                    "segment_end_seconds": index + 1.0,
                },
            }
        )
        for index in range(3)
    ]

    assert graph_row_id(first_run) == graph_row_id(rerun)
    assert len({graph_row_id(row) for row in segments}) == 3
    assert len({graph_row_id(row) for row in frames}) == 3


def test_relabelled_media_rows_keep_their_id() -> None:
    """Reshaping a mixed batch can relabel an audio row as text; its ID must not change."""
    audio = _audio_row()
    pdf_page = _pdf_page()
    reshaped = explode_content_to_rows(
        pd.DataFrame([pdf_page, {k: v for k, v in audio.items() if k != "text_embeddings_1b_v2"}]),
        content_columns=("table", "chart", "infographic"),
    ).to_dict(orient="records")
    relabelled = next(row for row in reshaped if row.get("source_path") == "/media/call.wav")

    assert relabelled["_content_type"] == "text"
    assert graph_row_id(_embedded(relabelled)) == graph_row_id(audio)


def test_ray_block_round_trip_keeps_ids() -> None:
    """Rows sharing a Ray block gain null metadata keys and float page numbers."""
    pytest.importorskip("ray")
    from ray.data.block import BlockAccessor

    from nemo_retriever.graph.executor import arrow_table_to_pandas

    rows = [
        _audio_row(),
        _embedded({"path": "/docs/a.pdf", "page_number": None, "text": "no page", "metadata": {"has_text": True}}),
        _embedded(
            {
                "path": "/docs/a.pdf",
                "page_number": 1,
                "text": "table text",
                "_content_type": "table",
                "_bbox_xyxy_norm": [0.25, 0.25, 0.75, 0.75],
                "metadata": {"source_path": "/docs/a.pdf", "chunk_index": 2, "chunk_count": 5},
            }
        ),
    ]
    round_tripped = arrow_table_to_pandas(BlockAccessor.batch_to_block(pd.DataFrame(rows))).to_dict(orient="records")

    assert round_tripped[0]["metadata"] != rows[0]["metadata"]
    assert [graph_row_id(row) for row in round_tripped] == [graph_row_id(row) for row in rows]


def test_ids_are_content_and_provenance_sensitive_but_order_independent() -> None:
    base = _embedded({"path": "/docs/a.pdf", "page_number": 1, "text": "body", "_content_type": "text"})
    variants = [
        base,
        {**base, "text": "other body"},
        {**base, "page_number": 2},
        {**base, "path": "/docs/b.pdf"},
        {**base, "_content_type": "table", "_bbox_xyxy_norm": [0.1, 0.1, 0.2, 0.2]},
        {**base, "_content_type": "table", "_bbox_xyxy_norm": [0.1, 0.1, 0.2, 0.3]},
    ]
    shuffled = variants[:]
    random.Random(7).shuffle(shuffled)

    ids = [graph_row_id(row) for row in variants]
    assert len(set(ids)) == len(variants)
    assert sorted(graph_row_id(row) for row in shuffled) == sorted(ids)
    # The same row produced twice is the same row; staging rejects it as a duplicate.
    assert graph_row_id(copy.deepcopy(base)) == ids[0]


def test_stream_ingest_stores_derived_ids(tmp_path) -> None:
    rows = [
        _embedded({"path": "/docs/a.pdf", "page_number": page, "text": f"page {page}", "metadata": {}})
        for page in range(1, 4)
    ]
    backend = LanceDB(uri=str(tmp_path), table_name="chunks", vector_dim=2, build_index=False)

    backend.stream_ingest(_iter_client_vdb_records(rows))

    import lancedb

    stored = lancedb.connect(str(tmp_path)).open_table("chunks").to_arrow()
    assert sorted(stored.column("id").to_pylist()) == sorted(graph_row_id(row) for row in rows)
