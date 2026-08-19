# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

lancedb = pytest.importorskip("lancedb")

from nemo_retriever.common.vdb.lancedb import LanceDB, _create_lancedb_results


def _records(text: str = "hello", vector: list[float] | None = None) -> list[list[dict]]:
    return [
        [
            {
                "document_type": "text",
                "metadata": {
                    "embedding": vector or [1.0, 0.0],
                    "content": text,
                    "content_metadata": {"page_number": 1},
                    "source_metadata": {"source_name": "doc.pdf"},
                },
            }
        ]
    ]


def _count_rows(uri: Path, table_name: str = "t") -> int:
    return int(lancedb.connect(str(uri)).open_table(table_name).count_rows())


def _image_only_records(vector: list[float] | None = None) -> list[list[dict]]:
    metadata = {
        "content": "",
        "content_metadata": {
            "page_number": 7,
            "type": "image",
        },
        "source_metadata": {"source_id": "scanned.pdf", "source_name": "scanned.pdf"},
    }
    if vector is not None:
        metadata["embedding"] = vector
    return [[{"document_type": "image", "metadata": metadata}]]


def _write_rows(tmp_path: Path, records: list[list[dict]], *, sparse: bool = False) -> list[dict]:
    op = LanceDB(uri=str(tmp_path), table_name="t", vector_dim=2, sparse=sparse, create_index=False)
    op.run(records)
    return lancedb.connect(str(tmp_path)).open_table("t").to_arrow().to_pylist()


def test_overwrite_same_records_twice_keeps_row_count_stable(tmp_path: Path) -> None:
    op = LanceDB(uri=str(tmp_path), table_name="t", vector_dim=2, create_index=False)

    op.run(_records())
    assert _count_rows(tmp_path) == 1

    op.run(_records())
    assert _count_rows(tmp_path) == 1


def test_append_to_missing_table_creates_it(tmp_path: Path) -> None:
    op = LanceDB(uri=str(tmp_path), table_name="t", vector_dim=2, overwrite=False, create_index=False)

    op.run(_records())

    assert _count_rows(tmp_path) == 1


def test_append_same_records_twice_doubles_row_count(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    op = LanceDB(uri=str(tmp_path), table_name="t", vector_dim=2, overwrite=False, create_index=False)

    op.run(_records())
    with caplog.at_level(logging.WARNING):
        op.run(_records())

    assert _count_rows(tmp_path) == 2
    assert "Append mode does not deduplicate" in caplog.text


def test_append_with_matching_embedding_model_succeeds(tmp_path: Path) -> None:
    model_name = "nvidia/embedding-model-a"
    LanceDB(
        uri=str(tmp_path),
        table_name="t",
        vector_dim=2,
        embedding_model_name=model_name,
        create_index=False,
    ).run(_records())

    LanceDB(
        uri=str(tmp_path),
        table_name="t",
        vector_dim=2,
        embedding_model_name=model_name,
        overwrite=False,
        create_index=False,
    ).run(_records())

    assert _count_rows(tmp_path) == 2


def test_embedding_model_revision_is_recorded_and_readable(tmp_path: Path) -> None:
    op = LanceDB(
        uri=str(tmp_path),
        table_name="t",
        vector_dim=2,
        embedding_model_name="nvidia/embedding-model-a",
        embedding_model_revision="a" * 40,
        create_index=False,
    )

    op.run(_records())

    assert op.get_index_metadata("embedding_model_name") == "nvidia/embedding-model-a"
    assert op.get_index_metadata("embedding_model_revision") == "a" * 40


def test_vector_dimension_can_be_inferred_from_model_output(tmp_path: Path) -> None:
    op = LanceDB(
        uri=str(tmp_path),
        table_name="t",
        vector_dim=None,
        embedding_model_name="nvidia/llama-embed-nemotron-8b",
        create_index=False,
    )

    op.run(_records(vector=[0.0] * 4096))

    table = lancedb.connect(str(tmp_path)).open_table("t")
    schema = table.schema() if callable(table.schema) else table.schema
    assert schema.field("vector").type.list_size == 4096


def test_append_with_inferred_dimension_uses_existing_table_schema(tmp_path: Path) -> None:
    kwargs = {
        "uri": str(tmp_path),
        "table_name": "t",
        "vector_dim": None,
        "embedding_model_name": "nvidia/llama-embed-nemotron-8b",
        "create_index": False,
    }
    LanceDB(**kwargs).run(_records(vector=[0.0] * 4096))

    LanceDB(**kwargs, overwrite=False).run(_records(vector=[0.0] * 4096))

    assert _count_rows(tmp_path) == 2


def test_append_with_mismatched_embedding_model_fails_before_write(tmp_path: Path) -> None:
    LanceDB(
        uri=str(tmp_path),
        table_name="t",
        vector_dim=2,
        embedding_model_name="nvidia/embedding-model-a",
        create_index=False,
    ).run(_records())

    op = LanceDB(
        uri=str(tmp_path),
        table_name="t",
        vector_dim=2,
        embedding_model_name="nvidia/embedding-model-b",
        overwrite=False,
        create_index=False,
    )

    with pytest.raises(ValueError, match="cannot append vectors"):
        op.run(_records())

    assert _count_rows(tmp_path) == 1


@pytest.mark.parametrize(
    ("stored_revision", "incoming_revision", "error_pattern", "expected_rows"),
    [
        pytest.param("a" * 40, "b" * 40, "cannot append vectors from revision", 1, id="mismatch"),
        pytest.param("a" * 40, None, "without a known revision", 1, id="missing"),
        pytest.param("a" * 40, "a" * 40, None, 2, id="matching"),
        pytest.param(None, "a" * 40, None, 2, id="legacy-table"),
    ],
)
def test_append_revision_compatibility(
    tmp_path: Path,
    stored_revision: str | None,
    incoming_revision: str | None,
    error_pattern: str | None,
    expected_rows: int,
) -> None:
    common = {
        "uri": str(tmp_path),
        "table_name": "t",
        "vector_dim": 2,
        "embedding_model_name": "nvidia/embedding-model-a",
        "create_index": False,
    }
    LanceDB(**common, embedding_model_revision=stored_revision).run(_records())
    incoming = LanceDB(
        **common,
        embedding_model_revision=incoming_revision,
        overwrite=False,
    )

    if error_pattern is not None:
        with pytest.raises(ValueError, match=error_pattern):
            incoming.run(_records())
    else:
        incoming.run(_records())

    assert _count_rows(tmp_path) == expected_rows


def test_append_incompatible_schema_raises_clear_error(tmp_path: Path) -> None:
    LanceDB(uri=str(tmp_path), table_name="t", vector_dim=3, create_index=False).run(_records(vector=[1.0, 0.0, 0.0]))

    op = LanceDB(uri=str(tmp_path), table_name="t", vector_dim=2, overwrite=False, create_index=False)

    with pytest.raises(ValueError, match="incompatible field 'vector'"):
        op.run(_records())


def test_create_index_kwarg_disables_index_build_without_shadowing_method(tmp_path: Path) -> None:
    op = LanceDB(uri=str(tmp_path), table_name="t", vector_dim=2, create_index=False)
    assert callable(op.create_index)
    assert op.build_index is False

    def fail_if_called(*_args, **_kwargs) -> None:
        raise AssertionError("write_to_index should not be called when create_index=False")

    op.write_to_index = fail_if_called  # type: ignore[method-assign]
    op.run(_records())

    assert _count_rows(tmp_path) == 1


@pytest.mark.parametrize(
    "caption",
    [pytest.param(None, id="empty"), pytest.param(" \n\t ", id="whitespace")],
)
def test_dense_write_stores_blank_canonical_image_record(tmp_path: Path, caption: str | None) -> None:
    records = _image_only_records([1.0, 0.0])
    if caption is not None:
        records[0][0]["metadata"]["image_metadata"] = {"caption": caption}

    table_rows = _write_rows(tmp_path, records)

    assert len(table_rows) == 1
    assert table_rows[0]["text"] == ""
    assert json.loads(table_rows[0]["metadata"]) == {"type": "image", "page_number": 7}
    assert json.loads(table_rows[0]["source"]) == {
        "source_id": "scanned.pdf",
        "source_name": "scanned.pdf",
    }


@pytest.mark.parametrize(
    "text",
    [pytest.param("", id="empty"), pytest.param(" \n\t ", id="whitespace")],
)
def test_dense_write_drops_blank_non_image_row(tmp_path: Path, text: str) -> None:
    table_rows = _write_rows(tmp_path, _records(text=text))

    assert table_rows == []


@pytest.mark.parametrize("missing_field", ["document_type", "content_metadata.type"])
def test_dense_write_requires_both_canonical_image_fields(tmp_path: Path, missing_field: str) -> None:
    records = _image_only_records([1.0, 0.0])
    if missing_field == "document_type":
        records[0][0]["document_type"] = "text"
    else:
        records[0][0]["metadata"]["content_metadata"].pop("type")

    table_rows = _write_rows(tmp_path, records)

    assert table_rows == []


@pytest.mark.parametrize(
    "vector",
    [
        pytest.param([], id="empty-embed-failure"),
    ],
)
def test_dense_write_fails_on_image_only_row_with_no_usable_embedding(
    tmp_path: Path, vector: list[float] | None
) -> None:
    """A row without a usable embedding must fail the write, not be dropped.

    The embed stage writes ``embedding: []`` for every row of a batch whose
    engine failed. On the unpatched tree that row was accepted here, because
    ``[]`` is not ``None`` and this write infers the dimension, so the length
    check does not run - which is how a run publishes a short index and still
    reports success.

    A ``None`` embedding is deliberately not covered: it has a legitimate
    producer and keeps its pre-existing drop. See
    ``test_lancedb_incomplete_index_guard.py``,
    ``test_a_none_embedding_keeps_its_pre_existing_silent_drop``.
    """
    with pytest.raises(RuntimeError, match="Refusing to build an incomplete index"):
        _write_rows(tmp_path, _image_only_records(vector))


def test_dense_write_still_drops_wrong_length_row_under_the_default_policy(tmp_path: Path) -> None:
    """``on_bad_vectors="drop"`` keeps working: a short vector is dropped, not fatal.

    Known-bad for the first revision of this fix, which folded
    ``dropped_bad_length`` into the fatal condition and made the shipped default
    unreachable. Passes on unmodified HEAD too - it pins the documented contract
    rather than a code change, so it is a guard test.
    """
    table_rows = _write_rows(tmp_path, _image_only_records([1.0]))

    assert table_rows == []


def test_dense_write_keeps_on_bad_vectors_fill_reachable(tmp_path: Path) -> None:
    """``on_bad_vectors="fill"`` with the wrapper check off still reaches LanceDB.

    With ``validate_vector_length=False`` the short row is forwarded and LanceDB
    fills it, which is what a user who configured ``fill`` asked for. The guard
    must not pre-empt that.

    Known-bad for the first revision of this fix, which raised before LanceDB
    ever saw the row. Passes on unmodified HEAD; guard test.

    Asserts the row survives at full schema width, not the exact filled
    composition: how LanceDB distributes ``fill_value`` over a short vector is
    its own detail and differs by version (0.34 replaces the whole vector, 0.37
    pads and keeps the produced component), and ``lancedb`` is unpinned here.
    What this guard owns is that the row reached the writer at all.
    """
    op = LanceDB(
        uri=str(tmp_path),
        table_name="t",
        vector_dim=2,
        create_index=False,
        on_bad_vectors="fill",
        fill_value=0.5,
        validate_vector_length=False,
    )
    op.run(_image_only_records([1.0]))

    table_rows = lancedb.connect(str(tmp_path)).open_table("t").to_arrow().to_pylist()
    assert len(table_rows) == 1
    assert len(table_rows[0]["vector"]) == 2


def test_sparse_write_drops_image_only_row_without_text(tmp_path: Path) -> None:
    table_rows = _write_rows(tmp_path, _image_only_records([1.0, 0.0]), sparse=True)

    assert table_rows == []


def test_sparse_write_drops_whitespace_only_text(tmp_path: Path) -> None:
    table_rows = _write_rows(tmp_path, _records(text=" \n\t "), sparse=True)

    assert table_rows == []

# --- incomplete-index guard: a row must not reach the index without an embedding ---
# ``[]`` is not ``None``, so it used to fall through to the length check, be counted a
# wrong-length vector, and be dropped - a short index published with exit 0.


def _record(embedding: Any, *, text: str = "page text") -> dict:
    return {
        "document_type": "text",
        "metadata": {
            "embedding": embedding,
            "content": text,
            "content_metadata": {"page_number": 1, "id": "row-1"},
            "source_metadata": {"source_name": "doc.pdf"},
        },
    }


def test_an_empty_embedding_fails_the_run() -> None:
    """Known-bad: returned ``(rows, counts)`` and only logged a WARNING.

    ``embedding: []`` is what ``embed_text_main_text_embed`` writes for every
    row of a batch it could not embed. It was not counted before this change:
    it is not ``None``, and it passed the length check as a wrong-length vector.
    """
    records = [[_record([]), _record([1.0, 2.0])]]

    with pytest.raises(RuntimeError) as excinfo:
        _create_lancedb_results(records, expected_dim=2)

    message = str(excinfo.value)
    assert "Refusing to build an incomplete index" in message
    assert "1 of 2 rows" in message
    assert "empty_embedding=1" in message


def test_create_lancedb_results_rejects_empty_embeddings_without_length_check() -> None:
    """The non-enforcing path must not write empty vectors into the index.

    Known-bad: with ``expected_dim=None`` an ``embedding: []`` row was accepted
    and counted in ``accepted``.
    """
    with pytest.raises(RuntimeError, match="empty_embedding=1"):
        _create_lancedb_results([[_record([])]], expected_dim=None)


@pytest.mark.parametrize("on_bad_vectors", ["drop", "fill", "error"])
def test_wrong_length_rows_stay_under_the_on_bad_vectors_policy(on_bad_vectors: str) -> None:
    """A short vector is a schema mismatch, not a missing embedding.

    ``on_bad_vectors`` is a documented, user-configured tolerance
    (``common/vdb/lancedb.py`` ``create_index``). The incomplete-index guard must
    not reach into it, or a user who deliberately configured ``drop`` or ``fill``
    would go from silent dropping to a hard run failure on upgrade.

    Known-bad for the first revision of this fix, which folded
    ``dropped_bad_length`` into the fatal condition and raised for all three
    values. It pins a contract rather than a code change, so it is a guard test:
    the drop it asserts is identical before and after. It cannot run on the
    unpatched tree, because the final assertion reads the new
    ``empty_embedding`` key.
    """
    records = [[_record([1.0]), _record([1.0, 2.0])]]

    # ``expected_dim=None`` is the shape ``create_index`` uses when the caller
    # asked LanceDB to own the policy (``on_bad_vectors="error"``) or turned the
    # wrapper's length check off.
    rows, counts = _create_lancedb_results(records, expected_dim=None if on_bad_vectors == "error" else 2)

    if on_bad_vectors == "error":
        # The wrapper forwards both rows so LanceDB itself raises, per the
        # documented strict-fail semantics of that policy.
        assert len(rows) == 2
        assert counts["dropped_bad_length"] == 0
    else:
        assert len(rows) == 1
        assert counts["dropped_bad_length"] == 1
    assert counts["dropped_no_embedding"] == 0
    assert counts["empty_embedding"] == 0


def test_empty_embeddings_from_the_endpoint_path_do_not_reach_the_index_silently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The writer is the backstop for producers that legitimately keep going.

    A local engine failure now propagates from
    ``models/inference/runtime.py`` and never reaches here. The endpoint path
    keeps its per-batch resilience by design - a single failed HTTP call should
    not kill a service-mode run - so it can still emit ``embedding: []`` rows.
    This is the case that makes the writer-side check load-bearing rather than
    redundant.

    Known-bad: the writer dropped those rows and returned normally.
    """
    from nemo_retriever.models.inference import runtime

    def _refuse(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        raise TimeoutError("read timed out waiting for the embedding endpoint")

    monkeypatch.setattr(runtime, "_embed_group", _refuse)

    batch_df = pd.DataFrame(
        {
            "text": ["page one", "page two"],
            "metadata": [
                {"content": "page one", "content_metadata": {"page_number": 1}, "source_metadata": {}},
                {"content": "page two", "content_metadata": {"page_number": 2}, "source_metadata": {}},
            ],
        }
    )

    out_df = runtime.embed_text_main_text_embed(
        batch_df,
        embedding_endpoint="http://embed.example/v1",
        inference_batch_size=2,
    )

    # The stage returns successfully - deliberate for the endpoint path - but
    # every row is empty.
    assert list(out_df["text_embeddings_1b_v2_has_embedding"]) == [False, False]
    assert [payload["embedding"] for payload in out_df["text_embeddings_1b_v2"]] == [[], []]

    records = [
        [
            {
                "document_type": "text",
                "metadata": {**row["metadata"], "embedding": row["text_embeddings_1b_v2"]["embedding"]},
            }
            for _index, row in out_df.iterrows()
        ]
    ]

    with pytest.raises(RuntimeError, match="Refusing to build an incomplete index"):
        _create_lancedb_results(records, expected_dim=2048)


def test_canonical_image_row_without_text_is_accepted_not_dropped() -> None:
    """The text carve-out for canonical image rows still works.

    An image row legitimately carries ``text=""``. It has a real embedding, so
    nothing here may touch it.
    """
    record = {
        "document_type": "image",
        "metadata": {
            "embedding": [1.0, 2.0],
            "content": "",
            "content_metadata": {"page_number": 3, "type": "image"},
            "source_metadata": {"source_name": "scan.pdf"},
        },
    }
    rows, counts = _create_lancedb_results([[record]], expected_dim=2)

    assert len(rows) == 1
    assert counts["accepted"] == 1
    assert counts["dropped_no_text"] == 0


def test_text_free_non_image_row_is_dropped_and_never_fatal() -> None:
    """``dropped_no_text`` stays out of the fatal condition.

    It is a content filter, not a loss: the row embedded successfully and was
    excluded for having nothing to search on.

    Cannot run on the unpatched tree: it asserts on the new
    ``empty_embedding`` key. The drop itself is unchanged.
    """
    rows, counts = _create_lancedb_results([[_record([1.0, 2.0], text="   ")]], expected_dim=2)

    assert rows == []
    assert counts["dropped_no_text"] == 1
    assert counts["empty_embedding"] == 0


@pytest.mark.parametrize(
    "embedding",
    [
        pytest.param([0.0, 0.0], id="all-zero-but-present"),
        pytest.param((1.0, 2.0), id="tuple"),
    ],
)
def test_present_vectors_are_accepted_whatever_their_values(embedding) -> None:
    """The check tests presence, never values.

    Deciding from the numbers would mean guessing which embeddings are "real",
    and an all-zero vector is a legal thing for a model to emit. Only the
    absence of a vector is fatal.
    """
    rows, counts = _create_lancedb_results([[_record(embedding)]], expected_dim=2)

    assert len(rows) == 1
    assert counts["accepted"] == 1


def test_an_empty_numpy_array_is_not_swallowed_by_the_new_check() -> None:
    """An empty ndarray is not a list, so it keeps its pre-existing route.

    This is the narrowness of the check made explicit: it fires on ``[]`` and
    ``()`` and nothing else.
    """
    numpy = pytest.importorskip("numpy")

    rows, counts = _create_lancedb_results([[_record(numpy.array([]))]], expected_dim=2)

    assert rows == []
    assert counts["empty_embedding"] == 0
    assert counts["dropped_bad_length"] == 1


def test_a_fully_healthy_batch_raises_nothing() -> None:
    """The whole point: a normal run is untouched.

    Cannot run on the unpatched tree, which has no ``empty_embedding``
    key; the acceptance of all 50 rows is identical there.
    """
    records = [[_record([1.0, 2.0]) for _ in range(50)]]

    rows, counts = _create_lancedb_results(records, expected_dim=2)

    assert len(rows) == 50
    assert counts["accepted"] == 50
    assert counts["empty_embedding"] == 0
    assert counts["dropped_no_embedding"] == 0


def test_a_none_embedding_keeps_its_pre_existing_silent_drop() -> None:
    """``None`` is NOT fatal, because it has a legitimate producer.

    ``operators/embed/text_embed.py`` writes ``{"embedding": None}`` on purpose
    for a row whose text was blank and which it therefore chose not to embed.
    Making ``dropped_no_embedding`` fatal would fail ingests containing such
    rows, which work today. Only ``[]`` - written solely on failure paths, at
    ``models/inference/runtime.py`` and ``models/inference/vllm.py`` - is fatal.

    Cannot run on the unpatched tree, which has no ``empty_embedding``
    key. The drop it asserts is behaviour this change deliberately does not
    touch.
    """
    rows, counts = _create_lancedb_results([[_record(None), _record([1.0, 2.0])]], expected_dim=2)

    assert len(rows) == 1
    assert counts["dropped_no_embedding"] == 1
    assert counts["empty_embedding"] == 0
