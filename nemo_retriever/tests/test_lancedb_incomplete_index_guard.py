# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The LanceDB writer must refuse to build an index that is missing rows.

The property under test: a row must not reach the index without an embedding,
and a run that loses rows must fail rather than publish a short index and report
success. It holds for any cause, any backend, and any GPU - the writer sees only
the rows, not what produced them.

Concretely, a row whose embedding is ``[]`` fails the write. The embed stage
emits that shape for a whole batch it could not embed. It used to be counted as
a wrong-length vector and dropped with a warning, so the run completed with a
short index.

Three things are deliberately NOT covered here.

* A wrong-length vector. That is a real vector against the wrong schema, which
  is what ``on_bad_vectors`` exists for; folding it in would turn a configured
  tolerance into a hard failure on upgrade.
* An absent or ``None`` embedding, which keeps its pre-existing silent drop
  because ``operators/embed/text_embed.py`` writes ``None`` on purpose for a
  blank-text row.
* A row the embedder zero-padded. It has the correct width and a non-zero
  length, so the writer cannot see it. That half is marked at the source; see
  ``test_vllm_embed.py``.

Scope: this guard covers the LanceDB writer. Other paths in the repo still drop
or zero-fill rows of their own accord.

Every test here is deterministic and needs no GPU. They assert on the write
contract rather than on end-to-end accuracy, which would drift with corpus size.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from nemo_retriever.common.vdb.adt_vdb import CollectionWriteContext
from nemo_retriever.common.vdb.lancedb import _create_lancedb_results
from nemo_retriever.common.vdb.lancedb_collections import _collection_rows


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


# --------------------------------------------------------------------------
# No false positives: everything the fatal condition must NOT fire on.
#
# Catching the defect is half the evidence. The other half is that a normal run
# still finishes, so there is one case here per legitimate drop and per value
# type the new check could have swallowed by accident.
# --------------------------------------------------------------------------


def test_wrong_length_is_counted_and_never_fatal() -> None:
    """``dropped_bad_length`` stays out of the fatal condition.

    It is the category ``on_bad_vectors`` governs: a real vector against the
    wrong schema. Folding it in is exactly the regression that would turn a
    user's configured ``drop``/``fill``/``null`` tolerance into a hard failure
    on upgrade.

    Cannot run on the unpatched tree: it asserts on the new
    ``empty_embedding`` key, which does not exist there. What it pins is
    a contract, not a behaviour change - the wrong-length row is dropped and not
    raised on, before and after.
    """
    rows, counts = _create_lancedb_results([[_record([1.0, 2.0, 3.0]), _record([1.0, 2.0])]], expected_dim=2)

    assert len(rows) == 1
    assert counts["dropped_bad_length"] == 1
    assert counts["empty_embedding"] == 0


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


def test_numpy_embeddings_keep_their_existing_handling() -> None:
    """The new check must not change how non-list types are treated.

    ``not embedding`` raises ``ValueError`` on a multi-element numpy array, so
    the check is written as ``isinstance(embedding, (list, tuple)) and
    len(embedding) == 0``. A numpy row therefore falls through to the
    pre-existing length check and behaves exactly as it does today: counted as
    ``dropped_bad_length``, not raised on, not crashed on.

    Not a known-bad test: it pins a hazard the obvious spelling of this check
    would have introduced. It cannot run on the unpatched tree, which has no
    ``empty_embedding`` key.
    """
    numpy = pytest.importorskip("numpy")

    rows, counts = _create_lancedb_results(
        [[_record(numpy.array([1.0, 2.0])), _record([1.0, 2.0])]],
        expected_dim=2,
    )

    assert len(rows) == 1
    assert counts["dropped_bad_length"] == 1
    assert counts["empty_embedding"] == 0


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


# ---------------------------------------------------------------------------
# The collection-managed write path.
#
# ``common/vdb/lancedb_collections.py::_collection_rows`` is the second writer
# in the repo. It builds the rows for a collection-managed document ingest and
# had the same defect as ``_create_lancedb_results``: ``not vector`` is true for
# ``[]``, so a row the embed stage failed to embed was skipped with no counter,
# no log and no failure, and the document was published short while the ingest
# reported success. Same property, same fatal condition, same explicit
# ``isinstance``/``len`` spelling.
# ---------------------------------------------------------------------------


def _collection_context() -> CollectionWriteContext:
    return CollectionWriteContext(
        scope="workspace-a",
        collection_name="collection-a",
        document_id="document-a",
        document_version="v1",
        content_sha256="sha-v1",
        filename="source.pdf",
        job_id="job-a",
        operation="append",
    )


def _collection_record(embedding: Any, *, text: str = "first chunk") -> dict:
    return {
        "document_type": "text",
        "metadata": {
            "embedding": embedding,
            "content": text,
            "content_metadata": {"type": "text", "page_number": 2},
            "source_metadata": {"source_id": "/inputs/source.pdf", "source_name": "source.pdf"},
        },
    }


def test_an_empty_embedding_on_the_collection_path_fails_the_write() -> None:
    """Known-bad: returned the surviving rows and skipped the empty one silently.

    On the unpatched tree ``not vector`` swallowed ``[]`` into the same branch as
    a malformed value, so this returned one row and no error, and the collection
    document was written short. It now raises before any row reaches LanceDB.

    Fails on the unpatched tree: no exception is raised.
    """
    records = [[_collection_record([]), _collection_record([1.0, 0.0])]]

    with pytest.raises(RuntimeError) as excinfo:
        _collection_rows(records, context=_collection_context())

    message = str(excinfo.value)
    assert "incomplete document" in message
    assert "empty_embedding=1" in message


def test_the_collection_path_reports_the_same_counter_name_as_the_pipeline_path() -> None:
    """Both writers name the condition ``empty_embedding`` so one grep finds both.

    Fails on the unpatched tree: no exception, so nothing to read the name from.
    """
    with pytest.raises(RuntimeError) as excinfo:
        _collection_rows([[_collection_record([])]], context=_collection_context())

    assert "empty_embedding" in str(excinfo.value)


def test_a_healthy_collection_document_still_writes_every_row() -> None:
    """False-failure guard: the fatal branch must not fire on good input.

    Runs on the unpatched tree and passes there too, which is the point.
    """
    records = [[_collection_record([1.0, 0.0]), _collection_record([0.0, 1.0], text="second chunk")]]

    rows = _collection_rows(records, context=_collection_context())

    assert len(rows) == 2
    assert [row["text"] for row in rows] == ["first chunk", "second chunk"]


@pytest.mark.parametrize(
    "embedding",
    [
        None,
        "not-a-vector",
        123,
        {"dense": [1.0, 0.0]},
    ],
    ids=["none", "string", "int", "dict"],
)
def test_a_malformed_collection_embedding_keeps_its_pre_existing_silent_skip(embedding: Any) -> None:
    """False-failure guard, and the one that matters most.

    ``not isinstance(vector, (list, tuple))`` covers malformed values. It was
    NOT made fatal: only ``[]`` was, because ``[]`` is the sole value written
    exclusively on a failure path. Widening the fatal set to these would fail
    ingests that work today. The healthy sibling row must still be written.

    Runs on the unpatched tree and passes there too.
    """
    records = [[_collection_record(embedding), _collection_record([1.0, 0.0], text="good chunk")]]

    rows = _collection_rows(records, context=_collection_context())

    assert len(rows) == 1
    assert rows[0]["text"] == "good chunk"


def test_a_malformed_collection_batch_or_record_keeps_its_pre_existing_silent_skip() -> None:
    """False-failure guard for the batch- and record-shaped skips.

    ``not isinstance(batch, list)``, ``not isinstance(record, dict)`` and
    ``not isinstance(metadata, dict)`` are untouched: they still skip silently
    and never fail the write.

    Runs on the unpatched tree and passes there too.
    """
    records = [
        "not-a-batch",
        ["not-a-record"],
        [{"document_type": "text", "metadata": "not-a-dict"}],
        [_collection_record([1.0, 0.0], text="good chunk")],
    ]

    rows = _collection_rows(records, context=_collection_context())

    assert len(rows) == 1
    assert rows[0]["text"] == "good chunk"


def test_a_text_free_non_image_collection_row_is_skipped_and_never_fatal() -> None:
    """False-failure guard: the content filter keeps its silent drop.

    Runs on the unpatched tree and passes there too.
    """
    records = [[_collection_record([1.0, 0.0], text="   "), _collection_record([0.0, 1.0], text="good chunk")]]

    rows = _collection_rows(records, context=_collection_context())

    assert len(rows) == 1
    assert rows[0]["text"] == "good chunk"


def test_a_numpy_collection_embedding_does_not_raise_on_truthiness() -> None:
    """The new branch must never evaluate ``not vector`` on an array.

    A multi-element numpy array raises ``ValueError`` on ``bool()``. The check
    is written as ``isinstance(vector, (list, tuple)) and len(vector) == 0``, so
    an array short-circuits out of it and keeps its pre-existing skip rather
    than crashing the ingest.

    Runs on the unpatched tree, where the existing ``or`` short-circuits for the
    same reason, and passes there too.
    """
    numpy = pytest.importorskip("numpy")

    records = [
        [
            _collection_record(numpy.array([1.0, 0.0])),
            _collection_record(numpy.array([])),
            _collection_record([1.0, 0.0], text="good chunk"),
        ]
    ]

    rows = _collection_rows(records, context=_collection_context())

    assert len(rows) == 1
    assert rows[0]["text"] == "good chunk"
