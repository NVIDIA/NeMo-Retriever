# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the tabular-data-specific LanceDB subclass.

``TabularLanceDB`` promotes ``label`` and ``database_name`` to first-class
top-level columns so the tabular retrieval layer can filter them with
plain SQL — the same shape Postgres already uses. These tests cover:

* The subclass schema exposes the extra columns.
* The per-row hook populates them from ``content_metadata``.
* End-to-end: ingest → ``WHERE label = '…' AND database_name = '…'`` → hits.
* The base ``LanceDB`` operator is **unchanged** (no extra columns leak
  into the reference schema).
* The VDB factory dispatches ``"tabular_lancedb"`` to the subclass.
"""

from __future__ import annotations

import tempfile

import pytest

lancedb = pytest.importorskip("lancedb")

from nemo_retriever.tabular_data.vdb import TabularLanceDB
from nemo_retriever.vdb.factory import get_vdb_op_cls
from nemo_retriever.vdb.lancedb import LanceDB
from nemo_retriever.vdb.operators import IngestVdbOperator, RetrieveVdbOperator


def _records_with_labels() -> list[list[dict]]:
    return [
        [
            {
                "document_type": "text",
                "metadata": {
                    "embedding": [1.0, 0.0],
                    "content": "alpha column row",
                    "content_metadata": {
                        "id": "c1",
                        "label": "Column",
                        "database_name": "dor_prod",
                    },
                    "source_metadata": {"source_id": "src://1"},
                },
            },
            {
                "document_type": "text",
                "metadata": {
                    "embedding": [0.0, 1.0],
                    "content": "beta table row",
                    "content_metadata": {
                        "id": "t1",
                        "label": "Table",
                        "database_name": "dor_prod",
                    },
                    "source_metadata": {"source_id": "src://2"},
                },
            },
            {
                "document_type": "text",
                "metadata": {
                    "embedding": [1.0, 1.0],
                    "content": "gamma column row in other db",
                    "content_metadata": {
                        "id": "c2",
                        "label": "Column",
                        "database_name": "other_db",
                    },
                    "source_metadata": {"source_id": "src://3"},
                },
            },
            # No label / database_name in content_metadata — should land as null.
            {
                "document_type": "text",
                "metadata": {
                    "embedding": [0.5, 0.5],
                    "content": "row with no tabular fields",
                    "content_metadata": {"id": "x"},
                    "source_metadata": {"source_id": "src://4"},
                },
            },
        ]
    ]


def test_factory_does_not_expose_tabular_lancedb() -> None:
    """The reference factory must not surface vertical-specific subclasses.

    ``TabularLanceDB`` is a tabular-vertical extension; the tabular code is
    expected to construct it directly and inject the instance via
    ``vdb=<instance>``. Registering it in the shared factory would expose
    external NRL customers to a name that is irrelevant to them.
    """
    with pytest.raises(ValueError, match="Invalid vdb_op"):
        get_vdb_op_cls("tabular_lancedb")


def test_instance_injection_via_ingest_and_retrieve_operators() -> None:
    """The graph operators accept ``vdb=<TabularLanceDB instance>`` directly,
    which is the wiring tabular_data uses instead of the shared factory.
    """
    vdb = TabularLanceDB(
        uri=tempfile.mkdtemp(),
        table_name="t",
        overwrite=True,
        vector_dim=2,
        validate_vector_length=False,
    )
    ingest = IngestVdbOperator(vdb=vdb)
    retrieve = RetrieveVdbOperator(vdb=vdb)
    assert ingest._vdb is vdb
    assert retrieve._vdb is vdb


def test_schema_exposes_label_and_database_name_columns() -> None:
    op = TabularLanceDB(
        uri=tempfile.mkdtemp(),
        table_name="t",
        overwrite=True,
        vector_dim=2,
        validate_vector_length=False,
    )
    schema = op._build_schema()
    field_names = {f.name for f in schema}
    assert {"vector", "text", "metadata", "source", "label", "database_name"} <= field_names

    # The extra columns must be nullable so non-tabular rows (or rows missing
    # the field) can be written without forcing a value.
    assert schema.field("label").nullable is True
    assert schema.field("database_name").nullable is True


def test_base_lancedb_schema_is_unchanged() -> None:
    """The reference (non-tabular) LanceDB schema must not leak tabular concerns."""
    op = LanceDB(
        uri=tempfile.mkdtemp(),
        table_name="t",
        overwrite=True,
        vector_dim=2,
        validate_vector_length=False,
    )
    field_names = {f.name for f in op._build_schema()}
    assert field_names == {"vector", "text", "metadata", "source"}


def test_make_row_extracts_label_and_database_name() -> None:
    op = TabularLanceDB(
        uri=tempfile.mkdtemp(),
        table_name="t",
        overwrite=True,
        vector_dim=2,
        validate_vector_length=False,
    )
    row = op._make_row(
        element={"document_type": "text"},
        embedding=[1.0, 0.0],
        text="x",
        content_meta={"label": "Column", "database_name": "dor_prod"},
        source_meta={"source_id": "s"},
    )
    assert row["label"] == "Column"
    assert row["database_name"] == "dor_prod"
    assert row["vector"] == [1.0, 0.0]
    assert row["text"] == "x"


def test_make_row_handles_missing_tabular_fields() -> None:
    op = TabularLanceDB(
        uri=tempfile.mkdtemp(),
        table_name="t",
        overwrite=True,
        vector_dim=2,
        validate_vector_length=False,
    )
    row = op._make_row(
        element={"document_type": "text"},
        embedding=[1.0, 0.0],
        text="x",
        content_meta={"id": "only-id"},
        source_meta={},
    )
    assert row["label"] is None
    assert row["database_name"] is None


def test_end_to_end_filter_by_label_and_database_name() -> None:
    """Ingest → ``WHERE label = '…' AND database_name = '…'`` → expected hits."""
    d = tempfile.mkdtemp()
    op = TabularLanceDB(
        uri=d,
        table_name="tabular",
        overwrite=True,
        vector_dim=2,
        validate_vector_length=False,
    )
    op.create_index(records=_records_with_labels(), table_name="tabular")

    # Sanity: the on-disk schema actually carries the new columns.
    table = lancedb.connect(d).open_table("tabular")
    table_fields = {f.name for f in table.schema}
    assert {"label", "database_name"} <= table_fields

    # Single label.
    hits = op.retrieval(
        [[1.0, 0.0]],
        top_k=10,
        table_path=d,
        table_name="tabular",
        where="label = 'Column'",
    )
    assert {h["label"] for h in hits[0]} == {"Column"}

    # Label + database_name (the predicate ``semantic_search`` emits).
    hits = op.retrieval(
        [[1.0, 0.0]],
        top_k=10,
        table_path=d,
        table_name="tabular",
        where="label = 'Column' AND database_name = 'dor_prod'",
    )
    assert len(hits[0]) == 1
    assert hits[0][0]["text"] == "alpha column row"
    assert hits[0][0]["database_name"] == "dor_prod"

    # IN-list across multiple labels.
    hits = op.retrieval(
        [[1.0, 0.0]],
        top_k=10,
        table_path=d,
        table_name="tabular",
        where="label IN ('Column', 'Table') AND database_name = 'dor_prod'",
    )
    assert {h["label"] for h in hits[0]} == {"Column", "Table"}

    # Rows without ``label`` / ``database_name`` land with null and so are filtered out.
    hits = op.retrieval(
        [[0.5, 0.5]],
        top_k=10,
        table_path=d,
        table_name="tabular",
        where="label IS NULL",
    )
    assert len(hits[0]) == 1
    assert hits[0][0]["text"] == "row with no tabular fields"


def test_dropped_rows_do_not_misalign_label_projection() -> None:
    """Records lacking text are dropped by the base filter — the tabular subclass
    must not produce a positional mismatch when constructing the row dicts.
    """
    op = TabularLanceDB(
        uri=tempfile.mkdtemp(),
        table_name="t",
        overwrite=True,
        vector_dim=2,
        validate_vector_length=False,
    )
    records = [
        [
            # First record has an embedding but no text → dropped.
            {
                "document_type": "text",
                "metadata": {
                    "embedding": [1.0, 0.0],
                    # no 'content'
                    "content_metadata": {"label": "Dropped", "database_name": "dropped_db"},
                    "source_metadata": {},
                },
            },
            # Second record survives.
            {
                "document_type": "text",
                "metadata": {
                    "embedding": [0.0, 1.0],
                    "content": "kept row",
                    "content_metadata": {"label": "Kept", "database_name": "kept_db"},
                    "source_metadata": {},
                },
            },
        ]
    ]
    rows, counts = op._build_rows(records, expected_dim=2)
    assert counts["accepted"] == 1
    assert counts["dropped_no_text"] == 1
    assert len(rows) == 1
    # Must be the surviving row's labels, not the dropped row's.
    assert rows[0]["label"] == "Kept"
    assert rows[0]["database_name"] == "kept_db"
