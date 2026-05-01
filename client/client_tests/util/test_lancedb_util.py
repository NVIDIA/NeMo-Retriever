import logging
import types
from unittest.mock import Mock

from nv_ingest_client.util.vdb import lancedb as lancedb_mod


def make_sample_results():
    """
    Create sample NV-Ingest pipeline results for testing.

    Returns records in the format produced by the extraction pipeline,
    including document_type which is required for text extraction routing.
    """
    return [
        [
            {
                "document_type": "text",
                "metadata": {
                    "embedding": [0.1, 0.2],
                    "content": "text content a",
                    "content_metadata": {"page_number": 1},
                    "source_metadata": {"source_id": "s1"},
                },
            },
            {
                "document_type": "text",
                "metadata": {
                    "embedding": None,  # Should be skipped - no embedding
                    "content": "skip me",
                    "content_metadata": {"page_number": 2},
                    "source_metadata": {"source_id": "s1"},
                },
            },
        ],
        [
            {
                "document_type": "text",
                "metadata": {
                    "embedding": [0.3, 0.4],
                    "content": "text content b",
                    "content_metadata": {"page_number": 3},
                    "source_metadata": {"source_id": "s2"},
                },
            },
        ],
    ]


def test_create_lancedb_results_filters_and_transforms():
    """Test that create_lancedb_results correctly filters and transforms records."""
    results = make_sample_results()
    out = lancedb_mod.create_lancedb_results(results, expected_dim=2)
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0]["vector"] == [0.1, 0.2]
    assert out[0]["text"] == "text content a"
    assert out[0]["metadata"] == '{"page_number":1}'
    assert out[0]["source"] == '{"source_id":"s1"}'


def test_create_index_uses_lancedb_and_pa_schema(monkeypatch):
    fake_db = Mock()
    fake_table = Mock()
    fake_db.create_table = Mock(return_value=fake_table)

    monkeypatch.setattr(lancedb_mod, "lancedb", Mock(connect=Mock(return_value=fake_db)))
    monkeypatch.setattr(lancedb_mod, "pa", Mock(schema=Mock(return_value="schema")))

    sample = make_sample_results()
    table = lancedb_mod.LanceDB(vector_dim=2).create_index(records=sample)

    fake_db.create_table.assert_called()
    assert table is fake_table


def test_write_to_index_creates_index_and_waits(monkeypatch):
    fake_table = Mock()
    fake_table.list_indices = Mock(
        return_value=[types.SimpleNamespace(name="idx1"), types.SimpleNamespace(name="idx2")]
    )
    fake_table.create_index = Mock()
    fake_table.wait_for_index = Mock()

    ldb = lancedb_mod.LanceDB()
    ldb.write_to_index(records=None, table=fake_table)

    fake_table.create_index.assert_called()
    fake_table.list_indices.assert_called()
    fake_table.wait_for_index.assert_called()


def test_run_calls_create_and_write(monkeypatch):
    ldb = lancedb_mod.LanceDB(hybrid=True, fts_language="Spanish")
    monkeypatch.setattr(ldb, "create_index", Mock(return_value="table_obj"))
    monkeypatch.setattr(ldb, "write_to_index", Mock())

    records = make_sample_results()
    out = ldb.run(records)

    ldb.create_index.assert_called_once_with(records=records, table_name="nv-ingest")
    ldb.write_to_index.assert_called_once_with(
        records,
        table="table_obj",
        index_type=ldb.index_type,
        metric=ldb.metric,
        num_partitions=ldb.num_partitions,
        num_sub_vectors=ldb.num_sub_vectors,
        hybrid=True,
        fts_language="Spanish",
    )
    assert records == out


def test_write_to_index_creates_fts_index_when_hybrid(monkeypatch):
    fake_table = Mock()
    fake_table.list_indices = Mock(
        return_value=[
            types.SimpleNamespace(name="vector_idx"),
            types.SimpleNamespace(name="text_idx"),
        ]
    )
    fake_table.create_index = Mock()
    fake_table.create_fts_index = Mock()
    fake_table.wait_for_index = Mock()

    ldb = lancedb_mod.LanceDB(hybrid=True)
    ldb.write_to_index(records=None, table=fake_table, hybrid=True)

    fake_table.create_fts_index.assert_called_once_with("text", language=ldb.fts_language)


def test_init_default_uri_and_overwrite():
    """Test LanceDB initialization with default values."""
    ldb = lancedb_mod.LanceDB()
    assert ldb.uri == "lancedb"
    assert ldb.overwrite is True


def test_create_lancedb_results_empty_list():
    """Test create_lancedb_results with empty input."""
    results = []
    out = lancedb_mod.create_lancedb_results(results)
    assert out == []


def test_create_lancedb_results_all_none_embeddings():
    """Test create_lancedb_results when all embeddings are None."""
    results = [
        [
            {
                "document_type": "text",
                "metadata": {
                    "embedding": None,
                    "content": "a",
                    "content_metadata": {"page_number": 1},
                    "source_metadata": {"source_id": "s1"},
                },
            },
        ]
    ]
    out = lancedb_mod.create_lancedb_results(results, expected_dim=1)
    assert out == []


def test_create_lancedb_results_mixed_embeddings():
    """Test create_lancedb_results retains only rows whose embedding length matches expected_dim."""
    results = [
        [
            {
                "document_type": "text",
                "metadata": {
                    "embedding": [0.1, 0.2],
                    "content": "keep",
                    "content_metadata": {"page_number": 1},
                    "source_metadata": {"source_id": "s1"},
                },
            },
            {
                "document_type": "text",
                "metadata": {
                    "embedding": None,
                    "content": "skip-no-embedding",
                    "content_metadata": {"page_number": 2},
                    "source_metadata": {"source_id": "s1"},
                },
            },
            {
                "document_type": "text",
                "metadata": {
                    "embedding": [0.3, 0.4],
                    "content": "keep2",
                    "content_metadata": {"page_number": 3},
                    "source_metadata": {"source_id": "s2"},
                },
            },
        ]
    ]
    out = lancedb_mod.create_lancedb_results(results, expected_dim=2)
    assert len(out) == 2
    assert out[0]["text"] == "keep"
    assert out[1]["text"] == "keep2"


def test_create_index_uses_uri_from_instance(monkeypatch):
    """Test that create_index uses the uri stored on the instance."""
    fake_db = Mock()
    fake_table = Mock()
    fake_db.create_table = Mock(return_value=fake_table)

    monkeypatch.setattr(lancedb_mod, "lancedb", Mock(connect=Mock(return_value=fake_db)))
    monkeypatch.setattr(lancedb_mod, "pa", Mock(schema=Mock(return_value="schema")))

    ldb = lancedb_mod.LanceDB(uri="my_custom_uri", vector_dim=2)
    sample = make_sample_results()
    ldb.create_index(records=sample)

    lancedb_mod.lancedb.connect.assert_called_once_with(uri="my_custom_uri")


def test_create_index_uses_table_name_kwarg(monkeypatch):
    """Test that create_index respects table_name keyword argument."""
    fake_db = Mock()
    fake_table = Mock()
    fake_db.create_table = Mock(return_value=fake_table)

    monkeypatch.setattr(lancedb_mod, "lancedb", Mock(connect=Mock(return_value=fake_db)))
    monkeypatch.setattr(lancedb_mod, "pa", Mock(schema=Mock(return_value="schema")))

    ldb = lancedb_mod.LanceDB(vector_dim=2)
    sample = make_sample_results()
    ldb.create_index(records=sample, table_name="custom_table")

    call_args = fake_db.create_table.call_args
    assert call_args[0][0] == "custom_table"


def test_create_index_uses_overwrite_mode(monkeypatch):
    """Test that create_index uses mode based on overwrite setting."""
    fake_db = Mock()
    fake_table = Mock()
    fake_db.create_table = Mock(return_value=fake_table)

    monkeypatch.setattr(lancedb_mod, "lancedb", Mock(connect=Mock(return_value=fake_db)))
    monkeypatch.setattr(lancedb_mod, "pa", Mock(schema=Mock(return_value="schema")))

    ldb = lancedb_mod.LanceDB(overwrite=False, vector_dim=2)
    sample = make_sample_results()
    ldb.create_index(records=sample)

    call_kwargs = fake_db.create_table.call_args[1]
    assert call_kwargs["mode"] == "append"

    fake_db.reset_mock()
    ldb2 = lancedb_mod.LanceDB(overwrite=True, vector_dim=2)
    ldb2.create_index(records=sample)

    call_kwargs = fake_db.create_table.call_args[1]
    assert call_kwargs["mode"] == "overwrite"


def test_write_to_index_respects_kwargs(monkeypatch):
    """Test that write_to_index can accept and use custom index parameters."""
    fake_table = Mock()
    fake_table.list_indices = Mock(return_value=[types.SimpleNamespace(name="idx1")])
    fake_table.create_index = Mock()
    fake_table.create_fts_index = Mock()
    fake_table.wait_for_index = Mock()

    ldb = lancedb_mod.LanceDB(hybrid=True, fts_language="English")
    ldb.write_to_index(records=None, table=fake_table)

    fake_table.create_index.assert_called_once()
    fake_table.create_fts_index.assert_called_once_with("text", language="English")


def test_hybrid_retrieval_search_chain(monkeypatch):
    def fake_infer_microservice(queries, **kwargs):
        fake_infer_microservice.captured_kwargs = kwargs
        return [[0.1, 0.2, 0.3] for _ in queries]

    monkeypatch.setattr(lancedb_mod, "infer_microservice", fake_infer_microservice)

    custom_endpoint = "http://custom-embed:8012/v1"
    custom_model = "custom/embedding-model"
    custom_key = "custom-key"

    captured_calls = {}

    class FakeHybridSearchResult:
        def vector(self, vec):
            captured_calls["vector"] = vec
            return self

        def text(self, txt):
            captured_calls["text"] = txt
            return self

        def select(self, cols):
            return self

        def limit(self, n):
            captured_calls["limit"] = n
            return self

        def refine_factor(self, n):
            captured_calls["refine_factor"] = n
            return self

        def nprobes(self, n):
            captured_calls["nprobes"] = n
            return self

        def rerank(self, reranker):
            captured_calls["reranker"] = reranker
            return self

        def to_list(self):
            return [{"text": "hybrid_hit", "metadata": "1", "source": "s1"}]

    def fake_search(query_type=None):
        captured_calls["query_type"] = query_type
        return FakeHybridSearchResult()

    fake_table = Mock()
    fake_table.search = Mock(side_effect=fake_search)

    results = lancedb_mod.lancedb_hybrid_retrieval(
        queries=["test query"],
        table=fake_table,
        top_k=5,
        refine_factor=25,
        n_probe=32,
        embedding_endpoint=custom_endpoint,
        model_name=custom_model,
        nvidia_api_key=custom_key,
    )

    assert captured_calls["query_type"] == "hybrid"
    assert captured_calls["vector"] == [0.1, 0.2, 0.3]
    assert captured_calls["text"] == "test query"
    assert captured_calls["limit"] == 5
    assert captured_calls["refine_factor"] == 25
    assert captured_calls["nprobes"] == 32
    assert captured_calls["reranker"] is not None
    assert isinstance(captured_calls["reranker"], lancedb_mod.RRFReranker)
    assert len(results) == 1
    assert results[0][0]["entity"]["text"] == "hybrid_hit"
    # Verify custom params were passed to infer_microservice
    assert fake_infer_microservice.captured_kwargs["embedding_endpoint"] == custom_endpoint
    assert fake_infer_microservice.captured_kwargs["model_name"] == custom_model
    assert fake_infer_microservice.captured_kwargs["nvidia_api_key"] == custom_key


def test_create_lancedb_results_missing_content_metadata_graceful():
    """Test that missing content_metadata is handled gracefully (empty dict default)."""
    results = [
        [
            {
                "document_type": "text",
                "metadata": {
                    "embedding": [0.1],
                    "content": "text with missing content_metadata",
                    "source_metadata": {"source_id": "s1"},
                },
            }
        ]
    ]
    out = lancedb_mod.create_lancedb_results(results, expected_dim=1)
    assert len(out) == 1
    assert out[0]["metadata"] == "{}"
    assert out[0]["text"] == "text with missing content_metadata"


# ============ Tests for _get_text_for_element() ============


def test_get_text_for_element_text_type():
    """Test text extraction for document_type='text'."""
    element = {
        "document_type": "text",
        "metadata": {
            "content": "This is plain text content",
        },
    }
    result = lancedb_mod._get_text_for_element(element)
    assert result == "This is plain text content"


def test_get_text_for_element_structured_table():
    """Test text extraction for structured tables."""
    element = {
        "document_type": "structured",
        "metadata": {
            "content_metadata": {"subtype": "table"},
            "table_metadata": {"table_content": "| Col1 | Col2 |\n| A | B |"},
        },
    }
    result = lancedb_mod._get_text_for_element(element)
    assert result == "| Col1 | Col2 |\n| A | B |"


def test_get_text_for_element_structured_chart():
    """Test text extraction for structured charts."""
    element = {
        "document_type": "structured",
        "metadata": {
            "content_metadata": {"subtype": "chart"},
            "table_metadata": {"table_content": "Chart showing revenue trends"},
        },
    }
    result = lancedb_mod._get_text_for_element(element)
    assert result == "Chart showing revenue trends"


def test_get_text_for_element_structured_infographic():
    """Test text extraction for infographics."""
    element = {
        "document_type": "structured",
        "metadata": {
            "content_metadata": {"subtype": "infographic"},
            "table_metadata": {"table_content": "Infographic content description"},
        },
    }
    result = lancedb_mod._get_text_for_element(element)
    assert result == "Infographic content description"


def test_get_text_for_element_image_caption():
    """Test text extraction for images uses caption (not base64 data)."""
    element = {
        "document_type": "image",
        "metadata": {
            "content_metadata": {"subtype": "image"},
            "image_metadata": {"caption": "A photo of a dog playing"},
        },
    }
    result = lancedb_mod._get_text_for_element(element)
    assert result == "A photo of a dog playing"


def test_get_text_for_element_page_image_uses_text():
    """Test that page_image subtype uses 'text' field (OCR) instead of caption."""
    element = {
        "document_type": "image",
        "metadata": {
            "content_metadata": {"subtype": "page_image"},
            "image_metadata": {
                "text": "OCR extracted text from page",
                "caption": "Should not use this",
            },
        },
    }
    result = lancedb_mod._get_text_for_element(element)
    assert result == "OCR extracted text from page"


def test_get_text_for_element_audio():
    """Test text extraction for audio uses transcript."""
    element = {
        "document_type": "audio",
        "metadata": {
            "audio_metadata": {"audio_transcript": "Hello, this is the audio transcript"},
        },
    }
    result = lancedb_mod._get_text_for_element(element)
    assert result == "Hello, this is the audio transcript"


def test_get_text_for_element_unknown_type_fallback():
    """Test that unknown document types fall back to metadata.content."""
    element = {
        "document_type": "unknown_type",
        "metadata": {
            "content": "Fallback content",
        },
    }
    result = lancedb_mod._get_text_for_element(element)
    assert result == "Fallback content"


def test_get_text_for_element_missing_metadata():
    """Test graceful handling of missing metadata fields."""
    element = {
        "document_type": "text",
        "metadata": {},  # No content field
    }
    result = lancedb_mod._get_text_for_element(element)
    assert result is None


# ============ Regression tests for variable-length vector handling ============


def _embedding_record(embedding, page: int = 1, content: str = "row"):
    """Build a minimal NV-Ingest pipeline record for variable-length-vector tests."""
    return {
        "document_type": "text",
        "metadata": {
            "embedding": embedding,
            "content": content,
            "content_metadata": {"page_number": page},
            "source_metadata": {"source_id": f"s{page}"},
        },
    }


def test_create_lancedb_results_drops_variable_length_vectors(caplog):
    """Mixed-length embeddings: only rows whose length matches expected_dim survive,
    and a single structured WARNING summary is emitted listing the drop counts.

    This guards against the regression where a small fraction of empty/short
    embeddings caused LanceDB's strict fixed-size schema to abort the entire
    write at the end of a long ingestion run.
    """
    expected_dim = 4
    results = [
        [
            _embedding_record([0.1, 0.2, 0.3, 0.4], page=1, content="good_a"),
            _embedding_record([], page=2, content="empty_should_drop"),
            _embedding_record([0.1, 0.2], page=3, content="short_should_drop"),
            _embedding_record(None, page=4, content="none_should_drop"),
            _embedding_record([0.5, 0.6, 0.7, 0.8], page=5, content="good_b"),
        ]
    ]

    with caplog.at_level(logging.WARNING, logger=lancedb_mod.logger.name):
        out = lancedb_mod.create_lancedb_results(results, expected_dim=expected_dim)

    assert [row["text"] for row in out] == ["good_a", "good_b"]
    summary_records = [
        r for r in caplog.records if r.levelno == logging.WARNING and "create_lancedb_results" in r.getMessage()
    ]
    assert len(summary_records) == 1
    summary = summary_records[0].getMessage()
    assert "accepted=2" in summary
    assert "dropped_no_embedding=1" in summary
    assert "dropped_bad_length=2" in summary
    assert f"expected_dim={expected_dim}" in summary


def test_create_lancedb_results_respects_expected_dim():
    """create_lancedb_results filters strictly by the caller-specified expected_dim."""
    results = [
        [
            _embedding_record([0.0] * 128, page=1, content="dim_128"),
            _embedding_record([0.0] * 256, page=2, content="dim_256_should_drop"),
        ]
    ]
    out = lancedb_mod.create_lancedb_results(results, expected_dim=128)
    assert len(out) == 1
    assert out[0]["text"] == "dim_128"
    assert len(out[0]["vector"]) == 128


def test_lancedb_init_validates_on_bad_vectors():
    """LanceDB.__init__ rejects unknown on_bad_vectors policies and non-positive vector_dim."""
    import pytest

    with pytest.raises(ValueError, match="on_bad_vectors must be one of"):
        lancedb_mod.LanceDB(on_bad_vectors="bogus")

    with pytest.raises(ValueError, match="vector_dim must be positive"):
        lancedb_mod.LanceDB(vector_dim=0)

    with pytest.raises(ValueError, match="vector_dim must be positive"):
        lancedb_mod.LanceDB(vector_dim=-5)

    ldb = lancedb_mod.LanceDB(on_bad_vectors="DROP")
    assert ldb.on_bad_vectors == "drop"
    ldb_fill = lancedb_mod.LanceDB(on_bad_vectors=" Fill ", fill_value=1.5)
    assert ldb_fill.on_bad_vectors == "fill"
    assert ldb_fill.fill_value == 1.5


def test_create_index_passes_on_bad_vectors_to_lancedb(monkeypatch):
    """create_index forwards on_bad_vectors (and fill_value when applicable) to db.create_table."""
    fake_db = Mock()
    fake_table = Mock()
    fake_db.create_table = Mock(return_value=fake_table)

    monkeypatch.setattr(lancedb_mod, "lancedb", Mock(connect=Mock(return_value=fake_db)))
    monkeypatch.setattr(lancedb_mod, "pa", Mock(schema=Mock(return_value="schema")))

    sample = make_sample_results()

    ldb_default = lancedb_mod.LanceDB(vector_dim=2)
    ldb_default.create_index(records=sample)
    default_kwargs = fake_db.create_table.call_args.kwargs
    assert default_kwargs["on_bad_vectors"] == "drop"
    assert "fill_value" not in default_kwargs

    fake_db.create_table.reset_mock()

    ldb_fill = lancedb_mod.LanceDB(vector_dim=2, on_bad_vectors="fill", fill_value=0.0)
    ldb_fill.create_index(records=sample)
    fill_kwargs = fake_db.create_table.call_args.kwargs
    assert fill_kwargs["on_bad_vectors"] == "fill"
    assert fill_kwargs["fill_value"] == 0.0

    fake_db.create_table.reset_mock()

    ldb_null = lancedb_mod.LanceDB(vector_dim=2, on_bad_vectors="null")
    ldb_null.create_index(records=sample)
    null_kwargs = fake_db.create_table.call_args.kwargs
    assert null_kwargs["on_bad_vectors"] == "null"
    assert "fill_value" not in null_kwargs


def test_create_index_uses_configured_vector_dim_in_schema(monkeypatch):
    """The vector field declared on the LanceDB schema reflects the configured vector_dim."""
    fake_db = Mock()
    fake_table = Mock()
    fake_db.create_table = Mock(return_value=fake_table)

    captured_dims = []

    def fake_list_(_dtype, dim):
        captured_dims.append(dim)
        return ("fixed_size_list", dim)

    fake_pa = Mock(
        schema=Mock(return_value="schema"),
        list_=Mock(side_effect=fake_list_),
        float32=Mock(return_value="float32"),
        string=Mock(return_value="string"),
        field=Mock(side_effect=lambda name, dtype: (name, dtype)),
    )

    monkeypatch.setattr(lancedb_mod, "lancedb", Mock(connect=Mock(return_value=fake_db)))
    monkeypatch.setattr(lancedb_mod, "pa", fake_pa)

    ldb = lancedb_mod.LanceDB(vector_dim=512)
    ldb.create_index(records=[])

    assert 512 in captured_dims
