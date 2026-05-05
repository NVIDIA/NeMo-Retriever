import json
from pathlib import Path

import pandas as pd
import pytest

from nemo_retriever.io import build_page_index, to_markdown, to_markdown_by_page


class _LazyRows:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)


def test_to_markdown_renders_page_dataframe() -> None:
    df = pd.DataFrame(
        [
            {
                "page_number": 1,
                "text": "Executive summary",
                "table": [{"text": "| Animal | Count |\n| --- | --- |\n| Cat | 2 |"}],
                "chart": [{"text": "Quarterly growth remained positive."}],
                "infographic": [],
            },
            {
                "page_number": 2,
                "text": "Appendix",
                "table": [],
                "chart": [],
                "infographic": [{"text": "Icon legend and callouts."}],
            },
        ]
    )

    markdown = to_markdown(df)

    assert markdown.startswith("# Extracted Content")
    assert "## Page 1" in markdown
    assert "Executive summary" in markdown
    assert "### Table 1" in markdown
    assert "### Chart 1" in markdown
    assert "## Page 2" in markdown
    assert "### Infographic 1" in markdown


def test_to_markdown_by_page_sorts_pages_and_groups_unknown() -> None:
    pages = to_markdown_by_page(
        [
            {"page_number": "2", "text": "Second page"},
            {"page_number": None, "text": "Unknown page"},
            {"page_number": 1, "text": "First page"},
            {"page_number": 2, "text": "Second page"},
        ]
    )

    assert list(pages.keys()) == [1, 2, -1]
    assert pages[1].startswith("## Page 1")
    assert pages[2].count("Second page") == 1
    assert pages[-1].startswith("## Page Unknown")


def test_to_markdown_supports_primitive_rows_from_lazy_iterable() -> None:
    rows = _LazyRows(
        [
            {
                "document_type": "text",
                "metadata": {
                    "content": "Page text",
                    "content_metadata": {"page_number": 1},
                },
            },
            {
                "document_type": "structured",
                "metadata": {
                    "content_metadata": {"page_number": 1, "subtype": "table"},
                    "table_metadata": {"table_content": "| A |\n| --- |\n| 1 |"},
                },
            },
            {
                "document_type": "image",
                "metadata": {
                    "content_metadata": {"page_number": 2, "subtype": "page_image"},
                    "image_metadata": {"text": "OCR fallback"},
                },
            },
        ]
    )

    pages = to_markdown_by_page(rows)

    assert "Page text" in pages[1]
    assert "### Table 1" in pages[1]
    assert "### Page Image 1" in pages[2]
    assert "OCR fallback" in pages[2]


def test_to_markdown_reads_saved_records_wrapper(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    payload = {
        "records": [
            {
                "page_number": 1,
                "text": "Saved result text",
                "table": [{"text": "| H |\n| --- |\n| V |"}],
                "metadata": {"source_path": "/tmp/example.pdf"},
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    markdown = to_markdown(path)

    assert "Saved result text" in markdown
    assert "### Table 1" in markdown


def test_to_markdown_empty_results_returns_none() -> None:
    assert to_markdown([]) is None


def test_to_markdown_rejects_multi_document_results() -> None:
    doc_a = pd.DataFrame([{"page_number": 1, "text": "A"}])
    doc_b = pd.DataFrame([{"page_number": 1, "text": "B"}])

    with pytest.raises(ValueError, match="single document result"):
        to_markdown([doc_a, doc_b])


def test_build_page_index_prunes_irrelevant_columns_on_dataframe_path() -> None:
    """Regression test: ``build_page_index(dataframe=)`` must not materialise
    huge columns that the markdown renderer never reads.

    Before the fix, the ``dataframe=`` branch passed the user's DataFrame
    straight through to ``df.iterrows()`` + ``row.to_dict()``, which
    materialised every column -- including ``page_image`` base64 blobs and
    embedding vectors -- for every record.  That produced the same multi-GB
    memory spikes ``_read_parquet_for_markdown`` was explicitly built to
    avoid on the Parquet path.  The fix mirrors the Parquet-path column
    pruning via ``_MARKDOWN_PARQUET_COLUMNS``.

    This test verifies three guarantees:
        1. Rendering still produces the same markdown output.
        2. A huge extraneous column does not propagate into the rendered
           records (catches the bug by construction).
        3. The caller's DataFrame is not mutated in place.
    """
    large_blob = "x" * 100_000
    df = pd.DataFrame(
        [
            {
                "path": "/tmp/doc.pdf",
                "page_number": 1,
                "text": "First page text",
                "page_image": large_blob,
                "embedding": [0.0] * 1024,
            },
            {
                "path": "/tmp/doc.pdf",
                "page_number": 2,
                "text": "Second page text",
                "page_image": large_blob,
                "embedding": [0.0] * 1024,
            },
        ]
    )
    original_columns = set(df.columns)

    index, failures = build_page_index(dataframe=df)

    assert not failures
    assert "/tmp/doc.pdf" in index
    rendered = index["/tmp/doc.pdf"]
    assert "1" in rendered and "2" in rendered
    assert "First page text" in rendered["1"]
    assert "Second page text" in rendered["2"]

    for page_md in rendered.values():
        assert large_blob not in page_md, "huge column leaked into rendered markdown"

    assert set(df.columns) == original_columns, "caller's DataFrame must not be mutated"
    assert "page_image" in df.columns
    assert "embedding" in df.columns


def test_build_page_index_no_op_when_all_columns_are_allow_listed() -> None:
    """When the caller already supplies a pruned DataFrame, the filter is
    a no-op: ``df`` is identical (same object, same columns)."""
    df = pd.DataFrame(
        [
            {
                "path": "/tmp/doc.pdf",
                "page_number": 1,
                "text": "Only essentials",
            }
        ]
    )

    index, failures = build_page_index(dataframe=df)

    assert not failures
    assert "/tmp/doc.pdf" in index
    assert "Only essentials" in index["/tmp/doc.pdf"]["1"]


def test_build_page_index_preserves_content_fallback_column() -> None:
    """Guards against regressing the ``content`` fallback path.

    ``_collect_page_record`` reads ``record.get("content")`` as a tertiary
    fallback when ``record.get("text")`` is absent.  The allow-list must
    include ``content`` so rows that carry only that column still render.
    """
    df = pd.DataFrame(
        [
            {
                "path": "/tmp/content_only.pdf",
                "page_number": 1,
                "content": "Fallback body text",
                "page_image": "x" * 100_000,
            }
        ]
    )

    index, failures = build_page_index(dataframe=df)

    assert not failures
    assert "Fallback body text" in index["/tmp/content_only.pdf"]["1"]
