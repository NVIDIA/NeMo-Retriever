# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd
import pytest

from nemo_retriever.graph import InprocessExecutor, ParquetReaderOperator, ParquetWriterOperator, UDFOperator
from nemo_retriever.operators.cpu_operator import CPUOperator


def _rows():
    return pd.DataFrame(
        {
            "text": ["first page", "a table"],
            "page_number": [1, 2],
            "_content_type": ["text", "table"],
            "metadata": [{"has_text": True, "dpi": 200}, {"has_text": False, "dpi": 200}],
            "page_image": [{"image_b64": "AAAA"}, {"image_b64": "BBBB"}],
        }
    )


def test_parquet_operators_are_cpu_operators():
    assert isinstance(ParquetWriterOperator(output_dir="unused"), CPUOperator)
    assert isinstance(ParquetReaderOperator(), CPUOperator)


def test_writer_passes_rows_through_and_writes_one_part_per_batch(tmp_path):
    writer = ParquetWriterOperator(output_dir=str(tmp_path))
    rows = _rows()

    assert writer.run(rows) is rows
    writer.run(rows.iloc[:1])

    assert len(list(tmp_path.glob("part-*.parquet"))) == 2


def test_writer_skips_empty_batches(tmp_path):
    ParquetWriterOperator(output_dir=str(tmp_path / "out")).run(pd.DataFrame())

    assert not (tmp_path / "out").exists()


def test_writer_drops_excluded_columns(tmp_path):
    ParquetWriterOperator(output_dir=str(tmp_path), exclude_columns=["page_image", "missing"]).run(_rows())

    loaded = ParquetReaderOperator().run(str(tmp_path))

    assert "page_image" not in loaded.columns
    assert loaded["text"].tolist() == ["first page", "a table"]


def test_round_trip_preserves_nested_metadata(tmp_path):
    rows = _rows()
    ParquetWriterOperator(output_dir=str(tmp_path)).run(rows)

    loaded = ParquetReaderOperator().run(tmp_path)

    assert loaded["metadata"].tolist() == rows["metadata"].tolist()
    assert loaded["page_number"].tolist() == [1, 2]


@pytest.mark.parametrize("as_list", [False, True])
def test_reader_accepts_files_and_directories(tmp_path, as_list):
    rows = _rows()
    rows.to_parquet(tmp_path / "single.parquet", index=False)
    parts = tmp_path / "parts"
    ParquetWriterOperator(output_dir=str(parts)).run(rows)
    ParquetWriterOperator(output_dir=str(parts)).run(rows)

    source = [str(tmp_path / "single.parquet"), str(parts)] if as_list else str(parts)
    loaded = ParquetReaderOperator().run(source)

    assert len(loaded) == (6 if as_list else 4)


def test_reader_rejects_unsupported_input():
    with pytest.raises(TypeError, match="Parquet path"):
        ParquetReaderOperator().run(42)


def test_reader_rejects_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        ParquetReaderOperator().run(tmp_path / "missing")


def test_reader_reads_delivered_bytes_when_path_is_not_local(tmp_path):
    rows = _rows()
    ParquetWriterOperator(output_dir=str(tmp_path)).run(rows)
    payload = next(tmp_path.glob("*.parquet")).read_bytes()
    batch = pd.DataFrame({"bytes": [payload], "path": ["/on/another/node/part-0.parquet"]})

    loaded = ParquetReaderOperator().run(batch)

    assert loaded["metadata"].tolist() == rows["metadata"].tolist()


def test_reader_falls_back_to_path_when_bytes_are_missing(tmp_path):
    ParquetWriterOperator(output_dir=str(tmp_path)).run(_rows())
    path = next(tmp_path.glob("*.parquet"))

    loaded = ParquetReaderOperator().run(pd.DataFrame({"bytes": [None], "path": [str(path)]}))

    assert loaded["text"].tolist() == ["first page", "a table"]


def test_reader_returns_empty_frame_for_empty_directory(tmp_path):
    assert ParquetReaderOperator().run(tmp_path).empty


def test_reader_runs_as_graph_root_under_inprocess_executor(tmp_path):
    ParquetWriterOperator(output_dir=str(tmp_path / "parts")).run(_rows())
    graph = ParquetReaderOperator() >> UDFOperator(lambda df: df.assign(chars=df["text"].str.len()), name="Chars")

    result = InprocessExecutor(graph).ingest(sorted((tmp_path / "parts").glob("*.parquet")))

    assert result["chars"].tolist() == [10, 7]
