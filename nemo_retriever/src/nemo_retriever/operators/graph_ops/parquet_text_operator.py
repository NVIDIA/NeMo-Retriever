# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize projected Parquet text rows for embedding and VDB ingestion."""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from nemo_retriever.graph.designer import designer_component
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.cpu_operator import CPUOperator


@designer_component(
    name="Parquet Text",
    category="Data Loading",
    compute="cpu",
    description="Normalizes projected Parquet text and provenance columns for embedding",
    category_color="#64b4ff",
)
class ParquetTextActor(AbstractOperator, CPUOperator):
    """Convert text-corpus Parquet rows into canonical graph rows.

    The Parquet reader remains a Ray Data source so it can push column
    projection into the scan. Use :meth:`projected_columns` when constructing
    that source. This actor then produces only the text and metadata required
    by an embedder and :class:`IngestVdbOperator`.
    """

    PRESERVE_PANDAS_OUTPUT: ClassVar[bool] = True

    def __init__(
        self,
        *,
        text_column: str = "text",
        url_column: str = "url",
        file_path_column: str = "file_path",
        id_column: str | None = "id",
        dump_column: str | None = "dump",
        drop_blank_text: bool = True,
    ) -> None:
        super().__init__(
            text_column=text_column,
            url_column=url_column,
            file_path_column=file_path_column,
            id_column=id_column,
            dump_column=dump_column,
            drop_blank_text=drop_blank_text,
        )

    def projected_columns(self) -> list[str]:
        """Return the columns that a Parquet source should read."""

        return list(
            dict.fromkeys(
                column
                for column in (
                    self.text_column,
                    self.url_column,
                    self.file_path_column,
                    self.id_column,
                    self.dump_column,
                )
                if column
            )
        )

    def preprocess(self, data: Any, **kwargs: Any) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            import pyarrow as pa

            if isinstance(data, (pa.Table, pa.RecordBatch)):
                # Projected Parquet tables can retain pandas metadata for
                # columns that were not read. Remove it before conversion so
                # stale extension dtypes cannot break an otherwise valid scan.
                data = data.replace_schema_metadata(None).to_pandas()
            elif hasattr(data, "to_pandas"):
                data = data.to_pandas()
            else:
                data = pd.DataFrame(data)

        required = {self.text_column, self.url_column, self.file_path_column}
        optional = {column for column in (self.id_column, self.dump_column) if column}
        missing = sorted((required | optional) - set(data.columns))
        if missing:
            raise ValueError(f"Parquet text input is missing projected column(s): {', '.join(missing)}")
        return data

    @staticmethod
    def _clean(value: Any) -> str:
        if value is None or value is pd.NA:
            return ""
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value).strip()

    def process(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for source in data.to_dict(orient="records"):
            text = self._clean(source.get(self.text_column))
            if self.drop_blank_text and not text:
                continue

            url = self._clean(source.get(self.url_column))
            file_path = self._clean(source.get(self.file_path_column))
            document_id = self._clean(source.get(self.id_column)) if self.id_column else ""
            dump = self._clean(source.get(self.dump_column)) if self.dump_column else ""
            stable_id = document_id or url or file_path

            source_metadata = {
                "source_id": url or stable_id,
                "source_name": url or stable_id,
                "url": url,
                "file_path": file_path,
            }
            if document_id:
                source_metadata["document_id"] = document_id
            if dump:
                source_metadata["dump"] = dump

            rows.append(
                {
                    "text": text,
                    "document_type": "text",
                    "_content_type": "text",
                    "metadata": {
                        "content_metadata": {"type": "text", "id": stable_id},
                        "source_metadata": source_metadata,
                    },
                }
            )

        return pd.DataFrame.from_records(
            rows,
            columns=["text", "document_type", "_content_type", "metadata"],
        )

    def postprocess(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        return data
