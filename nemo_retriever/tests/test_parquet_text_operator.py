# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd
import pytest

from nemo_retriever.common.vdb.records import to_client_vdb_records
from nemo_retriever.operators.graph_ops.parquet_text_operator import ParquetTextActor


def test_parquet_text_actor_projects_only_retrieval_columns_and_builds_metadata() -> None:
    actor = ParquetTextActor()
    data = pd.DataFrame(
        [
            {
                "text": " useful text ",
                "url": "https://example.test/doc",
                "file_path": "s3://commoncrawl/example.warc.gz",
                "id": "<urn:uuid:example>",
                "dump": "CC-MAIN-2026-30",
                "embedding": [9.0, 9.0],
                "score": 4.2,
            },
            {
                "text": "   ",
                "url": "https://example.test/blank",
                "file_path": "s3://commoncrawl/blank.warc.gz",
                "id": "blank",
                "dump": "CC-MAIN-2026-30",
            },
        ]
    )

    assert actor.projected_columns() == ["text", "url", "file_path", "id", "dump"]
    result = actor.run(data)

    assert result.columns.tolist() == ["text", "document_type", "_content_type", "metadata"]
    assert result["text"].tolist() == ["useful text"]
    metadata = result.iloc[0]["metadata"]
    assert "content" not in metadata
    assert metadata["content_metadata"] == {"type": "text", "id": "<urn:uuid:example>"}
    row = result.iloc[0].to_dict()
    row["metadata"]["embedding"] = [0.1, 0.2]
    canonical = to_client_vdb_records(pd.DataFrame([row]))[0][0]
    assert canonical["metadata"]["content"] == "useful text"

    assert metadata["source_metadata"] == {
        "source_id": "https://example.test/doc",
        "source_name": "https://example.test/doc",
        "url": "https://example.test/doc",
        "file_path": "s3://commoncrawl/example.warc.gz",
        "document_id": "<urn:uuid:example>",
        "dump": "CC-MAIN-2026-30",
    }


def test_parquet_text_actor_rejects_unprojected_optional_metadata() -> None:
    actor = ParquetTextActor()
    with pytest.raises(ValueError, match=r"dump, id"):
        actor.run(pd.DataFrame([{"text": "x", "url": "u", "file_path": "p"}]))
