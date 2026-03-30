# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MultiTypeExtractOperator grouping logic."""

from pathlib import Path

from nemo_retriever.graph import MultiTypeExtractOperator


def test_multi_type_grouping_against_repo_data_folder() -> None:
    extract_op = MultiTypeExtractOperator()
    data_folder = Path(__file__).resolve().parents[3] / "data"

    assert data_folder.is_dir(), f"Expected data directory at {data_folder}"

    grouped = extract_op.preprocess(str(data_folder))

    assert set(grouped) == {"pdf", "image", "text", "html", "audio", "video"}
    assert any(files for files in grouped.values()), f"No supported files were found in {data_folder}"
