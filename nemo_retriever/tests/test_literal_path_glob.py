# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from nemo_retriever.common.input_files import expand_input_file_patterns
from nemo_retriever.ingest.plan import expand_ingest_documents
from nemo_retriever.ingest.service import expand_service_file_patterns


def test_service_keeps_existing_file_with_glob_chars_in_name(tmp_path: Path) -> None:
    f = tmp_path / "Invoice [2024].pdf"
    f.write_bytes(b"%PDF-1.4")
    assert expand_service_file_patterns([str(f)]) == [str(f)]


def test_input_patterns_keep_existing_file_with_glob_chars_in_name(tmp_path: Path) -> None:
    f = tmp_path / "Invoice [2024].pdf"
    f.write_bytes(b"%PDF-1.4")
    assert expand_input_file_patterns([str(f)]) == [str(f)]


def test_directory_with_bracketed_and_plain_names_is_not_collapsed(tmp_path: Path) -> None:
    plain = tmp_path / "report1.txt"
    bracketed = tmp_path / "report[1].txt"
    plain.write_text("a")
    bracketed.write_text("b")
    result = expand_ingest_documents([str(tmp_path)])
    assert sorted(result) == sorted([str(plain), str(bracketed)])


def test_real_glob_pattern_still_expands(tmp_path: Path) -> None:
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4")
    pattern = str(tmp_path / "*.pdf")
    assert expand_service_file_patterns([pattern]) == [str(tmp_path / "a.pdf"), str(tmp_path / "b.pdf")]
    assert expand_input_file_patterns([pattern]) == [str(tmp_path / "a.pdf"), str(tmp_path / "b.pdf")]
