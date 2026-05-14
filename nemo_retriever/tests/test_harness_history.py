# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_retriever.harness import history


def test_managed_dataset_persists_ocr_lang(tmp_path):
    db_path = str(tmp_path / "history.db")
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    created = history.create_dataset(
        {
            "name": "ocr-lang-smoke",
            "path": str(dataset_dir),
            "evaluation_mode": "custom",
            "ocr_version": "v2",
            "ocr_lang": "english",
        },
        db_path,
    )

    assert created["ocr_lang"] == "english"

    updated = history.update_dataset(created["id"], {"ocr_lang": "multi"}, db_path)
    assert updated is not None
    assert updated["ocr_lang"] == "multi"
