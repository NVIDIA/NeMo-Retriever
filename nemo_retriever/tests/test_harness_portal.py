# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import sys
import types

from nemo_retriever.harness import history


def _install_fake_apscheduler(monkeypatch):
    apscheduler = types.ModuleType("apscheduler")
    triggers = types.ModuleType("apscheduler.triggers")
    cron = types.ModuleType("apscheduler.triggers.cron")

    class CronTrigger:
        pass

    cron.CronTrigger = CronTrigger
    monkeypatch.setitem(sys.modules, "apscheduler", apscheduler)
    monkeypatch.setitem(sys.modules, "apscheduler.triggers", triggers)
    monkeypatch.setitem(sys.modules, "apscheduler.triggers.cron", cron)
    monkeypatch.setitem(sys.modules, "nemo_retriever.harness.scheduler", types.ModuleType("scheduler"))


def test_update_managed_dataset_can_clear_ocr_lang_when_switching_to_v1(tmp_path, monkeypatch):
    _install_fake_apscheduler(monkeypatch)
    from nemo_retriever.harness.portal.app import DatasetUpdateRequest, update_managed_dataset

    db_path = str(tmp_path / "history.db")
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    monkeypatch.setenv("RETRIEVER_HARNESS_HISTORY_DB", db_path)

    created = history.create_dataset(
        {
            "name": "ocr-lang-smoke",
            "path": str(dataset_dir),
            "evaluation_mode": "custom",
            "ocr_version": "v2",
            "ocr_lang": "english",
        }
    )

    updated = asyncio.run(
        update_managed_dataset(
            created["id"],
            DatasetUpdateRequest(ocr_version="v1", ocr_lang=None),
        )
    )

    assert updated["ocr_version"] == "v1"
    assert updated["ocr_lang"] is None
