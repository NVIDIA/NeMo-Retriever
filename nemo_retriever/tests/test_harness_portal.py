# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import sys
import types
from datetime import datetime, timedelta, timezone

from nemo_retriever.harness import history


def _iso(delta: timedelta) -> str:
    return (datetime.now(timezone.utc) + delta).isoformat()


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


def test_expired_runner_direct_work_poll_gets_no_job(tmp_path, monkeypatch):
    _install_fake_apscheduler(monkeypatch)
    from nemo_retriever.harness.portal.app import runner_get_work

    db_path = str(tmp_path / "history.db")
    monkeypatch.setenv("RETRIEVER_HARNESS_HISTORY_DB", db_path)
    runner = history.register_runner(
        {
            "name": "expired-runner",
            "hostname": "expired-host",
            "status": "online",
            "valid_until": _iso(timedelta(minutes=-1)),
        }
    )

    response = asyncio.run(runner_get_work(runner["id"]))

    assert response.status_code == 204
