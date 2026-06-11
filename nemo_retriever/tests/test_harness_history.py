# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
import types
from datetime import datetime, timedelta, timezone

from nemo_retriever.harness import history


def _iso(delta: timedelta) -> str:
    return (datetime.now(timezone.utc) + delta).isoformat()


def _install_fake_apscheduler(monkeypatch):
    apscheduler = types.ModuleType("apscheduler")
    schedulers = types.ModuleType("apscheduler.schedulers")
    background = types.ModuleType("apscheduler.schedulers.background")
    triggers = types.ModuleType("apscheduler.triggers")
    cron = types.ModuleType("apscheduler.triggers.cron")
    interval = types.ModuleType("apscheduler.triggers.interval")

    class BackgroundScheduler:
        pass

    class CronTrigger:
        pass

    class IntervalTrigger:
        pass

    background.BackgroundScheduler = BackgroundScheduler
    cron.CronTrigger = CronTrigger
    interval.IntervalTrigger = IntervalTrigger
    monkeypatch.setitem(sys.modules, "apscheduler", apscheduler)
    monkeypatch.setitem(sys.modules, "apscheduler.schedulers", schedulers)
    monkeypatch.setitem(sys.modules, "apscheduler.schedulers.background", background)
    monkeypatch.setitem(sys.modules, "apscheduler.triggers", triggers)
    monkeypatch.setitem(sys.modules, "apscheduler.triggers.cron", cron)
    monkeypatch.setitem(sys.modules, "apscheduler.triggers.interval", interval)


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


def test_expired_runner_heartbeat_stays_offline(tmp_path):
    db_path = str(tmp_path / "history.db")
    runner = history.register_runner(
        {
            "name": "expired-runner",
            "hostname": "expired-host",
            "status": "online",
            "valid_until": _iso(timedelta(minutes=-1)),
        },
        db_path,
    )

    status = history.heartbeat_runner(runner["id"], db_path)

    assert status == "offline"
    refreshed = history.get_runner_by_id(runner["id"], db_path)
    assert refreshed is not None
    assert refreshed["status"] == "offline"
    assert refreshed["valid_until"] == runner["valid_until"]


def test_scheduler_skips_expired_online_runner(monkeypatch):
    _install_fake_apscheduler(monkeypatch)
    scheduler = importlib.import_module("nemo_retriever.harness.scheduler")
    scheduler._round_robin_index = 0
    monkeypatch.setattr(
        history,
        "get_runners",
        lambda: [
            {
                "id": 1,
                "name": "expired",
                "status": "online",
                "gpu_count": 8,
                "valid_until": _iso(timedelta(minutes=-1)),
            },
            {
                "id": 2,
                "name": "fresh",
                "status": "online",
                "gpu_count": 8,
                "valid_until": _iso(timedelta(hours=1)),
            },
        ],
    )

    runner = scheduler.match_runner(min_gpu_count=1)

    assert runner is not None
    assert runner["name"] == "fresh"
