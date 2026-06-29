# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from nemo_retriever.harness.config import HarnessConfig
from nemo_retriever.harness.replay_command import (
    build_harness_run_command,
    build_service_replay_command,
    persist_replay_command,
    reconstruct_command_from_record,
)


def test_build_harness_run_command_includes_set_overrides() -> None:
    command = build_harness_run_command(
        "jp20_beir",
        output_dir="/tmp/out",
        run_id="run-1",
        mode="batch",
        overrides=["dataset.path=/data/jp20"],
        requirements=["recall_5>=0.8"],
    )
    assert "retriever harness run" in command
    assert "jp20_beir" in command
    assert "--mode batch" in command
    assert "--set dataset.path=/data/jp20" in command
    assert "--require 'recall_5>=0.8'" in command


def test_persist_replay_command_writes_command_txt(tmp_path: Path) -> None:
    meta = persist_replay_command(tmp_path, "retriever harness run jp20_smoke")
    command_path = tmp_path / "command.txt"
    assert command_path.is_file()
    assert command_path.read_text(encoding="utf-8").strip() == "retriever harness run jp20_smoke"
    assert meta["command_file"] == str(command_path.resolve())
    assert meta["replay_command"] == "retriever harness run jp20_smoke"


def test_reconstruct_command_from_record_prefers_replay_command() -> None:
    command = reconstruct_command_from_record(
        {
            "replay_command": "retriever harness run jp20_beir --output-dir /tmp/out",
            "test_config": {"dataset_label": "jp20"},
        }
    )
    assert command == "retriever harness run jp20_beir --output-dir /tmp/out"


def test_reconstruct_command_from_record_builds_service_command() -> None:
    command = reconstruct_command_from_record(
        {
            "artifact_dir": "/tmp/artifacts/run-1",
            "test_config": {
                "run_mode": "service",
                "service_url": "http://localhost:7670",
                "dataset_dir": "/data/jp20",
                "preset": "single_gpu",
                "dataset_label": "jp20",
            },
        }
    )
    assert command is not None
    assert "service-url http://localhost:7670" in command
    assert "--dataset /data/jp20" in command


def test_build_service_replay_command() -> None:
    cfg = HarnessConfig(
        dataset_dir="/data/jp20",
        dataset_label="jp20",
        preset="single_gpu",
        run_mode="service",
        service_url="http://localhost:7670",
    )
    command = build_service_replay_command(cfg)
    assert "portal service-mode job" in command
    assert "http://localhost:7670" in command
