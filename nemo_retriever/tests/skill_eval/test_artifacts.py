# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from nemo_retriever.tools.skill_eval import artifacts


def test_session_summary_uses_skill_eval_owned_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(artifacts, "last_commit", lambda: "abc123")

    session_dir = artifacts.create_session_dir("skilleval", base_dir=str(tmp_path))
    summary_path = artifacts.write_session_summary(
        session_dir,
        [{"success": True, "metric": 0.75}],
        session_type="skill_eval",
        config_path="config.yaml",
        run_commit="run456",
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert session_dir.parent == tmp_path
    assert summary["run_commit"] == "run456"
    assert summary["latest_commit"] == "abc123"
    assert summary["all_passed"] is True
    assert summary["results"] == [{"success": True, "metric": 0.75}]
