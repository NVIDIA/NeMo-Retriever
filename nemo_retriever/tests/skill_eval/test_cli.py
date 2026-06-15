# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from nemo_retriever.tools.skill_eval.cli import _resolve_workdir_root


def test_resolve_workdir_root_makes_relative_path_absolute(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)

    workdir_root = _resolve_workdir_root({"per_trial_workdir_root": "./tmp/skill_eval"})

    assert workdir_root == tmp_path / "tmp" / "skill_eval"
    assert workdir_root.is_absolute()


def test_resolve_workdir_root_preserves_absolute_path(tmp_path) -> None:
    configured = tmp_path / "skill_eval"

    workdir_root = _resolve_workdir_root({"per_trial_workdir_root": str(configured)})

    assert workdir_root == configured
    assert isinstance(workdir_root, Path)
