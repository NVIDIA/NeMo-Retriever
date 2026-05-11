# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the bundled ``retriever-service.yaml`` shipped with the package."""

from __future__ import annotations

from pathlib import Path

from nemo_retriever.service.config import _bundled_yaml_path, load_config


def test_bundled_yaml_loads_cleanly():
    bundled = _bundled_yaml_path()
    assert bundled.is_file()
    cfg = load_config(config_path=str(bundled))
    assert cfg.processing.num_workers > 0
    assert cfg.processing.results_dir
    assert cfg.vector_store.lancedb_uri


def test_bundled_yaml_uses_cwd_relative_paths(tmp_path, monkeypatch):
    """`retriever service start` with no overrides must work in any cwd —
    the bundled YAML's writable-state defaults must be relative.
    """
    monkeypatch.chdir(tmp_path)  # avoid picking up a stray ./retriever-service.yaml
    cfg = load_config(config_path=None)

    assert not Path(
        cfg.processing.results_dir
    ).is_absolute(), f"bundled YAML pins absolute results_dir: {cfg.processing.results_dir!r}"
    assert not Path(
        cfg.vector_store.lancedb_uri
    ).is_absolute(), f"bundled YAML pins absolute lancedb_uri: {cfg.vector_store.lancedb_uri!r}"
