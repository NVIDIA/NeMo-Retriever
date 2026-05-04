# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_retriever.utils.remote_auth import collect_remote_auth_runtime_env


def test_collect_remote_auth_runtime_env_preserves_legacy_nvidia_api_key(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-secret")
    monkeypatch.setenv("NGC_API_KEY", "ngc-secret")

    env = collect_remote_auth_runtime_env()

    assert env == {"NVIDIA_API_KEY": "nv-secret"}


def test_collect_remote_auth_runtime_env_preserves_empty_values(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "")

    env = collect_remote_auth_runtime_env()

    assert env == {"NVIDIA_API_KEY": ""}
