# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_retriever.utils.hf_cache import collect_hf_runtime_env


def test_collect_hf_runtime_env_defaults_to_online(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "legacy-token")
    monkeypatch.setenv("HF_HOME", "/cache/home")
    monkeypatch.setenv("HF_HUB_CACHE", "/cache/hub")
    monkeypatch.setenv("TRANSFORMERS_CACHE", "/cache/transformers")
    monkeypatch.setenv("NEMO_RETRIEVER_HF_CACHE_DIR", "/cache/nemo")
    monkeypatch.setenv("HF_ENDPOINT", "https://hf.example")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/certs/ca.pem")
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-secret")
    monkeypatch.setenv("NGC_API_KEY", "ngc-secret")

    env = collect_hf_runtime_env()

    assert env["HF_HUB_OFFLINE"] == "0"
    assert env["HF_TOKEN"] == "secret-token"
    assert env["HUGGING_FACE_HUB_TOKEN"] == "legacy-token"
    assert env["HF_HOME"] == "/cache/home"
    assert env["HF_HUB_CACHE"] == "/cache/hub"
    assert env["TRANSFORMERS_CACHE"] == "/cache/transformers"
    assert env["NEMO_RETRIEVER_HF_CACHE_DIR"] == "/cache/nemo"
    assert env["HF_ENDPOINT"] == "https://hf.example"
    assert env["HTTPS_PROXY"] == "http://proxy.example:8080"
    assert env["REQUESTS_CA_BUNDLE"] == "/certs/ca.pem"
    assert "NVIDIA_API_KEY" not in env
    assert "NGC_API_KEY" not in env


def test_collect_hf_runtime_env_preserves_explicit_offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    env = collect_hf_runtime_env()

    assert env["HF_HUB_OFFLINE"] == "1"


def test_collect_hf_runtime_env_preserves_explicit_empty_values(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "")

    env = collect_hf_runtime_env()

    assert env["NO_PROXY"] == ""
