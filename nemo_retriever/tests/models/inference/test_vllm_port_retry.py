# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
import os
import socket
import time
from types import SimpleNamespace

import pytest

from nemo_retriever.models.inference import vllm as vllm_mod


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.delenv("VLLM_PORT", raising=False)
    monkeypatch.setattr(vllm_mod, "time", SimpleNamespace(sleep=lambda _s: None))


def _occupy(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", port))
    sock.listen()
    return sock


def test_retries_with_new_port_when_port_was_taken():
    ports, held = [], []

    def fake_llm(**_kwargs):
        ports.append(int(os.environ["VLLM_PORT"]))
        if len(ports) == 1:
            held.append(_occupy(ports[0]))  # another engine won the race for this port
            raise RuntimeError("Engine core initialization failed. See root cause above.")
        return "llm"

    try:
        assert vllm_mod._build_llm_with_port_retry(fake_llm, {}) == "llm"
    finally:
        for sock in held:
            sock.close()
    assert len(ports) == 2 and ports[0] != ports[1]
    assert "VLLM_PORT" not in os.environ


def test_other_failures_are_not_retried():
    calls = []

    def fake_llm(**_kwargs):
        calls.append(1)
        raise RuntimeError("Engine core initialization failed. See root cause above.")

    with pytest.raises(RuntimeError):
        vllm_mod._build_llm_with_port_retry(fake_llm, {})
    assert len(calls) == 1


def test_non_runtime_errors_propagate_without_retry():
    calls, held = [], []

    def fake_llm(**_kwargs):
        calls.append(1)
        held.append(_occupy(int(os.environ["VLLM_PORT"])))
        raise ValueError("bad engine args")

    try:
        with pytest.raises(ValueError):
            vllm_mod._build_llm_with_port_retry(fake_llm, {})
    finally:
        for sock in held:
            sock.close()
    assert len(calls) == 1
    assert "VLLM_PORT" not in os.environ


def test_gives_up_after_bounded_attempts():
    calls = []

    class DistNetworkError(RuntimeError):
        pass

    def fake_llm(**_kwargs):
        calls.append(1)
        raise DistNetworkError("EADDRINUSE")

    with pytest.raises(DistNetworkError):
        vllm_mod._build_llm_with_port_retry(fake_llm, {})
    assert len(calls) == vllm_mod.VLLM_BUILD_ATTEMPTS


def test_user_pinned_port_is_left_alone(monkeypatch):
    monkeypatch.setenv("VLLM_PORT", "29555")
    seen = []
    assert vllm_mod._build_llm_with_port_retry(lambda **_k: seen.append(os.environ["VLLM_PORT"]) or "ok", {}) == "ok"
    assert seen == ["29555"]


def test_cleanup_only_stops_engine_processes_from_the_failed_attempt():
    healthy = multiprocessing.Process(target=time.sleep, args=(60,), name="EngineCore_healthy", daemon=True)
    healthy.start()
    failed, held = [], []

    def fake_llm(**_kwargs):
        if not failed:
            proc = multiprocessing.Process(target=time.sleep, args=(60,), name="EngineCore_failed", daemon=True)
            proc.start()
            failed.append(proc)
            held.append(_occupy(int(os.environ["VLLM_PORT"])))
            raise RuntimeError("Engine core initialization failed. See root cause above.")
        return "llm"

    try:
        assert vllm_mod._build_llm_with_port_retry(fake_llm, {}) == "llm"
        assert not failed[0].is_alive()
        assert healthy.is_alive()
    finally:
        for sock in held:
            sock.close()
        for proc in [healthy, *failed]:
            proc.terminate()
            proc.join()
