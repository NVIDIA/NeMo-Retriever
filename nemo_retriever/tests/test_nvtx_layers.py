# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

from nemo_retriever.common import nvtx_layers


class _Domain:
    def __init__(self, events):
        self.events = events

    def get_event_attributes(self, message, color):
        return message

    def push_range(self, attrs):
        self.events.append(("push", attrs))

    def pop_range(self):
        self.events.append(("pop", None))

    def start_range(self, attrs):
        self.events.append(("start", attrs))
        return attrs

    def end_range(self, range_id):
        self.events.append(("end", range_id))


@pytest.fixture
def enabled(monkeypatch):
    events = []
    monkeypatch.setitem(sys.modules, "nvtx", SimpleNamespace(get_domain=lambda layer: _Domain(events)))
    monkeypatch.setattr(nvtx_layers, "ENABLED", True)
    monkeypatch.setattr(nvtx_layers, "_SPANS", {})
    monkeypatch.setattr(nvtx_layers, "_INSTALLED", False)
    return events


def test_disabled_span_and_install_do_nothing(monkeypatch):
    monkeypatch.setattr(nvtx_layers, "ENABLED", False)
    monkeypatch.setattr(nvtx_layers, "_INSTALLED", False)
    monkeypatch.setattr(nvtx_layers, "_patch", lambda *args: pytest.fail("patched while disabled"))
    assert nvtx_layers.span("NRL", "x") is nvtx_layers.span("vLLM", "y")
    nvtx_layers.install()
    assert nvtx_layers._INSTALLED is False


def test_span_closes_range_when_body_raises(enabled):
    with pytest.raises(ValueError):
        with nvtx_layers.span("NRL", "work"):
            raise ValueError("boom")
    assert enabled == [("push", "work"), ("pop", None)]


def test_wrapped_function_generator_and_coroutine_close_ranges(enabled):
    def fails():
        raise RuntimeError("boom")

    def gen():
        yield 1
        yield 2

    async def coro():
        return 3

    with pytest.raises(RuntimeError):
        nvtx_layers._wrap(fails, "torch", "f")()
    assert list(nvtx_layers._wrap(gen, "Ray", "g")()) == [1, 2]
    assert asyncio.run(nvtx_layers._wrap(coro, "vLLM", "c")()) == 3
    assert enabled[:2] == [("push", "f"), ("pop", None)]
    assert enabled[2:8] == [("push", "g"), ("pop", None)] * 3
    assert enabled[8:] == [("start", "c"), ("end", "c")]


def test_install_patches_targets_once_and_skips_missing(enabled, monkeypatch):
    module = ModuleType("nrl_fake_target")

    class Target:
        def run(self):
            return "ran"

    module.Target = Target
    monkeypatch.setitem(sys.modules, "nrl_fake_target", module)
    monkeypatch.setattr(nvtx_layers, "_MODEL_TARGETS", (("nrl_fake_target", "Target.run", "NRL", "run"),))
    monkeypatch.setattr(nvtx_layers, "_VLLM_TARGETS", (("nrl_missing_target", "X.y", "vLLM", "y"),))
    monkeypatch.setattr(nvtx_layers, "_RAY_TARGETS", ())
    monkeypatch.setattr(nvtx_layers, "_install_tensor_syncs", lambda: None)

    nvtx_layers.install()
    wrapped = Target.run
    nvtx_layers.install()

    assert Target.run is wrapped and wrapped.__nrl_nvtx__
    assert Target().run() == "ran"
    assert enabled == [("push", "run"), ("pop", None)]
