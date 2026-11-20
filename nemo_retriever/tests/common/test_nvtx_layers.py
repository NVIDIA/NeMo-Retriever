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
    monkeypatch.setattr(nvtx_layers, "FULL", False)
    monkeypatch.setattr(nvtx_layers, "_SPANS", {})
    monkeypatch.setattr(nvtx_layers, "_INSTALLED", set())
    return events


def test_disabled_span_and_install_do_nothing(monkeypatch):
    monkeypatch.setattr(nvtx_layers, "ENABLED", False)
    monkeypatch.setattr(nvtx_layers, "_INSTALLED", set())
    monkeypatch.setattr(nvtx_layers, "_patch", lambda *args: pytest.fail("patched while disabled"))
    assert nvtx_layers.span("NRL", "x") is nvtx_layers.span("vLLM", "y")
    nvtx_layers.install("vllm")
    assert nvtx_layers._INSTALLED == set()


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
    monkeypatch.setattr(
        nvtx_layers,
        "_TARGETS",
        {
            "fake": (("nrl_fake_target", "Target.run", "NRL", "run"),),
            "vllm": (("nrl_missing_target", "X.y", "vLLM", "y"),),
        },
    )
    monkeypatch.setattr(nvtx_layers, "_FULL_TARGETS", {})

    nvtx_layers.install("fake", "vllm")
    wrapped = Target.run
    nvtx_layers.install("fake")

    assert Target.run is wrapped and wrapped.__nrl_nvtx__
    assert Target().run() == "ran"
    assert enabled == [("push", "run"), ("pop", None)]


@pytest.mark.parametrize("full", [False, True])
def test_full_level_adds_sync_and_serialization_targets(enabled, monkeypatch, full):
    patched, syncs = [], []
    monkeypatch.setattr(nvtx_layers, "FULL", full)
    monkeypatch.setattr(nvtx_layers, "_patch", lambda module, *args: patched.append(module))
    monkeypatch.setattr(nvtx_layers, "_install_tensor_syncs", lambda: syncs.append(True))
    monkeypatch.setattr(nvtx_layers, "_TARGETS", {"ocr": (("ocr_mod", "A.b", "NRL", "b"),)})
    monkeypatch.setattr(
        nvtx_layers,
        "_FULL_TARGETS",
        {"torch": (("torch_mod", "c", "torch", "c"),), "ocr": (("ocr_full_mod", "A.d", "NRL", "d"),)},
    )

    nvtx_layers.install("ocr")

    assert patched == (["ocr_mod", "ocr_full_mod", "torch_mod"] if full else ["ocr_mod"])
    assert syncs == ([True] if full else [])


def test_wrapped_async_generator_closes_range_on_completion_close_and_cancel(enabled):
    async def agen():
        yield 1
        yield 2

    wrapped = nvtx_layers._wrap(agen, "vLLM", "a")

    async def consume():
        return [item async for item in wrapped()]

    async def close_early():
        it = wrapped()
        assert await it.__anext__() == 1
        await it.aclose()

    async def cancel():
        started = asyncio.Event()

        async def slow():
            yield 1
            started.set()
            await asyncio.sleep(10)
            yield 2

        async def drain():
            async for _ in nvtx_layers._wrap(slow, "vLLM", "s")():
                pass

        task = asyncio.ensure_future(drain())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert asyncio.run(consume()) == [1, 2]
    asyncio.run(close_early())
    asyncio.run(cancel())
    assert enabled == [("start", "a"), ("end", "a")] * 2 + [("start", "s"), ("end", "s")]


def test_tensor_to_check_reads_devices_without_formatting_tensors():
    torch = pytest.importorskip("torch")

    class _NoFormat(torch.Tensor):
        def __format__(self, spec):
            pytest.fail("formatted a tensor argument")

        __repr__ = __str__ = __format__

    other = torch.zeros(1).as_subclass(_NoFormat)
    assert nvtx_layers._to_blocks_on_cpu((other,), {})
    assert nvtx_layers._to_blocks_on_cpu(("cpu",), {})
    assert nvtx_layers._to_blocks_on_cpu((torch.device("cpu"),), {})
    assert nvtx_layers._to_blocks_on_cpu((), {"device": "cpu"})
    assert not nvtx_layers._to_blocks_on_cpu(("cuda:0",), {})
    assert not nvtx_layers._to_blocks_on_cpu((torch.float16,), {})
    assert not nvtx_layers._to_blocks_on_cpu((), {})


def test_tensor_sync_wrappers_forward_results(enabled, monkeypatch):
    torch = pytest.importorskip("torch")
    for name in ("item", "tolist", "cpu", "numpy", "to", "__bool__", "__int__", "__float__"):
        monkeypatch.setattr(torch.Tensor, name, getattr(torch.Tensor, name))
    nvtx_layers._install_tensor_syncs()

    tensor = torch.tensor([1.5, 2.5])
    assert torch.Tensor.to.__nrl_nvtx__
    assert tensor.tolist() == [1.5, 2.5]
    assert tensor[0].item() == 1.5
    assert float(tensor[1]) == 2.5
    assert tensor.cpu().numpy().tolist() == [1.5, 2.5]
    assert tensor.to(torch.float64).dtype == torch.float64
    assert tensor.to(torch.zeros(1, dtype=torch.int64)).dtype == torch.int64
    # CPU tensors never block on the GPU, so no host-sync range opens.
    assert enabled == []
