from __future__ import annotations

import ctypes
import logging

import pytest

from nemo_retriever.operators import gpu_operator
from nemo_retriever.operators import abstract_operator
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.gpu_operator import GPUOperator


@pytest.fixture(autouse=True)
def reset_configured_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_operator, "_cuda_blocking_sync_configured", False)


def test_unset_env_does_not_load_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NRL_CUDA_BLOCKING_SYNC", raising=False)
    monkeypatch.setattr(gpu_operator.ctypes, "CDLL", lambda *_args, **_kwargs: pytest.fail("unexpected CDLL call"))

    gpu_operator._configure_cuda_blocking_sync()


def test_enabled_env_sets_flags_once_for_each_visible_device(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []

    class FakeCuda:
        def cuInit(self, flags: int) -> int:
            calls.append(("init", flags))
            return 0

        def cuDeviceGetCount(self, count_ptr: object) -> int:
            ctypes.cast(count_ptr, ctypes.POINTER(ctypes.c_int)).contents.value = 2
            return 0

        def cuDeviceGet(self, device_ptr: object, ordinal: int) -> int:
            ctypes.cast(device_ptr, ctypes.POINTER(ctypes.c_int)).contents.value = ordinal + 10
            return 0

        def cuDevicePrimaryCtxSetFlags_v2(self, device: ctypes.c_int, flags: int) -> int:
            calls.append(("flags", device.value, flags))
            return 0

    monkeypatch.setenv("NRL_CUDA_BLOCKING_SYNC", "1")
    monkeypatch.setattr(gpu_operator.ctypes, "CDLL", lambda path: (calls.append(("load", path)), FakeCuda())[1])

    gpu_operator._configure_cuda_blocking_sync()
    gpu_operator._configure_cuda_blocking_sync()

    assert calls == [
        ("load", "libcuda.so.1"),
        ("init", 0),
        ("flags", 10, 4),
        ("flags", 11, 4),
    ]


def test_missing_libcuda_warns_and_continues(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv("NRL_CUDA_BLOCKING_SYNC", "1")

    def fail_load(_path: str) -> object:
        raise OSError("missing driver")

    monkeypatch.setattr(gpu_operator.ctypes, "CDLL", fail_load)
    with caplog.at_level(logging.WARNING, logger=gpu_operator.__name__):
        gpu_operator._configure_cuda_blocking_sync()

    assert "NRL_CUDA_BLOCKING_SYNC=1 ignored:" in caplog.text
    assert "missing driver" in caplog.text


def test_nonzero_cuinit_warns_and_skips_set_flags(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    set_flags_calls: list[tuple[object, ...]] = []

    class FakeCuda:
        def cuInit(self, flags: int) -> int:
            return 7

        def cuDeviceGetCount(self, count_ptr: object) -> int:
            pytest.fail("cuDeviceGetCount should not run after cuInit fails")

        def cuDevicePrimaryCtxSetFlags_v2(self, device: ctypes.c_int, flags: int) -> int:
            set_flags_calls.append((device, flags))
            return 0

    monkeypatch.setenv("NRL_CUDA_BLOCKING_SYNC", "1")
    monkeypatch.setattr(gpu_operator.ctypes, "CDLL", lambda _path: FakeCuda())
    with caplog.at_level(logging.WARNING, logger=gpu_operator.__name__):
        gpu_operator._configure_cuda_blocking_sync()

    assert "CUresult=7" in caplog.text
    assert set_flags_calls == []


def test_abstract_operator_configures_gpu_subclasses_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setenv("NRL_CUDA_BLOCKING_SYNC", "1")
    monkeypatch.setattr(abstract_operator, "_configure_cuda_blocking_sync", lambda: calls.append("configured"))

    class DummyGPUOperator(AbstractOperator, GPUOperator):
        def preprocess(self, data: object, **kwargs: object) -> object:
            return data

        def process(self, data: object, **kwargs: object) -> object:
            return data

        def postprocess(self, data: object, **kwargs: object) -> object:
            return data

    class DummyCPUOperator(AbstractOperator):
        def preprocess(self, data: object, **kwargs: object) -> object:
            return data

        def process(self, data: object, **kwargs: object) -> object:
            return data

        def postprocess(self, data: object, **kwargs: object) -> object:
            return data

    DummyGPUOperator()
    DummyCPUOperator()

    assert calls == ["configured"]
