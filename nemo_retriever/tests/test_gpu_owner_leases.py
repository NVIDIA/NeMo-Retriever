# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
import gc
import weakref
from concurrent.futures import ThreadPoolExecutor
import pytest
from nemo_retriever.operators.extract.pdf.gpu_owner_leases import OwnerLeaseBook


class Reference:
    pass


def test_source_survives_all_external_borrowers_until_final_ack():
    book = OwnerLeaseBook(100)
    token = book.reserve(100, 3)
    source = Reference()
    weak = weakref.ref(source)
    book.commit(token, source)
    del source
    gc.collect()
    assert weak() is not None
    assert book.acknowledge(token, [2, 0, 2])
    gc.collect()
    assert weak() is not None
    assert book.reserve(1, 1) is None
    assert book.acknowledge(token, [1])
    gc.collect()
    assert weak() is None
    assert not book.acknowledge(token, [1])
    assert book.status()["active_bytes"] == 0


def test_admission_is_atomic_without_blocking_consumer_ack_threads():
    book = OwnerLeaseBook(32)
    with ThreadPoolExecutor(max_workers=32) as pool:
        tokens = list(pool.map(lambda _: book.reserve(1, 1), range(100)))
    admitted = [t for t in tokens if t is not None]
    assert len(admitted) == 32
    for token in admitted:
        book.commit(token, Reference())
    with ThreadPoolExecutor(max_workers=32) as pool:
        assert all(pool.map(lambda token: book.acknowledge(token, [0]), admitted))
    assert book.status()["peak_bytes"] == 32
    assert book.status()["active_bytes"] == 0
    assert book.reserve(32, 16) is not None


def test_failed_render_can_cancel_but_published_source_cannot():
    book = OwnerLeaseBook(10)
    token = book.reserve(10, 2)
    book.cancel(token)
    token = book.reserve(10, 2)
    book.commit(token, Reference())
    with pytest.raises(RuntimeError):
        book.cancel(token)
    assert book.status()["active_bytes"] == 10
    book.acknowledge(token, [0, 1])


def test_invalid_ack_cannot_partially_release_live_pages():
    book = OwnerLeaseBook(10)
    token = book.reserve(10, 3)
    book.commit(token, Reference())
    with pytest.raises(ValueError):
        book.acknowledge(token, [0, 10])
    assert book.status()["pending_pages"] == 3
    book.acknowledge(token, [2])
    book.acknowledge(token, [2, 0])
    assert book.status()["pending_pages"] == 1
    book.acknowledge(token, [1])
    assert book.status()["active_bytes"] == 0


def test_oversize_batch_is_rejected_before_gpu_allocation():
    book = OwnerLeaseBook(10)
    with pytest.raises(ValueError):
        book.reserve(11, 1)
    assert book.status()["active_bytes"] == 0


def test_producer_keeps_publication_until_consumer_ack(monkeypatch):
    import threading
    from types import SimpleNamespace
    import ray
    from nemo_retriever.operators.extract.pdf.gpu_transport import _GpuProducer

    producer = _GpuProducer.__new__(_GpuProducer)
    producer._owner_leases = OwnerLeaseBook(3 * 1024**2)
    producer._ready_transport = False
    producer._stats_lock = threading.Lock()
    producer.pages = 0
    producer.renderer = SimpleNamespace(
        replay=lambda scene: (SimpleNamespace(is_cuda=True), {"cpu_raster_calls": 0, "readback_ms": 0})
    )
    refs = []

    def put(*args, **kwargs):
        ref = Reference()
        refs.append(weakref.ref(ref))
        return ref

    monkeypatch.setattr(ray, "put", put)
    scene = SimpleNamespace(height=1, width=1)
    metadata = producer.render([scene])
    token = metadata["owner_token"]
    del metadata
    gc.collect()
    assert refs[0]() is not None
    assert producer.render([scene]) == {"busy": True, "owner_progress": 0}
    producer.acknowledge_images({token: [0]})
    gc.collect()
    assert refs[0]() is None
    assert producer._owner_leases.status()["active_batches"] == 0


def test_partial_submit_failure_drains_reader_before_returning_credit():
    import threading
    import time
    from types import SimpleNamespace
    from nemo_retriever.operators.extract.pdf.gpu_transport import _GpuProducer

    producer = _GpuProducer.__new__(_GpuProducer)
    producer._owner_leases = OwnerLeaseBook(6 * 1024**2)
    finished = threading.Event()

    def reader(handle):
        time.sleep(0.02)
        assert producer._owner_leases.status()["active_bytes"] == 6 * 1024**2
        finished.set()

    producer._render_shared_page = reader
    with ThreadPoolExecutor(max_workers=1) as pool:

        class FailSecond:
            calls = 0

            def submit(self, fn, handle):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("executor stopped")
                return pool.submit(fn, handle)

        producer._pages = FailSecond()
        with pytest.raises(RuntimeError, match="executor stopped"):
            producer.render_handles([SimpleNamespace(height=1, width=1)] * 2)
        assert finished.is_set()
        assert producer._owner_leases.status()["active_bytes"] == 0


def test_final_ocr_ack_follows_gpu_scope_and_skips_failed_batches(monkeypatch):
    from contextlib import contextmanager
    import pandas as pd
    from nemo_retriever.operators.extract.ocr import gpu_ocr
    from nemo_retriever.operators.extract.pdf import gpu_transport

    events = []

    @contextmanager
    def scope(**kwargs):
        events.append("enter")
        yield
        events.append("gpu_finished")

    monkeypatch.setattr(gpu_transport, "gpu_image_scope", scope)
    monkeypatch.setattr(gpu_transport, "acknowledge_gpu_images", lambda frame: events.append("ack"))
    operator = gpu_ocr.OCRActor.__new__(gpu_ocr.OCRActor)
    operator._cuda_rdt = True
    operator._model = operator._remote_retry = operator._nim_client = None
    operator.ocr_kwargs = {}
    frame = pd.DataFrame({"page_image": [{"orig_shape_hw": (1, 1)}], "ocr": [None]})
    monkeypatch.setattr(gpu_ocr, "ocr_page_elements", lambda *a, **k: frame)
    result = operator.process(frame)
    assert events == ["enter", "gpu_finished", "ack"]
    assert result.page_image.iloc[0]["encoding"] == "gpu_consumed"
    events.clear()
    frame["ocr"] = [{"error": {"message": "model failed"}}]
    with pytest.raises(gpu_transport.GpuTransportError, match="model failed"):
        operator.process(frame)
    assert events == ["enter", "gpu_finished"]


def test_batch_charge_rounds_each_image_and_rejects_before_render():
    from types import SimpleNamespace
    from nemo_retriever.operators.extract.pdf.gpu_transport import _GpuProducer

    producer = _GpuProducer.__new__(_GpuProducer)
    slot = 3 * 1024**2
    producer._owner_leases = OwnerLeaseBook(slot * 2)
    with pytest.raises(ValueError, match="exceeds owner byte limit"):
        producer.render_handles([SimpleNamespace(height=1, width=1), SimpleNamespace(height=1025, width=1024)])
    assert producer._owner_leases.status()["active_bytes"] == 0


def test_credit_timeout_stops_instead_of_rendering_without_ownership(monkeypatch):
    import time
    import ray
    from types import SimpleNamespace
    from nemo_retriever.operators.extract.pdf.gpu_transport import _render_with_credit, GpuTransportError

    calls = []
    producer = SimpleNamespace(
        render_handles=SimpleNamespace(
            remote=lambda scenes, **kwargs: calls.append((scenes, kwargs["request_id"])) or object()
        )
    )
    clock = iter([0, 121])
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(ray, "get", lambda result: {"busy": True})
    with pytest.raises(GpuTransportError, match="capacity remained exhausted"):
        _render_with_credit(producer, "render_handles", ["owned_scene"])
    assert [item[0] for item in calls] == [["owned_scene"], ["owned_scene"]]
    assert calls[0][1] == calls[1][1]


def test_successful_executor_exit_requires_owner_drain_without_masking_original_failure(monkeypatch):
    import ray
    from types import SimpleNamespace
    from nemo_retriever.graph.executor import RayDataExecutor
    from nemo_retriever.operators.extract.pdf.gpu_transport import GpuTransportError

    monkeypatch.delenv("CUDA_PDF_RDT_AUDIT_PATH", raising=False)
    pool = SimpleNamespace(close=SimpleNamespace(remote=lambda: "summary"))
    killed = []
    monkeypatch.setattr(ray, "get", lambda ref, **kwargs: [{"owner_leases": {"active_batches": 1}}])
    monkeypatch.setattr(ray, "kill", lambda actor: killed.append(actor))
    executor = RayDataExecutor.__new__(RayDataExecutor)
    executor._cuda_rdt_pool = pool
    executor._ingest_graph = lambda *args, **kwargs: "complete"
    with pytest.raises(GpuTransportError, match="ownership did not drain"):
        executor._ingest(None, return_results=False)
    assert killed == [pool]
    executor._cuda_rdt_pool = pool

    def failed(*args, **kwargs):
        raise ValueError("original model failure")

    executor._ingest_graph = failed
    with pytest.raises(ValueError, match="original model failure"):
        executor._ingest(None, return_results=False)
    assert killed == [pool, pool]


def test_waiting_requests_cannot_be_overtaken_when_credits_return():
    book = OwnerLeaseBook(10)
    current = book.reserve(10, 2)
    book.commit(current, Reference())
    assert book.reserve(10, 1, "first") is None
    assert book.reserve(1, 1, "second") is None
    assert book.reserve(1, 1, "third") is None
    book.acknowledge(current, [0, 1])
    assert book.reserve(1, 1, "third") is None
    assert book.reserve(1, 1) is None
    admitted = book.reserve(10, 1, "first")
    assert admitted is not None
    book.commit(admitted, Reference())
    book.acknowledge(admitted, [0])
    assert book.reserve(1, 1, "third") is None
    assert book.reserve(1, 1, "second") is not None
    assert book.reserve(1, 1, "third") is not None
    assert book.status()["queued_requests"] == 0


def test_abandoned_head_can_be_cancelled_without_releasing_published_images():
    book = OwnerLeaseBook(10)
    current = book.reserve(10, 1)
    book.commit(current, Reference())
    assert book.reserve(10, 1, "abandoned") is None
    assert book.reserve(10, 1, "next") is None
    assert book.cancel_request("abandoned")
    assert not book.cancel_request("abandoned")
    assert book.status()["active_bytes"] == 10
    book.acknowledge(current, [0])
    assert book.reserve(10, 1, "next") is not None


def test_pending_queue_is_bounded_and_retries_cannot_change_batch():
    book = OwnerLeaseBook(10)
    current = book.reserve(10, 1)
    book.commit(current, Reference())
    assert book.reserve(10, 1, "first") is None
    with pytest.raises(ValueError, match="changed its batch"):
        book.reserve(1, 1, "first")
    for i in range(1023):
        assert book.reserve(1, 1, str(i)) is None
    with pytest.raises(RuntimeError, match="queue limit"):
        book.reserve(1, 1, "overflow")
    assert book.status()["queued_requests"] == 1024
    assert book.status()["active_bytes"] == 10


def test_duplicate_acks_do_not_count_as_progress():
    book = OwnerLeaseBook(10)
    current = book.reserve(10, 2)
    book.commit(current, Reference())
    assert book.admission_progress() == 0
    book.acknowledge(current, [0])
    assert book.admission_progress() == 1
    book.acknowledge(current, [0, 0])
    assert book.admission_progress() == 1
    book.acknowledge(current, [1])
    assert book.admission_progress() == 2


def test_progress_watchdog_allows_slow_fair_queue_and_cancels_on_stall(monkeypatch):
    import time
    import ray
    from types import SimpleNamespace
    from nemo_retriever.operators.extract.pdf.gpu_transport import _render_with_credit, GpuTransportError

    calls, cancellations = [], []
    producer = SimpleNamespace(
        render_handles=SimpleNamespace(remote=lambda scenes, **kwargs: calls.append(kwargs["request_id"]) or "render"),
        cancel_admission=SimpleNamespace(remote=lambda request: cancellations.append(request) or "cancel"),
    )
    clock = iter([0, 119, 238])
    replies = iter([{"busy": True, "owner_progress": value} for value in [0, 1, 2]] + [{"reference": "complete"}])
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(ray, "get", lambda result, **kwargs: next(replies) if result == "render" else True)
    assert _render_with_credit(producer, "render_handles", ["scene"]) == {"reference": "complete"}
    assert len(set(calls)) == 1 and not cancellations
    clock = iter([0, 121])
    replies = iter([{"busy": True, "owner_progress": 0}] * 2)
    with pytest.raises(GpuTransportError, match="without final-consumer progress"):
        _render_with_credit(producer, "render_handles", ["scene"])
    assert cancellations == [calls[-1]]
