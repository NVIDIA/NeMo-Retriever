from __future__ import annotations

import asyncio
from types import SimpleNamespace

from nemo_retriever.operators.embed.continuous_batch import (
    AsyncVllmEmbeddingEngine,
    ContinuousBatchEngineRouter,
    _RayAsyncEngineProxy,
)


def test_router_balances_and_releases_weight() -> None:
    router = ContinuousBatchEngineRouter(3)

    assert router.reserve(10) == 0
    assert router.reserve(4) == 1
    assert router.reserve(6) == 2
    assert router.loads() == [10, 4, 6]

    router.release(0, 10)
    assert router.reserve(3) == 0
    assert router.loads() == [3, 4, 6]


def test_router_rotates_across_equal_load_engines() -> None:
    router = ContinuousBatchEngineRouter(3)

    selections = []
    for _ in range(6):
        index = router.reserve(1)
        selections.append(index)
        router.release(index, 1)

    assert selections == [0, 1, 2, 0, 1, 2]


def test_async_engine_gathers_pooling_outputs() -> None:
    class FakeEngine:
        async def encode(self, prompt, pooling_params, request_id):
            del pooling_params
            value = float(len(prompt))
            yield SimpleNamespace(outputs=SimpleNamespace(data=[value, value + 1]), request_id=request_id)

    actor = AsyncVllmEmbeddingEngine.__new__(AsyncVllmEmbeddingEngine)
    actor._engine = FakeEngine()
    actor._engine_index = 3
    actor._dimensions = 2
    actor._request_counter = 0
    actor._request_slots = asyncio.Semaphore(2)

    result = asyncio.run(actor.embed(["a", "abcd"]))

    assert result == [[1.0, 2.0], [4.0, 5.0]]
    assert actor._request_counter == 2


def test_proxy_reserves_least_loaded_engine_and_releases(monkeypatch) -> None:
    class RemoteMethod:
        def __init__(self, result=None):
            self.result = result
            self.calls = []

        def remote(self, *args):
            self.calls.append(args)
            return self.result

    reserve = RemoteMethod(1)
    release = RemoteMethod(None)
    router = SimpleNamespace(reserve=reserve, release=release)
    engines = [
        SimpleNamespace(embed=RemoteMethod([[0.0]])),
        SimpleNamespace(embed=RemoteMethod([[1.0], [2.0]])),
    ]
    monkeypatch.setattr("ray.get", lambda value: value)

    result = _RayAsyncEngineProxy(engines, router).embed(["a", "bc"])

    assert result == [[1.0], [2.0]]
    assert reserve.calls == [(3,)]
    assert engines[1].embed.calls == [(["a", "bc"],)]
    assert release.calls == [(1, 3)]
