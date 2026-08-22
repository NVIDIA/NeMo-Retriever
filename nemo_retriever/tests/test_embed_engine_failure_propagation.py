# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime propagation tests for local embed-engine lifecycle failures."""

from __future__ import annotations

from typing import Any

import httpx
import pandas as pd
import pytest

from nemo_retriever.models.embed_errors import (
    LocalEmbedderReturnedNothingError,
    LocalEmbedderRowsLostError,
)
from nemo_retriever.models.inference import runtime
from nemo_retriever.models.nim.error_reporter import drain_errors


class EngineDeadError(Exception):
    """Stand-in for ``vllm.v1.engine.exceptions.EngineDeadError``."""


class OutOfMemoryError(Exception):
    """Stand-in for ``torch.OutOfMemoryError``."""


class EngineGenerateError(Exception):
    """Stand-in for ``vllm.v1.engine.exceptions.EngineGenerateError``."""


ADMISSION_REFUSAL = (
    "Free memory on device cuda:0 (14.73/44.39 GiB) on startup is less than desired "
    "GPU memory utilization (0.45, 19.98 GiB). Decrease GPU memory utilization or "
    "reduce GPU memory used by other processes."
)
ENGINE_INIT_FAILED = "Engine core initialization failed. See root cause above. Failed core proc(s): {}"
OOM_IN_GELU = (
    "CUDA out of memory. Tried to allocate 380.00 MiB. GPU 0 has a total capacity of "
    "44.39 GiB of which 274.69 MiB is free."
)


@pytest.fixture(autouse=True)
def _clear_reported_errors():
    drain_errors()
    yield
    drain_errors()


def _batch(rows: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [f"page {index}" for index in range(rows)],
            "metadata": [
                {"content": f"page {index}", "content_metadata": {"page_number": index}, "source_metadata": {}}
                for index in range(rows)
            ],
        }
    )


class _TextModel:
    """Model-boundary double for the local runtime."""

    def __init__(self, result: Any) -> None:
        self.result = result

    def embed(self, _texts: Any, *, batch_size: int) -> Any:
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def _wrapped(outer: BaseException, cause: BaseException) -> BaseException:
    """Build ``outer`` raised from ``cause``, as vLLM re-raises engine errors."""
    try:
        try:
            raise cause
        except BaseException as inner:  # noqa: BLE001 - constructing a chain on purpose
            raise outer from inner
    except BaseException as chained:  # noqa: BLE001
        return chained


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(RuntimeError(ENGINE_INIT_FAILED), id="refused-at-startup"),
        pytest.param(ValueError(ADMISSION_REFUSAL), id="admission-gate-valueerror"),
        pytest.param(EngineDeadError("EngineCore encountered an issue"), id="dead-after-admission"),
        pytest.param(
            _wrapped(RuntimeError("Worker proc died unexpectedly"), EngineDeadError("EngineCore is dead")),
            id="engine-dead-wrapped-in-runtimeerror",
        ),
        pytest.param(
            LocalEmbedderReturnedNothingError(
                "Local embedder returned no embeddings for a batch of 2 input(s). "
                "The in-process engine produced nothing, so it is not serving."
            ),
            id="engine-answered-with-nothing",
        ),
        pytest.param(
            LocalEmbedderRowsLostError(lost=7, total=64, embedder="LlamaNemotronEmbedVL1BV2VLLMEmbedder"),
            id="engine-lost-part-of-the-batch",
        ),
    ],
)
def test_local_engine_failure_propagates(exc: BaseException) -> None:
    with pytest.raises(type(exc)):
        runtime.embed_text_main_text_embed(_batch(), model=_TextModel(exc), inference_batch_size=2)


def test_local_embedder_returning_nothing_is_raised_as_a_classified_failure() -> None:
    with pytest.raises(LocalEmbedderReturnedNothingError):
        runtime.embed_text_main_text_embed(_batch(2), model=_TextModel([]), inference_batch_size=2)


def test_partial_local_result_keeps_the_pre_existing_fallback() -> None:
    out_df = runtime.embed_text_main_text_embed(_batch(2), model=_TextModel([[1.0, 2.0]]), inference_batch_size=2)

    assert list(out_df["text_embeddings_1b_v2_has_embedding"]) == [False, False]
    assert [payload["embedding"] for payload in out_df["text_embeddings_1b_v2"]] == [[], []]


@pytest.mark.parametrize(
    "model",
    [pytest.param(None, id="model-nulled-by-the-operator"), pytest.param(object(), id="model-also-set")],
)
def test_endpoint_mode_absorbs_engine_lifecycle_failure(monkeypatch: pytest.MonkeyPatch, model: object) -> None:
    original_client = httpx.Client

    def client_factory(*_args: Any, **_kwargs: Any) -> httpx.Client:
        transport = httpx.MockTransport(lambda _request: httpx.Response(400, text=ENGINE_INIT_FAILED))
        return original_client(transport=transport)

    monkeypatch.setattr(httpx, "Client", client_factory)

    out_df = runtime.embed_text_main_text_embed(
        _batch(),
        model=model,
        embedding_endpoint="http://embed.example/v1",
        inference_batch_size=4,
    )

    assert list(out_df["text_embeddings_1b_v2_has_embedding"]) == [False] * 4
    assert [payload["embedding"] for payload in out_df["text_embeddings_1b_v2"]] == [[]] * 4


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(OutOfMemoryError(OOM_IN_GELU), id="bare-oom"),
        pytest.param(EngineGenerateError("generate() failed"), id="recoverable-generate-error"),
    ],
)
def test_recoverable_local_failure_keeps_the_pre_existing_fallback(exc: BaseException) -> None:
    out_df = runtime.embed_text_main_text_embed(_batch(), model=_TextModel(exc), inference_batch_size=2)

    assert list(out_df["text_embeddings_1b_v2_has_embedding"]) == [False] * 4
    assert [payload["embedding"] for payload in out_df["text_embeddings_1b_v2"]] == [[]] * 4
