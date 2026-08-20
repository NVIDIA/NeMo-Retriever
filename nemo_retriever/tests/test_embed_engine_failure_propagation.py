# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime propagation tests for local embed-engine lifecycle failures."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from nemo_retriever.models.embed_errors import (
    LocalEmbedderReturnedNothingError,
    LocalEmbedderRowsLostError,
)
from nemo_retriever.models.inference import main_text_embed, runtime
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


def _raise(exc: BaseException):
    def _fail(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        raise exc

    return _fail


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
def test_local_engine_failure_propagates(monkeypatch: pytest.MonkeyPatch, exc: BaseException) -> None:
    monkeypatch.setattr(runtime, "_embed_group", _raise(exc))

    with pytest.raises(type(exc)):
        runtime.embed_text_main_text_embed(_batch(), model=object(), inference_batch_size=2)


def test_local_embedder_returning_nothing_is_raised_as_a_classified_failure() -> None:
    with pytest.raises(LocalEmbedderReturnedNothingError):
        main_text_embed._callable_runner([["page one", "page two"]], embedder=lambda _texts: [], batch_size=2)

    assert runtime._is_engine_lifecycle_failure(LocalEmbedderReturnedNothingError("no vectors"))


def test_partial_local_result_is_a_plain_value_error() -> None:
    with pytest.raises(ValueError) as excinfo:
        main_text_embed._callable_runner([["page one", "page two"]], embedder=lambda _texts: [[1.0, 2.0]], batch_size=2)

    assert not isinstance(excinfo.value, LocalEmbedderReturnedNothingError)
    assert not runtime._is_engine_lifecycle_failure(excinfo.value)


@pytest.mark.parametrize(
    "model",
    [pytest.param(None, id="model-nulled-by-the-operator"), pytest.param(object(), id="model-also-set")],
)
def test_endpoint_mode_absorbs_engine_lifecycle_failure(monkeypatch: pytest.MonkeyPatch, model: object) -> None:
    monkeypatch.setattr(runtime, "_embed_group", _raise(RuntimeError(ENGINE_INIT_FAILED)))

    out_df = runtime.embed_text_main_text_embed(
        _batch(),
        model=model,
        embedding_endpoint="http://embed.example/v1",
        inference_batch_size=2,
    )

    assert list(out_df["text_embeddings_1b_v2_has_embedding"]) == [False] * 4
    assert [payload["embedding"] for payload in out_df["text_embeddings_1b_v2"]] == [[]] * 4


def test_an_oom_alone_is_not_classified_as_an_engine_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "_embed_group", _raise(OutOfMemoryError(OOM_IN_GELU)))

    out_df = runtime.embed_text_main_text_embed(_batch(), model=object(), inference_batch_size=2)

    assert list(out_df["text_embeddings_1b_v2_has_embedding"]) == [False] * 4
    assert [payload["embedding"] for payload in out_df["text_embeddings_1b_v2"]] == [[]] * 4


def test_engine_lifecycle_classifier_rejects_unrelated_failures() -> None:
    assert not runtime._is_engine_lifecycle_failure(ValueError("could not decode image payload"))
    assert not runtime._is_engine_lifecycle_failure(TimeoutError("read timed out"))
    # Recoverable on the HuggingFace backend; see the module docstring.
    assert not runtime._is_engine_lifecycle_failure(OutOfMemoryError(OOM_IN_GELU))
    # vLLM documents this one as recoverable in its own source.
    assert not runtime._is_engine_lifecycle_failure(EngineGenerateError("generate() failed"))
    assert runtime._is_engine_lifecycle_failure(RuntimeError(ENGINE_INIT_FAILED))
