# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A run that loses embedding rows must fail, not publish a short index.

Correctness comes from the LanceDB writer guard, which is general: any row, any
cause, any backend. It refuses to build an index when rows arrive with no
embedding. See ``test_lancedb_incomplete_index_guard.py``.

This module tests the layer above it, which is narrow on purpose.
``embed_text_main_text_embed`` catches every exception from ``_embed_group`` and
returns ``{"embedding": [], "error": ...}`` for the whole batch. That per-batch
resilience is right for the endpoint path, where one HTTP call can fail alone.
It is wrong for an in-process engine that has stopped serving: every later batch
fails too, and a failed engine returns instantly, so it drains the queue far
faster than a healthy one. Aborting there is a fast-fail optimisation on top of
the writer guard - it turns a failure at the terminal write into one within
minutes. It is not what makes the result correct.

Because it only buys latency, the fatal set stays limited to signals that mean
"the engine is not serving" and that a recoverable backend cannot produce.
Classifying an unmeasured exception would ship false failures. That is why a
bare ``OutOfMemoryError`` is excluded: the HuggingFace backend raises it from an
ordinary forward pass where a smaller next batch can succeed, and the two
backends are not distinguishable at the point of classification. A run that
loses rows to an OOM still fails - at the writer.

``LocalEmbedderRowsLostError`` is the signal for *partial* loss and comes from
where the loss happens. The local embedders' ``_finalize_vectors`` is the only
function that knows how many rows failed to embed; it used to zero-pad them and
discard the count, which is why ``has_embedding`` could report ``True`` for a
row carrying nothing. The writer cannot detect a padded row, so neither layer
subsumes the other.

The tests use stand-in exception classes rather than importing vLLM or torch,
matching the module under test, which identifies failures by class name and
message so it keeps working on the endpoint-only path.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from nemo_retriever.models.embed_errors import (
    LocalEmbedderReturnedNothingError,
    LocalEmbedderRowsLostError,
)
from nemo_retriever.models.inference import main_text_embed, runtime


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
    ("exc", "label"),
    [
        pytest.param(RuntimeError(ENGINE_INIT_FAILED), "engine-core-init-failed", id="refused-at-startup"),
        pytest.param(ValueError(ADMISSION_REFUSAL), "free-memory-gate", id="admission-gate-valueerror"),
        pytest.param(EngineDeadError("EngineCore encountered an issue"), "engine-dead", id="dead-after-admission"),
        pytest.param(
            _wrapped(RuntimeError("Worker proc died unexpectedly"), EngineDeadError("EngineCore is dead")),
            "wrapped-engine-dead",
            id="engine-dead-wrapped-in-runtimeerror",
        ),
        pytest.param(
            LocalEmbedderReturnedNothingError(
                "Local embedder returned no embeddings for a batch of 2 input(s). "
                "The in-process engine produced nothing, so it is not serving."
            ),
            "returned-nothing",
            id="engine-answered-with-nothing",
        ),
        pytest.param(
            LocalEmbedderRowsLostError(lost=7, total=64, embedder="LlamaNemotronEmbedVL1BV2VLLMEmbedder"),
            "rows-lost",
            id="engine-lost-part-of-the-batch",
        ),
    ],
)
def test_local_engine_failure_propagates(monkeypatch: pytest.MonkeyPatch, exc: BaseException, label: str) -> None:
    """Known-bad: returns a full batch of ``embedding: []`` and the run continues."""
    monkeypatch.setattr(runtime, "_embed_group", _raise(exc))

    with pytest.raises(type(exc)):
        runtime.embed_text_main_text_embed(_batch(), model=object(), inference_batch_size=2)


def test_local_embedder_returning_nothing_is_raised_as_a_classified_failure() -> None:
    """An all-empty local batch must not escape as a plain ``ValueError``.

    An embedder that answers a non-empty batch with nothing is not serving.
    ``_callable_runner`` used to report that as a bare count mismatch, which
    ``_is_engine_lifecycle_failure`` does not match, so the run continued and
    emptied the index anyway.

    This is not a backstop for a hypothetical callable. It is the only layer
    that sees a shipped failure: when vLLM yields no outputs for a batch,
    ``embed_with_vllm_llm`` returns ``[]``, and ``_finalize_vectors`` counts no
    loss because there are no rows to count - see
    ``test_vllm_embed.py::test_finalize_vectors_cannot_see_a_zero_output_batch``.
    That path raises nothing anywhere else.

    Known-bad: raises ``ValueError`` with "mismatched number of embeddings",
    which the classifier rejects.
    """
    with pytest.raises(LocalEmbedderReturnedNothingError):
        main_text_embed._callable_runner([["page one", "page two"]], embedder=lambda _texts: [], batch_size=2)

    assert runtime._is_engine_lifecycle_failure(LocalEmbedderReturnedNothingError("no vectors"))


def test_partial_local_result_is_still_a_plain_value_error() -> None:
    """Guard: a count mismatch that is not "nothing at all" stays unclassified.

    The embedder did produce vectors, so this is a data-shape problem rather
    than a dead engine, and it must keep its per-batch handling. Passes before
    and after.
    """
    with pytest.raises(ValueError) as excinfo:
        main_text_embed._callable_runner([["page one", "page two"]], embedder=lambda _texts: [[1.0, 2.0]], batch_size=2)

    assert not isinstance(excinfo.value, LocalEmbedderReturnedNothingError)
    assert not runtime._is_engine_lifecycle_failure(excinfo.value)


@pytest.mark.parametrize(
    "model",
    [pytest.param(None, id="model-nulled-by-the-operator"), pytest.param(object(), id="model-also-set")],
)
def test_endpoint_mode_still_absorbs_the_same_failure(monkeypatch: pytest.MonkeyPatch, model: object) -> None:
    """Guard: endpoint mode has no in-process engine, so nothing changes there.

    Passes before and after. A service-mode user must keep per-batch resilience.

    Both parameters matter. In production the model is always ``None`` when an
    endpoint is configured - ``operators/embed/cpu_operator.py:36`` and
    ``operators/embed/gpu_operator.py:37`` null it - so the ``model is None``
    case alone would still pass if the ``endpoint is None`` term were dropped
    from the re-raise condition. The second parameter pins that term directly,
    as defence against a future change to those two actors.
    """
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
    """A bare OOM is absorbed here; a run that loses rows still fails at the writer.

    The HuggingFace backend raises ``OutOfMemoryError`` from an ordinary forward
    pass, where a smaller next batch can succeed. The backend is not visible at
    the point of classification, so treating the exception as fatal would abort
    on a per-batch condition.

    Absorbing it is not the same as the run surviving. The batch becomes
    ``{"embedding": []}`` for every row, which the writer guard treats as fatal,
    so a run that really lost those rows fails at the write. The second
    assertion pins that shape, so this test cannot be read as a claim that the
    run recovers.
    """
    monkeypatch.setattr(runtime, "_embed_group", _raise(OutOfMemoryError(OOM_IN_GELU)))

    out_df = runtime.embed_text_main_text_embed(_batch(), model=object(), inference_batch_size=2)

    # Claim 1: the runtime absorbed it rather than re-raising.
    assert list(out_df["text_embeddings_1b_v2_has_embedding"]) == [False] * 4

    # Claim 2: what it absorbed into is the writer-fatal shape, so the run ends
    # at the write, not here.
    assert [payload["embedding"] for payload in out_df["text_embeddings_1b_v2"]] == [[]] * 4


def test_engine_death_after_an_oom_is_fatal_from_the_next_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Excluding the OOM costs one batch, not the run.

    An engine that dies mid-inference raises ``EngineDeadError`` from the next
    batch onward, and that is fatal. This pins the cost of the OOM carve-out as
    bounded rather than open-ended.
    """
    calls: list[int] = []

    def _fail_then_die(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        calls.append(1)
        if len(calls) == 1:
            raise OutOfMemoryError(OOM_IN_GELU)
        raise EngineDeadError("EngineCore encountered an issue")

    monkeypatch.setattr(runtime, "_embed_group", _fail_then_die)

    absorbed = runtime.embed_text_main_text_embed(_batch(), model=object(), inference_batch_size=2)
    assert list(absorbed["text_embeddings_1b_v2_has_embedding"]) == [False] * 4

    with pytest.raises(EngineDeadError):
        runtime.embed_text_main_text_embed(_batch(), model=object(), inference_batch_size=2)


def test_local_batch_level_failure_is_still_absorbed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard: a failure that is not an engine-lifecycle failure stays non-fatal.

    Passes before and after. The change is deliberately narrow - only failures
    that mean the engine stopped serving are fatal, because only those were
    measured to affect every subsequent batch.
    """
    monkeypatch.setattr(runtime, "_embed_group", _raise(ValueError("could not decode image payload for row 3")))

    out_df = runtime.embed_text_main_text_embed(_batch(), model=object(), inference_batch_size=2)

    assert list(out_df["text_embeddings_1b_v2_has_embedding"]) == [False] * 4


def test_engine_lifecycle_classifier_rejects_unrelated_failures() -> None:
    """Pins the classifier's specificity - it must not fire on ordinary failures.

    Cannot run on unmodified HEAD: the helper does not exist there, so this
    fails with ``AttributeError`` rather than with a wrong answer.
    """
    assert not runtime._is_engine_lifecycle_failure(ValueError("could not decode image payload"))
    assert not runtime._is_engine_lifecycle_failure(TimeoutError("read timed out"))
    # Recoverable on the HuggingFace backend; see the module docstring.
    assert not runtime._is_engine_lifecycle_failure(OutOfMemoryError(OOM_IN_GELU))
    # vLLM documents this one as recoverable in its own source.
    assert not runtime._is_engine_lifecycle_failure(EngineGenerateError("generate() failed"))
    assert runtime._is_engine_lifecycle_failure(RuntimeError(ENGINE_INIT_FAILED))


def test_the_fatal_set_stays_narrow() -> None:
    """Pins the exact fatal set, so widening it is a deliberate edit.

    Each name here has a cited reason in ``runtime.py``. ``EngineDeadError`` is
    "Unrecoverable" in vLLM's own definition; ``LocalEmbedderReturnedNothingError``
    and ``LocalEmbedderRowsLostError`` are raised by this repo, each in one place
    with one meaning. ``OutOfMemoryError`` and ``EngineGenerateError`` were
    removed because a recoverable backend raises them.

    Cannot run on unmodified HEAD: the set does not exist there.
    """
    assert set(runtime._ENGINE_LIFECYCLE_EXC_NAMES) == {
        "EngineDeadError",
        "LocalEmbedderReturnedNothingError",
        "LocalEmbedderRowsLostError",
    }


def test_engine_lifecycle_classifier_visits_a_cyclic_chain_once_per_link() -> None:
    """The ``seen`` set, not the depth cap, is what stops a cyclic chain.

    Termination alone does not test ``seen``: the depth-5 cap ends the walk
    either way, so an assertion that the call returns would still pass with
    ``seen`` deleted. Visit count does distinguish them. A two-link cycle costs
    two visits with ``seen`` and five without, because the cap then does the
    stopping.

    Same status as the test above: it cannot run on unmodified HEAD, where the
    helper does not exist.
    """
    visits: list[str] = []

    class _CountingError(RuntimeError):
        def __str__(self) -> str:
            visits.append(self.args[0])
            return str(self.args[0])

    first = _CountingError("first")
    second = _CountingError("second")
    first.__context__ = second
    second.__context__ = first

    assert not runtime._is_engine_lifecycle_failure(first)
    assert visits == ["first", "second"]


# ---------------------------------------------------------------------------
# Known gap: two embed entry points swallow ``LocalEmbedderRowsLostError``.
#
# Neither is on the shipped route, so these tests pin the gap rather than close
# it. They exist so a future import-site change that flips onto one of these
# functions shows up as a failing, clearly named test instead of as a silently
# short index in production. See ``models/embed_errors.py``
# ``LocalEmbedderRowsLostError`` for the full analysis and for why the writer's
# fatal set must NOT be widened to cover ``None``.
# ---------------------------------------------------------------------------


def test_the_shipped_embed_actors_use_the_guarded_runtime_function() -> None:
    """All three shipped embed actors resolve to the guarded implementation.

    This is the reachability claim the two gap tests below depend on. If it ever
    fails, the gap stops being theoretical and the swallowing paths must be
    fixed.
    """
    from nemo_retriever.operators.embed import cpu_operator, gpu_operator, operators

    for module in (operators, gpu_operator, cpu_operator):
        assert module.embed_text_main_text_embed is runtime.embed_text_main_text_embed


def test_text_embed_operator_route_swallows_the_rows_lost_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pins the gap in ``operators/embed/text_embed.py``.

    Its ``except BaseException`` around ``model.embed(...)`` converts the raise
    into ``{"embedding": None}``, which the LanceDB writer treats as a
    legitimate silent drop. Unprotected today, and off the shipped route.

    Passes before and after this change: it asserts pre-existing behaviour.
    """
    from nemo_retriever.operators.embed import text_embed

    class _LosingEmbedder:
        def embed(self, texts: list[str], **_kwargs: Any) -> list:
            raise LocalEmbedderRowsLostError(lost=1, total=len(texts), embedder="_LosingEmbedder")

    out_df = text_embed.embed_text_1b_v2(_batch(), model=_LosingEmbedder(), inference_batch_size=2)

    payloads = list(out_df["text_embeddings_1b_v2"])
    assert [p["embedding"] for p in payloads] == [None] * 4
    assert all(p["error"]["type"] == "LocalEmbedderRowsLostError" for p in payloads)
    # The shape the writer deliberately does not treat as fatal, because
    # ``text_embed.py`` also writes ``None`` for a legitimate blank-text row.
    assert not any(p["embedding"] == [] for p in payloads)


def test_modality_pipeline_duplicate_swallows_the_rows_lost_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pins the gap in ``common/modality/pipeline/embedding.py``.

    This module exports a function with the *identical* name as the patched one
    and is re-exported from its package ``__init__``, so it is the higher-risk
    of the two: a future import-site change could flip to it and remove all
    three guard layers with no visible diff at the call site.

    Passes before and after this change: it asserts pre-existing behaviour.
    """
    from nemo_retriever.common.modality.pipeline import embedding as duplicate

    assert duplicate.embed_text_main_text_embed is not runtime.embed_text_main_text_embed

    monkeypatch.setattr(
        duplicate,
        "_embed_group",
        _raise(LocalEmbedderRowsLostError(lost=2, total=4, embedder="LlamaNemotronEmbed1BV2Embedder")),
    )

    out_df = duplicate.embed_text_main_text_embed(_batch(), model=object(), inference_batch_size=2)

    payloads = list(out_df["text_embeddings_1b_v2"])
    assert [p["embedding"] for p in payloads] == [None] * 4
    assert all(p["error"]["type"] == "LocalEmbedderRowsLostError" for p in payloads)


class _StubVLEmbedder:
    """VL embedder stub whose per-call return length is scripted by the test."""

    def __init__(self, *, images=None, text_image=None, text=None):
        self._images = images
        self._text_image = text_image
        self._text = text

    def embed_images(self, images_b64, *, batch_size=64):
        return self._images

    def embed_text_image(self, texts, images_b64, *, batch_size=64):
        return self._text_image

    def embed(self, texts, *, batch_size=64):
        return self._text


def _image_frame(image_values):
    return pd.DataFrame({"_image_b64": list(image_values), "text": [""] * len(image_values)})


# --- the guard must fire: the engine answered short, rows would be lost ---


def test_image_mode_short_answer_is_fatal():
    """A row submitted with an image and answered with nothing must fail the run.

    Without the guard the shortfall is padded with ``None``, which both writers
    ignore, so the row is dropped and the run still succeeds.
    """
    df = _image_frame(["b64-a", "b64-b", "b64-c"])
    embedder = _StubVLEmbedder(images=[[0.1, 0.2]])  # 1 vector for 3 images

    with pytest.raises(LocalEmbedderRowsLostError) as excinfo:
        main_text_embed._multimodal_callable_runner(df, embedder=embedder, batch_size=8, embed_modality="image")
    assert excinfo.value.lost == 2
    assert excinfo.value.total == 3


def test_image_mode_empty_answer_is_fatal():
    """Zero vectors for a chunk that did submit images is the whole-batch failure."""
    df = _image_frame(["b64-a", "b64-b"])
    embedder = _StubVLEmbedder(images=[])

    with pytest.raises(LocalEmbedderRowsLostError):
        main_text_embed._multimodal_callable_runner(df, embedder=embedder, batch_size=8, embed_modality="image")


def test_text_image_mode_short_multimodal_answer_is_fatal():
    df = pd.DataFrame({"_image_b64": ["b64-a", "b64-b"], "text": ["alpha", "beta"]})
    embedder = _StubVLEmbedder(text_image=[[0.1, 0.2]])  # 1 vector for 2 paired rows

    with pytest.raises(LocalEmbedderRowsLostError):
        main_text_embed._multimodal_callable_runner(df, embedder=embedder, batch_size=8, embed_modality="text_image")


def test_text_image_mode_short_text_fallback_answer_is_fatal():
    """The text-only fallback subset is owed one vector per row as well."""
    df = pd.DataFrame({"_image_b64": ["", ""], "text": ["alpha", "beta"]})
    embedder = _StubVLEmbedder(text=[[0.1, 0.2]])  # 1 vector for 2 fallback rows

    with pytest.raises(LocalEmbedderRowsLostError):
        main_text_embed._multimodal_callable_runner(df, embedder=embedder, batch_size=8, embed_modality="text_image")


# --- the guard must stay silent: these are legitimate short answers ---


def test_image_mode_rows_without_images_do_not_fire_the_guard():
    """Rows with no image are owed nothing; they get ``None`` by contract.

    This is the false-failure the naive ``if not vecs_list: raise`` would cause.
    """
    df = _image_frame(["b64-a", "", "b64-c"])
    embedder = _StubVLEmbedder(images=[[0.1, 0.2], [0.3, 0.4]])  # 2 vectors for 2 images

    out = main_text_embed._multimodal_callable_runner(df, embedder=embedder, batch_size=8, embed_modality="image")
    assert out["embeddings"] == [[0.1, 0.2], None, [0.3, 0.4]]


def test_image_mode_chunk_with_no_images_at_all_does_not_fire_the_guard():
    """An image-free chunk submits nothing, so zero vectors back is correct."""
    df = _image_frame(["", ""])
    embedder = _StubVLEmbedder(images=[])

    out = main_text_embed._multimodal_callable_runner(df, embedder=embedder, batch_size=8, embed_modality="image")
    assert out["embeddings"] == [None, None]


def test_text_image_mode_mixed_rows_do_not_fire_the_guard():
    """Paired, text-only and empty rows each get their own correct treatment."""
    df = pd.DataFrame({"_image_b64": ["b64-a", "", ""], "text": ["alpha", "beta", "   "]})
    embedder = _StubVLEmbedder(text_image=[[0.1, 0.2]], text=[[0.3, 0.4]])

    out = main_text_embed._multimodal_callable_runner(df, embedder=embedder, batch_size=8, embed_modality="text_image")
    assert out["embeddings"] == [[0.1, 0.2], [0.3, 0.4], None]


def test_image_mode_healthy_batch_does_not_fire_the_guard():
    df = _image_frame(["b64-a", "b64-b"])
    embedder = _StubVLEmbedder(images=[[0.1, 0.2], [0.3, 0.4]])

    out = main_text_embed._multimodal_callable_runner(df, embedder=embedder, batch_size=8, embed_modality="image")
    assert out["embeddings"] == [[0.1, 0.2], [0.3, 0.4]]
