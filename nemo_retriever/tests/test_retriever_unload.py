# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise cached cleanup ownership without loading models or using a GPU."""

from threading import Event, Lock, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock
from weakref import ref

import pytest

from nemo_retriever.graph.pipeline_graph import Graph, Node
from nemo_retriever.graph.retriever import Retriever
from nemo_retriever.models.local.llama_nemotron_embed_1b_v2_embedder import LlamaNemotronEmbed1BV2Embedder
from nemo_retriever.models.local.nemotron_rerank_vl_v2 import NemotronRerankVLV2VLLM
from nemo_retriever.models.local.llama_nemotron_embed_vl_1b_v2_embedder import (
    LlamaNemotronEmbedVL1BV2VLLMEmbedder,
)
from nemo_retriever.operators.embed.gpu_operator import _BatchEmbedActor as GPUEmbedActor
from nemo_retriever.operators.embed.operators import _BatchEmbedActor
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.rerank import NemotronRerankActor, NemotronRerankGPUActor
from nemo_retriever.query.agentic import AgenticRetriever


@pytest.mark.parametrize(
    "model_class",
    [LlamaNemotronEmbed1BV2Embedder, LlamaNemotronEmbedVL1BV2VLLMEmbedder, NemotronRerankVLV2VLLM],
)
@pytest.mark.parametrize("cuda_available", [False, True])
def test_unload_releases_engine_reference_before_cuda_cleanup(monkeypatch, model_class, cuda_available):
    events = []

    class EngineCore:
        def shutdown(self, *, timeout):
            events.append(("shutdown", timeout))

    class LLM:
        def __init__(self):
            self.llm_engine = SimpleNamespace(engine_core=EngineCore())

    model = model_class.__new__(model_class)
    model._llm = LLM()
    engine_ref = ref(model._llm)

    def is_available():
        assert model._llm is None
        assert engine_ref() is None
        return cuda_available

    def empty_cache():
        assert engine_ref() is None
        events.append(("empty_cache",))

    monkeypatch.setattr("torch.cuda.is_available", is_available)
    monkeypatch.setattr("torch.cuda.empty_cache", empty_cache)
    model.unload()
    model.unload()
    assert events == [("shutdown", 30.0)] + ([("empty_cache",)] if cuda_available else [])


def _cached_chain(embedder_class=LlamaNemotronEmbedVL1BV2VLLMEmbedder):
    embedder = embedder_class.__new__(embedder_class)
    llm = MagicMock()
    embedder._llm = llm
    operator = GPUEmbedActor.__new__(GPUEmbedActor)
    operator._model = embedder
    operator._owns_model = True
    archetype = _BatchEmbedActor(params=None)
    archetype._resolved_delegate = operator
    archetype._resolved_delegate_key = (1, 1)
    inner = Retriever()
    inner._cached_graph = Graph()
    inner._cached_graph.add_root(Node(archetype, operator_kwargs={}))
    inner._cache_key = ("cached",)
    agent = AgenticRetriever.__new__(AgenticRetriever)
    agent._lock = Lock()
    agent._chat_completion_fn = None
    agent._retriever = inner
    return agent, inner, archetype, operator, embedder, llm


@pytest.mark.parametrize("cuda_available", [False, True])
@pytest.mark.parametrize("embedder_class", [LlamaNemotronEmbed1BV2Embedder, LlamaNemotronEmbedVL1BV2VLLMEmbedder])
def test_agentic_unload_shuts_down_cached_embedding_chain(monkeypatch, cuda_available, embedder_class):
    agent, inner, archetype, operator, embedder, llm = _cached_chain(embedder_class)
    monkeypatch.setattr("torch.cuda.is_available", lambda: cuda_available)
    empty_cache = MagicMock()
    monkeypatch.setattr("torch.cuda.empty_cache", empty_cache)
    resolve = MagicMock(side_effect=AssertionError("cleanup must not resolve operators"))
    monkeypatch.setattr(archetype, "_resolve_delegate", resolve)

    def shutdown(*, timeout):
        assert timeout == 30.0
        assert embedder._llm is llm
        empty_cache.assert_not_called()

    llm.llm_engine.engine_core.shutdown.side_effect = shutdown
    agent.unload()
    agent.unload()

    llm.llm_engine.engine_core.shutdown.assert_called_once_with(timeout=30.0)
    assert empty_cache.call_count == int(cuda_available)
    assert embedder._llm is None
    assert operator._model is None
    assert archetype._resolved_delegate is None
    assert archetype._resolved_delegate_key is None
    assert inner._cached_graph is None
    assert inner._cache_key is None
    resolve.assert_not_called()


@pytest.mark.parametrize("embedder_class", [LlamaNemotronEmbed1BV2Embedder, LlamaNemotronEmbedVL1BV2VLLMEmbedder])
def test_shutdown_failure_preserves_entire_chain_for_retry(monkeypatch, embedder_class):
    agent, inner, archetype, operator, embedder, llm = _cached_chain(embedder_class)
    graph = inner._cached_graph
    empty_cache = MagicMock()
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.empty_cache", empty_cache)
    llm.llm_engine.engine_core.shutdown.side_effect = [RuntimeError("shutdown failed"), None]

    with pytest.raises(RuntimeError, match="shutdown failed"):
        agent.unload()

    assert inner._cached_graph is graph
    assert inner._cache_key == ("cached",)
    assert archetype._resolved_delegate is operator
    assert archetype._resolved_delegate_key == (1, 1)
    assert operator._model is embedder
    assert embedder._llm is llm
    empty_cache.assert_not_called()

    agent.unload()
    assert llm.llm_engine.engine_core.shutdown.call_count == 2
    empty_cache.assert_called_once_with()
    assert inner._cached_graph is None


def test_unload_unused_retriever_and_archetype_does_not_construct_operators(monkeypatch):
    inner = Retriever()
    archetype = _BatchEmbedActor(params=None)
    build = MagicMock(side_effect=AssertionError("cleanup must not build a graph"))
    resolve = MagicMock(side_effect=AssertionError("cleanup must not resolve operators"))
    monkeypatch.setattr(inner, "_build_default_graph", build)
    monkeypatch.setattr(archetype, "_resolve_delegate", resolve)
    inner.unload()
    archetype.unload()
    build.assert_not_called()
    resolve.assert_not_called()


def test_unload_leaves_custom_graph_owned_by_caller():
    graph = MagicMock()
    inner = Retriever(graph=graph)
    inner.unload()
    assert inner.graph is graph
    assert graph.mock_calls == []


def test_unload_visits_shared_graph_operators_once():
    operator = MagicMock(spec=AbstractOperator)
    operator.unload = MagicMock()
    node = Node(operator, operator_kwargs={})
    child = Node(operator, operator_kwargs={})
    node.add_child(child)
    inner = Retriever()
    inner._cached_graph = Graph()
    inner._cached_graph.roots = [node, child, node]
    inner.unload()
    operator.unload.assert_called_once_with()


def test_agent_llm_failure_retains_owner_and_still_cleans_inner_retriever():
    agent = AgenticRetriever.__new__(AgenticRetriever)
    agent._lock = Lock()
    chat_fn = MagicMock()
    chat_fn.unload.side_effect = [RuntimeError("agent shutdown failed"), None]
    agent._chat_completion_fn = chat_fn
    agent._retriever = MagicMock()
    with pytest.raises(RuntimeError, match="agent shutdown failed"):
        agent.unload()
    assert agent._chat_completion_fn is chat_fn
    agent._retriever.unload.assert_called_once_with()
    agent.unload()
    assert agent._chat_completion_fn is None


def test_reranker_cleanup_forwards_through_cached_delegate_and_retries(monkeypatch):
    model = NemotronRerankVLV2VLLM.__new__(NemotronRerankVLV2VLLM)
    model._llm = llm = MagicMock()
    llm.llm_engine.engine_core.shutdown.side_effect = [RuntimeError("shutdown failed"), None]
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    operator = NemotronRerankGPUActor.__new__(NemotronRerankGPUActor)
    operator._model = model
    archetype = NemotronRerankActor()
    archetype._resolved_delegate = operator
    archetype._resolved_delegate_key = (1, 1)
    inner = Retriever(rerank=True)
    graph = Graph()
    graph.add_root(Node(archetype, operator_kwargs={}))
    inner._cached_graph = graph

    with pytest.raises(RuntimeError, match="shutdown failed"):
        inner.unload()
    assert inner._cached_graph is graph
    assert archetype._resolved_delegate is operator
    assert operator._model is model
    assert model._llm is llm

    inner.unload()
    inner.unload()
    assert llm.llm_engine.engine_core.shutdown.call_count == 2
    llm.llm_engine.engine_core.shutdown.assert_called_with(timeout=30.0)
    assert model._llm is None
    assert operator._model is None
    assert archetype._resolved_delegate is None


def test_query_configuration_change_unloads_old_graph_before_building_new(monkeypatch):
    inner = Retriever()
    old_operator = MagicMock(spec=AbstractOperator)
    old_operator.unload = MagicMock()
    old_graph = Graph()
    old_graph.add_root(Node(old_operator, operator_kwargs={}))
    new_graph = Graph()
    builds = []

    def build(**kwargs):
        if builds:
            old_operator.unload.assert_called_once_with()
            assert inner._cached_graph is None
        builds.append(kwargs)
        return old_graph if len(builds) == 1 else new_graph

    monkeypatch.setattr(inner, "_build_default_graph", build)
    assert inner._get_graph() is old_graph
    assert inner._get_graph() is old_graph
    old_operator.unload.assert_not_called()
    inner.embed_kwargs["model_name"] = "different"
    old_key = inner._cache_key
    old_operator.unload.side_effect = RuntimeError("shutdown failed")
    with pytest.raises(RuntimeError, match="shutdown failed"):
        inner._get_graph()
    assert inner._cached_graph is old_graph
    assert inner._cache_key == old_key
    assert len(builds) == 1

    old_operator.unload.reset_mock(side_effect=True)
    assert inner._get_graph() is new_graph
    old_operator.unload.assert_called_once_with()
    assert len(builds) == 2
    inner.unload()
    assert inner._cached_graph is None


def test_agentic_unload_waits_for_active_retrieval_hop():
    agent = AgenticRetriever.__new__(AgenticRetriever)
    agent._lock = Lock()
    agent._chat_completion_fn = None
    agent._cfg = SimpleNamespace(candidate_k=None)
    agent._retriever = MagicMock()
    query_started = Event()
    release_query = Event()
    cleanup_attempted = Event()
    errors = []

    def query(*args, **kwargs):
        query_started.set()
        assert release_query.wait(5)
        return []

    def unload_inner():
        # Verify cleanup holds the same lock used by retrieval hops.
        acquired = agent._lock.acquire(blocking=False)
        if acquired:
            agent._lock.release()
        assert not acquired
        assert release_query.is_set()

    def run_hop():
        try:
            agent._retrieve_for_agent("query", 1)
        except BaseException as exc:
            errors.append(exc)

    def run_cleanup():
        try:
            cleanup_attempted.set()
            agent.unload()
        except BaseException as exc:
            errors.append(exc)

    agent._retriever.query.side_effect = query
    agent._retriever.unload.side_effect = unload_inner
    hop = Thread(target=run_hop)
    cleanup = Thread(target=run_cleanup)
    hop.start()
    try:
        assert query_started.wait(5)
        cleanup.start()
        assert cleanup_attempted.wait(5)
        agent._retriever.unload.assert_not_called()
    finally:
        release_query.set()
        hop.join(5)
        if cleanup.ident is not None:
            cleanup.join(5)
    assert not hop.is_alive()
    assert not cleanup.is_alive()
    assert not errors
    agent._retriever.unload.assert_called_once_with()
