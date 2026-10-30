# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""System One (Jev) client and the Jev-controlled retrieval loop.

Run with:
    cd nemo_retriever && uv run pytest tests/test_nemo_agent_jev.py -v
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Dict, List

import httpx
import pandas as pd
import pytest

from nemo_retriever._agentic.nemo_agent import create_retrieve_tool
from nemo_retriever._agentic.nemo_agent.jev import (
    JevCallError,
    JevClient,
    JevClientConfig,
    JevContextLengthError,
    JevControlledAgent,
    JevControllerConfig,
    choice,
    noul,
)
from nemo_retriever._agentic.nemo_agent.llm import bind_query_id, bind_stage, create_llm, create_llm_config
from nemo_retriever.operators.graph_ops.react_agent_operator import ReActAgentOperator
from nemo_retriever.operators.graph_ops.selection_agent_operator import SelectionAgentOperator

_URL = "https://jev.test/v1/systemone"


def _jev_body(answers: Dict[str, Any], input_tokens: int = 100) -> Dict[str, Any]:
    return {"model": "jev", "answers": answers, "usage": {"input_tokens": input_tokens, "output_tokens": 5}}


def _decision(action: str, sufficient: float, next_search: str = "s0") -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """A Jev reply to a controller request: the action, P(sufficient), and the pending-search pick."""

    def reply(payload: Dict[str, Any]) -> Dict[str, Any]:
        answers: Dict[str, Any] = {
            "action": {"type": "choice", "choice": action, "probabilities": {action: 0.9}, "confidence": 0.9},
            "sufficient": {"type": "noul", "noul": sufficient},
        }
        if "next_search" in payload["questions"]:
            answers["next_search"] = {"type": "choice", "choice": next_search, "probabilities": {}, "confidence": 1.0}
        return _jev_body(answers)

    return reply


class _ScriptedJev:
    """Stands in for the System One endpoint by replaying one reply per request."""

    def __init__(self, replies: List[Callable[[Dict[str, Any]], Dict[str, Any]]]) -> None:
        self.replies = list(replies)
        self.payloads: List[Dict[str, Any]] = []

    def transport(self) -> httpx.MockTransport:
        def handle(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            self.payloads.append(payload)
            return httpx.Response(200, json=self.replies.pop(0)(payload))

        return httpx.MockTransport(handle)

    def client(self) -> JevClient:
        return JevClient(JevClientConfig(url=_URL, api_key="k", retry_sleep_s=0.0), transport=self.transport())


def _tool_call_reply(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
    }


def _scripted_llm(replies: List[Dict[str, Any]], seen: List[Dict[str, Any]] | None = None):
    def completion(**kwargs):
        if seen is not None:
            seen.append(kwargs)
        return replies.pop(0)

    return create_llm(create_llm_config("callable", model="test-model"), completion_fn=completion)


def _retriever(calls: List[str]):
    def retrieve(query: str, top_k: int):
        calls.append(query)
        slug = query.split()[0].lower()
        return [{"id": f"{slug}_{i}", "text": f"{query} evidence {i}", "score": 1.0 - i / 10} for i in range(3)]

    return create_retrieve_tool("default", retrieve, default_top_k=3)


def _agent(jev: _ScriptedJev, llm_replies: List[Dict[str, Any]], calls: List[str], **options: Any):
    config = JevControllerConfig(mode="select", on_error="raise_all", **options)
    return JevControlledAgent(
        config=config, llm=_scripted_llm(llm_replies), retrieve_tool=_retriever(calls), jev=jev.client()
    )


def _plan(*subqueries: str) -> Dict[str, Any]:
    return _tool_call_reply("propose_subqueries", {"subqueries": list(subqueries)})


class TestJevClient:
    def test_answers_and_usage_are_attributed_to_query_and_stage(self):
        jev = _ScriptedJev([lambda p: _jev_body({"q": {"type": "noul", "noul": 0.8}}, input_tokens=42)])
        client = jev.client()

        async def ask():
            with bind_query_id("q1"), bind_stage("s"):
                return await client.aevaluate({"query": "x"}, {"q": noul("Is it?")})

        answers = asyncio.run(ask())
        assert answers["q"]["noul"] == 0.8
        assert jev.payloads[0]["model"] == "jev-latest"
        usage = client.pop_query_usage("q1")
        assert usage["s"]["input_tokens"] == 42
        assert usage["s"]["requests"] == 1
        assert client.pop_query_usage("q1") == {}

    def test_retries_rate_limits_then_succeeds(self):
        statuses = [429, 200]

        def handle(request: httpx.Request) -> httpx.Response:
            status = statuses.pop(0)
            body = _jev_body({"q": {"type": "noul", "noul": 0.1}}) if status == 200 else {"error": "slow down"}
            return httpx.Response(status, json=body)

        client = JevClient(JevClientConfig(url=_URL, retry_sleep_s=0.0), transport=httpx.MockTransport(handle))
        assert asyncio.run(client.aevaluate("s", {"q": noul("?")}))["q"]["noul"] == 0.1

    def test_auth_failure_is_not_retried(self):
        attempts: List[int] = []

        def handle(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            return httpx.Response(403, json={"error": "key_model_access_denied"})

        client = JevClient(JevClientConfig(url=_URL, retry_sleep_s=0.0), transport=httpx.MockTransport(handle))
        with pytest.raises(JevCallError, match="HTTP 403"):
            asyncio.run(client.aevaluate("s", {"q": noul("?")}))
        assert len(attempts) == 1

    def test_wrapped_context_length_error_is_not_retried(self):
        attempts: List[int] = []

        def handle(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            return httpx.Response(502, text='upstream 400: "This model\'s maximum context length is 4096 tokens."')

        client = JevClient(JevClientConfig(url=_URL, retry_sleep_s=0.0), transport=httpx.MockTransport(handle))
        with pytest.raises(JevContextLengthError, match="HTTP 502"):
            asyncio.run(client.aevaluate("s", {"q": noul("?")}))
        assert len(attempts) == 1

    def test_missing_answers_raise(self):
        jev = _ScriptedJev([lambda p: _jev_body({})])
        with pytest.raises(JevCallError, match="missing answers"):
            asyncio.run(jev.client().aevaluate("s", {"q": noul("?")}))

    def test_choice_requires_two_options(self):
        with pytest.raises(ValueError):
            choice("Pick", {"only": None})


class TestJevControlledAgent:
    def test_plan_retrieve_finish_flow(self):
        calls: List[str] = []
        jev = _ScriptedJev([_decision("retrieve", 0.2, next_search="s1"), _decision("finish", 0.9)])
        agent = _agent(jev, [_plan("alpha facts", "beta facts")], calls, batch_retrieve=False)

        result = agent.run_sync("Original question?", query_id="q1")

        assert result.succeeded
        assert result.final_doc_ids == []
        assert calls == ["Original question?", "beta facts"]
        assert [entry["query_type"] for entry in result.retrieval_log] == ["main", "agent"]
        decisions = result.extra_data["jev_decisions"]
        assert [d["action"] for d in decisions] == ["retrieve", "finish"]
        state = jev.payloads[0]["state"]
        assert state["pending_searches"] == ["alpha facts", "beta facts"]
        assert {doc["id"] for doc in state["evidence"]} == {"original_0", "original_1", "original_2"}

    def test_batch_retrieve_runs_rounds_and_asks_jev_between_them(self):
        calls: List[str] = []
        jev = _ScriptedJev([_decision("expand", 0.1), _decision("finish", 0.9)])
        agent = _agent(jev, [_plan("alpha facts", "beta facts"), _plan("gamma facts")], calls)

        result = agent.run_sync("Original question?", query_id="q1")

        assert calls == ["Original question?", "alpha facts", "beta facts", "gamma facts"]
        assert [d["action"] for d in result.extra_data["jev_decisions"]] == ["retrieve", "expand", "retrieve", "finish"]
        assert len(jev.payloads) == 2
        first = jev.payloads[0]
        assert first["state"]["pending_searches"] == []
        assert set(first["questions"]["action"]["criteria"]) == {"expand", "think", "finish"}
        assert "next_search" not in first["questions"]

    def test_context_overflow_retries_with_less_evidence(self):
        calls: List[str] = []
        payloads: List[Dict[str, Any]] = []
        replies = [_decision("finish", 0.9)]

        def handle(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            payloads.append(payload)
            if len(payloads) == 1:
                return httpx.Response(502, text="upstream 400: maximum context length is 4096 tokens")
            return httpx.Response(200, json=replies.pop(0)(payload))

        jev = JevClient(JevClientConfig(url=_URL, retry_sleep_s=0.0), transport=httpx.MockTransport(handle))
        config = JevControllerConfig(
            mode="select", on_error="raise_all", min_retrievals=0, evidence_chars=500, evidence_char_budget=100
        )
        agent = JevControlledAgent(
            config=config, llm=_scripted_llm([_plan("alpha facts")]), retrieve_tool=_retriever(calls), jev=jev
        )

        result = agent.run_sync("Original question?", query_id="q1")

        assert result.extra_data["jev_decisions"][-1]["reason"] == "jev choice"
        evidence_chars = [sum(len(d["text"]) for d in p["state"]["evidence"]) for p in payloads]
        assert len(payloads) == 2 and evidence_chars[1] < evidence_chars[0]

    def test_jev_failure_runs_pending_searches_and_finishes(self):
        calls: List[str] = []

        def handle(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="unavailable")

        jev = JevClient(JevClientConfig(url=_URL, max_retries=0), transport=httpx.MockTransport(handle))
        config = JevControllerConfig(mode="select", on_error="raise_all")
        agent = JevControlledAgent(
            config=config,
            llm=_scripted_llm([_plan("alpha facts", "beta facts")]),
            retrieve_tool=_retriever(calls),
            jev=jev,
        )

        result = agent.run_sync("Original question?", query_id="q1")

        assert result.succeeded
        assert calls == ["Original question?", "alpha facts", "beta facts"]
        assert result.extra_data["jev_decisions"][-1]["reason"] == "jev call failed"

    def test_finish_below_threshold_falls_back_to_search(self):
        calls: List[str] = []
        jev = _ScriptedJev([_decision("finish", 0.3), _decision("finish", 0.95)])
        agent = _agent(jev, [_plan("alpha facts")], calls, min_retrievals=0, batch_retrieve=False)

        result = agent.run_sync("Original question?", query_id="q1")

        assert calls == ["Original question?", "alpha facts"]
        first = result.extra_data["jev_decisions"][0]
        assert first["picked"] == "finish"
        assert first["action"] == "retrieve"

    def test_finish_requires_min_retrievals(self):
        calls: List[str] = []
        jev = _ScriptedJev([_decision("finish", 0.99), _decision("finish", 0.99)])
        agent = _agent(jev, [_plan("alpha facts")], calls, min_retrievals=1)

        agent.run_sync("Original question?", query_id="q1")

        assert calls == ["Original question?", "alpha facts"]

    def test_expand_and_think_call_the_llm(self):
        calls: List[str] = []
        seen: List[Dict[str, Any]] = []
        jev = _ScriptedJev(
            [_decision("expand", 0.1), _decision("think", 0.1), _decision("retrieve", 0.1), _decision("finish", 0.9)]
        )
        llm_replies = [
            _plan(),
            _plan("gamma facts"),
            _tool_call_reply("record_analysis", {"analysis": "Needs the 2019 figure.", "subqueries": []}),
        ]
        config = JevControllerConfig(mode="select", on_error="raise_all", batch_retrieve=False)
        agent = JevControlledAgent(
            config=config, llm=_scripted_llm(llm_replies, seen), retrieve_tool=_retriever(calls), jev=jev.client()
        )

        result = agent.run_sync("Original question?", query_id="q1")

        assert calls == ["Original question?", "gamma facts"]
        assert [s["tools"][0]["function"]["name"] for s in seen] == [
            "propose_subqueries",
            "propose_subqueries",
            "record_analysis",
        ]
        assert jev.payloads[-1]["state"]["notes"] == ["Needs the 2019 figure."]
        usage = agent.llm.pop_query_usage("q1")
        assert set(usage) == {"jev_plan", "jev_expand", "jev_think"}
        assert agent.jev.pop_query_usage("q1")["jev_controller"]["requests"] == 4
        assert result.succeeded

    def test_budget_exhaustion_finishes_without_jev(self):
        calls: List[str] = []
        jev = _ScriptedJev([])
        agent = _agent(jev, [_plan()], calls, max_expansions=0, max_thinks=0)

        result = agent.run_sync("Original question?", query_id="q1")

        assert jev.payloads == []
        assert result.extra_data["jev_decisions"][0]["reason"] == "no other action available"

    def test_evidence_respects_character_budget(self):
        calls: List[str] = []
        jev = _ScriptedJev([_decision("retrieve", 0.1), _decision("finish", 0.9)])
        agent = _agent(
            jev, [_plan("alpha facts")], calls, evidence_chars=10, evidence_char_budget=30, batch_retrieve=False
        )

        agent.run_sync("Original question?", query_id="q1")

        evidence = jev.payloads[0]["state"]["evidence"]
        assert [doc["id"] for doc in evidence] == ["original_0", "original_1"]
        assert all(len(doc["text"]) <= 13 for doc in evidence)

    def test_duplicate_subqueries_are_dropped(self):
        calls: List[str] = []
        jev = _ScriptedJev([_decision("retrieve", 0.1), _decision("finish", 0.9)])
        agent = _agent(jev, [_plan("Original  QUESTION?", "alpha facts", "alpha facts")], calls, batch_retrieve=False)

        agent.run_sync("Original question?", query_id="q1")

        assert jev.payloads[0]["state"]["pending_searches"] == ["alpha facts"]

    def test_rejects_answer_mode(self):
        with pytest.raises(ValueError, match="select"):
            JevControlledAgent(
                config=JevControllerConfig(mode="answer"),
                llm=_scripted_llm([]),
                retrieve_tool=_retriever([]),
                jev=_ScriptedJev([]).client(),
            )


class TestOperatorWiring:
    def test_react_operator_requires_jev_settings(self):
        with pytest.raises(ValueError, match="jev_client"):
            ReActAgentOperator(llm_model="m", retriever_fn=lambda q, k: [], controller="jev")
        with pytest.raises(ValueError, match="select"):
            ReActAgentOperator(
                llm_model="m", retriever_fn=lambda q, k: [], controller="jev", mode="answer", jev_client={"url": _URL}
            )

    def test_agentic_config_validates_controller(self):
        from nemo_retriever.query.agentic import AgenticRetrievalConfig

        remote = {"llm_model": "m", "invoke_url": "http://llm/v1"}
        with pytest.raises(ValueError, match="jev_url"):
            AgenticRetrievalConfig(**remote, controller="jev")
        with pytest.raises(ValueError, match="controller"):
            AgenticRetrievalConfig(**remote, controller="system1", jev_url=_URL)
        with pytest.raises(ValueError, match="selection_max_candidates"):
            AgenticRetrievalConfig(**remote, selection_max_candidates=5)
        assert AgenticRetrievalConfig(**remote, controller=" JEV ", jev_url=_URL).controller == "jev"

    def test_selection_operator_ranks_only_max_candidates(self):
        seen: List[str] = []

        def completion_fn(**kwargs: Any) -> Dict[str, Any]:
            seen.append(json.dumps(kwargs["messages"]))
            return _tool_call_reply("log_selected_documents", {"doc_ids": ["b", "a"]})

        operator = SelectionAgentOperator(llm_model="m", top_k=2, chat_completion_fn=completion_fn, max_candidates=2)
        data = pd.DataFrame(
            {
                "query_id": ["0"] * 3,
                "query_text": ["q"] * 3,
                "doc_id": ["a", "b", "c"],
                "text": ["doc a", "doc b", "doc c"],
                "rrf_score": [0.9, 0.8, 0.7],
            }
        )

        out = operator.run(data)

        assert out["doc_id"].tolist() == ["b", "a"]
        assert "doc b" in seen[0] and "doc c" not in seen[0]

    def test_selection_operator_merges_extra_body_into_requests(self):
        seen: List[Dict[str, Any]] = []

        def completion_fn(**kwargs: Any) -> Dict[str, Any]:
            seen.append(kwargs["extra_body"])
            return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

        operator = SelectionAgentOperator(
            llm_model="m",
            chat_completion_fn=completion_fn,
            reasoning_effort="low",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        operator._build_llm().completion(messages=[{"role": "user", "content": "hi"}])

        assert seen == [{"reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": False}}]

    def test_react_operator_runs_jev_agent_and_reports_jev_usage(self, monkeypatch):
        replies = [_decision("finish", 0.9)]

        async def fake_post(self, payload, headers):
            return replies.pop(0)(payload)

        monkeypatch.setattr(JevClient, "_post", fake_post)
        operator = ReActAgentOperator(
            llm_model="m",
            retriever_fn=lambda query, top_k: [{"id": "d1", "text": "doc", "score": 1.0}],
            chat_completion_fn=lambda **kwargs: _plan("alpha facts"),
            controller="jev",
            jev_client={"url": _URL},
            jev_controller_options={"min_retrievals": 0},
        )

        agent = operator._ensure_agent()
        result = agent.run_sync("q", query_id="0")

        assert isinstance(agent, JevControlledAgent)
        assert [d["action"] for d in result.extra_data["jev_decisions"]] == ["retrieve", "finish"]
        usage = operator.pop_query_usage("0")
        assert usage["jev_controller"]["requests"] == 1
        assert "jev_plan" in usage
