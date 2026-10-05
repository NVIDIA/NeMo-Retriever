# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Continuous-batching vLLM actors for Ray Data embedding pipelines.

The Ray Data workers remain CPU-only and submit prepared text batches to a
small, persistent pool of full-GPU vLLM engines.  Each engine accepts several
concurrent calls and submits every prompt to ``AsyncLLM`` independently, which
allows vLLM to continuously batch requests across Ray input batches.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Sequence

from nemo_retriever.common.params import EmbedParams
from nemo_retriever.models.inference.embedding_input import ensure_embedding_input_policy_for_batch
from nemo_retriever.models.inference.runtime import embed_text_main_text_embed
from nemo_retriever.models.inference.shared import build_embed_kwargs
from nemo_retriever.operators.abstract_operator import AbstractOperator


class AsyncVllmEmbeddingEngine:
    """One full-GPU asynchronous vLLM pooling engine."""

    def __init__(
        self,
        *,
        engine_index: int,
        model: str,
        revision: str | None,
        max_model_len: int,
        gpu_memory_utilization: float,
        dimensions: int | None,
        max_num_seqs: int = 512,
        pooler_use_activation: bool | None = None,
    ) -> None:
        # Every engine needs a stable, disjoint rendezvous range.  A fixed range
        # is safe because there is exactly one engine actor for each index.
        os.environ["VLLM_PORT"] = str(20_000 + int(engine_index) * 64)
        os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")

        from vllm import AsyncEngineArgs
        from vllm.config.pooler import PoolerConfig
        from vllm.v1.engine.async_llm import AsyncLLM

        # Passing even the native dimension is interpreted by vLLM as a
        # matryoshka override. This checkpoint is not matryoshka-capable, so
        # preserve its native output width and validate it downstream.
        expected_dimensions = int(dimensions) if dimensions is not None else None
        del dimensions
        pooler_config = PoolerConfig(
            task="embed",
            pooling_type="MEAN",
            use_activation=pooler_use_activation,
        )
        engine_args = AsyncEngineArgs(
            model=model,
            revision=revision,
            runner="pooling",
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=int(max_model_len),
            gpu_memory_utilization=float(gpu_memory_utilization),
            max_num_seqs=int(max_num_seqs),
            pooler_config=pooler_config,
            disable_log_stats=True,
            enforce_eager=False,
        )
        self._engine = AsyncLLM.from_engine_args(engine_args)
        self._engine_index = int(engine_index)
        self._dimensions = expected_dimensions
        self._request_counter = 0
        self._request_slots = asyncio.Semaphore(int(max_num_seqs))

    async def ready(self) -> dict[str, int]:
        return {"engine_index": self._engine_index}

    async def _encode_one(self, prompt: str, request_id: str) -> list[float]:
        from vllm import PoolingParams

        final = None
        async with self._request_slots:
            async for output in self._engine.encode(prompt, PoolingParams(task="embed"), request_id):
                final = output
        pooling_output = getattr(final, "outputs", None)
        embedding = getattr(pooling_output, "embedding", None)
        if embedding is None:
            embedding = getattr(pooling_output, "data", None)
        if embedding is None:
            raise RuntimeError(f"vLLM returned no embedding for request {request_id}")
        tolist = getattr(embedding, "tolist", None)
        if callable(tolist):
            vector = tolist()
        else:
            vector = list(embedding)
        if len(vector) == 1 and isinstance(vector[0], (list, tuple)):
            vector = vector[0]
        if vector and isinstance(vector[0], (list, tuple)):
            raise RuntimeError(f"vLLM returned a non-vector pooling output for request {request_id}")
        result = [float(value) for value in vector]
        if self._dimensions is not None and len(result) != self._dimensions:
            raise RuntimeError(
                f"vLLM returned dimension {len(result)} for request {request_id}; " f"expected {self._dimensions}"
            )
        return result

    async def embed(self, prompts: Sequence[str]) -> list[list[float]]:
        request_base = self._request_counter
        self._request_counter += len(prompts)
        requests = [
            self._encode_one(str(prompt), f"e{self._engine_index}-{request_base + offset}")
            for offset, prompt in enumerate(prompts)
        ]
        return await asyncio.gather(*requests)

    def shutdown(self) -> None:
        self._engine.shutdown()


class ContinuousBatchEngineRouter:
    """Assign work to the engine with the least estimated queued text."""

    def __init__(self, engine_count: int) -> None:
        if int(engine_count) <= 0:
            raise ValueError("engine_count must be positive")
        self._loads = [0] * int(engine_count)
        self._next_index = 0

    def reserve(self, weight: int) -> int:
        minimum = min(self._loads)
        engine_count = len(self._loads)
        index = next(
            candidate
            for offset in range(engine_count)
            if self._loads[candidate := (self._next_index + offset) % engine_count] == minimum
        )
        self._next_index = (index + 1) % engine_count
        self._loads[index] += max(1, int(weight))
        return index

    def release(self, index: int, weight: int) -> None:
        self._loads[int(index)] = max(0, self._loads[int(index)] - max(1, int(weight)))

    def loads(self) -> list[int]:
        return list(self._loads)


class _RayAsyncEngineProxy:
    """Synchronous model interface used inside CPU Ray Data workers."""

    def __init__(self, engine_handles: Sequence[Any], router_handle: Any) -> None:
        self._engines = list(engine_handles)
        self._router = router_handle

    def embed(self, texts: Sequence[str], *, batch_size: int = 512) -> list[list[float]]:
        del batch_size  # AsyncLLM performs continuous batching across calls.
        import ray

        prompts = [str(text) for text in texts]
        weight = sum(max(1, len(prompt)) for prompt in prompts)
        engine_index = int(ray.get(self._router.reserve.remote(weight)))
        try:
            return ray.get(self._engines[engine_index].embed.remote(prompts))
        finally:
            self._router.release.remote(engine_index, weight)


class ContinuousBatchEmbedActor(AbstractOperator):
    """CPU preprocessing/finalization actor backed by shared async GPU engines."""

    def __init__(self, params: EmbedParams, engine_handles: Sequence[Any], router_handle: Any) -> None:
        super().__init__(params=params, engine_handles=engine_handles, router_handle=router_handle)
        self._kwargs = build_embed_kwargs(params)
        self._model = _RayAsyncEngineProxy(engine_handles, router_handle)

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        ensure_embedding_input_policy_for_batch(self._kwargs, data)
        return embed_text_main_text_embed(data, model=self._model, **self._kwargs)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def __call__(self, batch_df: Any) -> Any:
        return self.run(batch_df)
