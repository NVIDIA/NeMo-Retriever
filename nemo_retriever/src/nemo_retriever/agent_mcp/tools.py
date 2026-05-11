# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from nemo_retriever.agent_mcp.backend import RetrieverBackend
from nemo_retriever.agent_mcp.models import AgentMcpError, AgentMcpErrorCode, EvidenceHit


def _dump(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, list):
        return [_dump(item) for item in value]
    if isinstance(value, dict):
        return {key: _dump(item) for key, item in value.items()}
    return value


class AgentMcpToolService:
    def __init__(self, backend: RetrieverBackend) -> None:
        self.backend: RetrieverBackend = backend

    def _call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return _dump(fn(*args, **kwargs))
        except AgentMcpError as exc:
            return {"error": exc.to_dict()}

    def list_collections(self) -> Any:
        return self._call(self.backend.list_collections)

    def create_collection(
        self,
        name: str = "default",
        temporary: bool = False,
        hybrid: bool = False,
    ) -> Any:
        return self._call(
            self.backend.create_collection,
            name=name,
            temporary=temporary,
            hybrid=hybrid,
        )

    def describe_collection(self, name: str = "default") -> Any:
        return self._call(self.backend.describe_collection, name=name)

    def delete_collection(self, name: str, delete_data: bool = False) -> Any:
        return self._call(self.backend.delete_collection, name=name, delete_data=delete_data)

    def start_ingestion(
        self,
        collection: str = "default",
        paths: list[str] | None = None,
        run_mode: str = "inprocess",
    ) -> Any:
        return self._call(
            self.backend.start_ingestion,
            collection=collection,
            paths=paths or [],
            run_mode=run_mode,
        )

    def get_ingestion_status(self, job_id: str) -> Any:
        return self._call(self.backend.get_ingestion_status, job_id=job_id)

    def ingest_local_paths(
        self,
        collection: str = "default",
        paths: list[str] | None = None,
        wait: bool = True,
        timeout_s: float | None = None,
        run_mode: str = "inprocess",
    ) -> Any:
        return self._call(
            self.backend.ingest_local_paths,
            collection=collection,
            paths=paths or [],
            wait=wait,
            timeout_s=timeout_s,
            run_mode=run_mode,
        )

    def query_collection(
        self,
        collection: str = "default",
        query: str = "",
        top_k: int = 10,
        hybrid: bool = False,
        rerank: bool = False,
        filters: dict[str, Any] | None = None,
    ) -> Any:
        return self._call(
            self.backend.query_collection,
            collection=collection,
            query=query,
            top_k=top_k,
            hybrid=hybrid,
            rerank=rerank,
            filters=filters,
        )

    def rerank_results(
        self,
        query: str,
        hits: list[dict[str, Any]],
        top_n: int | None = None,
    ) -> Any:
        def rerank_validated_hits() -> list[EvidenceHit]:
            try:
                validated_hits = [EvidenceHit.model_validate(hit) for hit in hits]
            except ValidationError as exc:
                raise AgentMcpError(
                    AgentMcpErrorCode.BACKEND_ERROR,
                    "Invalid rerank hit payload.",
                    details={"errors": exc.errors()},
                ) from exc
            return self.backend.rerank_results(query=query, hits=validated_hits, top_n=top_n)

        return self._call(rerank_validated_hits)
