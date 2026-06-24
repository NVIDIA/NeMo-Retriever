# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str | list[str]
    top_k: int = Field(default=10, ge=1, le=1000)


class QueryResult(BaseModel):
    hits: list[dict[str, Any]]


class QueryResponse(BaseModel):
    results: list[QueryResult]

    def hits_by_query(self, *, expected_results: int | None = None) -> list[list[dict[str, Any]]]:
        if expected_results is not None and len(self.results) != expected_results:
            raise ValueError(f"expected {expected_results} result set(s), got {len(self.results)}")
        return [result.hits for result in self.results]
