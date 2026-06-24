# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: str
    query_file: str | None = None
    input_type: str = "pdf"
    beir_loader: str | None = None
    beir_doc_id_field: str = "pdf_page"
    beir_ks: tuple[int, ...] = (1, 3, 5, 10)
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    dataset: str
    ingest: Mapping[str, Any]
    query: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    summary_keys: tuple[str, ...]
    tags: tuple[str, ...] = ()
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunSet:
    name: str
    runs: tuple[str, ...]
    tags: tuple[str, ...] = ()
    description: str | None = None
    matrix: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
