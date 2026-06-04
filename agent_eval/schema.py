# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared data contracts for the agent_eval harness.

Pure stdlib — this module (and the runner that uses it) must NOT import
``nemo_retriever`` so the harness stays copy-out portable and can run a
baseline in an environment without the codebase installed.

Two contracts live here:

* ``Query`` / ``QuerySet`` — the *public*, answer-free queries file produced by
  ``extract_queries.py`` and consumed by ``run_agent_eval.py``. It deliberately
  carries no gold answer, no relevant pages, and no eval-intent labels
  (``category``/``scoring_mode``), since e.g. a ``refusal`` label would tell the
  agent the question is a trap. Gold is recovered by the report from the
  original manifest, joined on ``query_id``.

* ``SelectedChunk`` / ``AgentOutput`` — the contract every run must emit per
  question (``output.json``): a ranked list of top-k chunks plus a synthesized
  answer. The schema is multi-modal by design; v1 only implements
  ``pdf_page``.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

QUERIES_SCHEMA_VERSION = 1
OUTPUT_SCHEMA_VERSION = 1

# Modalities a selected chunk may use. v1 implements pdf_page only.
MODALITY_PDF_PAGE = "pdf_page"
MODALITY_TEXT = "text"
MODALITY_HTML = "html"
MODALITY_AUDIO = "audio"
MODALITY_VIDEO = "video"
KNOWN_MODALITIES = frozenset({MODALITY_PDF_PAGE, MODALITY_TEXT, MODALITY_HTML, MODALITY_AUDIO, MODALITY_VIDEO})


# --------------------------------------------------------------------------- #
# Public queries contract (Script A output / Script B input)
# --------------------------------------------------------------------------- #
@dataclass
class Query:
    """One answer-free evaluation query handed to the agent."""

    query_id: str  # == manifest primary_eval_id, e.g. "vidore_v3_hr:q03:v1"
    prompt: str  # the natural-language user prompt the agent sees
    domain: str  # corpus identity; the runner mounts this domain's documents

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Query":
        return cls(query_id=str(d["query_id"]), prompt=str(d["prompt"]), domain=str(d["domain"]))


@dataclass
class QuerySet:
    """A queries.json document: provenance header + the query list.

    Provenance (source manifest path, timestamp, filters) is retained because
    the report re-reads the original manifest for gold — it is *not* answer
    data. No per-query field leaks the answer or eval intent.
    """

    source_manifest: str
    extracted_at: str
    count: int
    queries: list[Query]
    schema_version: int = QUERIES_SCHEMA_VERSION
    category_filter: list[str] | None = None
    domain_filter: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "QuerySet":
        raw = json.loads(Path(path).read_text())
        queries = [Query.from_dict(q) for q in raw.get("queries", [])]
        return cls(
            source_manifest=raw.get("source_manifest", ""),
            extracted_at=raw.get("extracted_at", ""),
            count=raw.get("count", len(queries)),
            queries=queries,
            schema_version=raw.get("schema_version", QUERIES_SCHEMA_VERSION),
            category_filter=raw.get("category_filter"),
            domain_filter=raw.get("domain_filter"),
        )

    @staticmethod
    def now_iso() -> str:
        return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Per-question output contract (Script B output / Script C input)
# --------------------------------------------------------------------------- #
@dataclass
class SelectedChunk:
    """One retrieved chunk the agent used to answer.

    ``locator`` is modality-specific:
      * pdf_page -> {"page_number": int}
      * text/html -> {"chunk_index": int} or {"char_start": int, "char_end": int}
      * audio/video -> {"start_sec": float, "end_sec": float}
    """

    rank: int
    doc_id: str
    modality: str
    locator: dict[str, Any]
    content: str = ""
    score: float | None = None


@dataclass
class AgentOutput:
    """The contract every trial must emit as output.json."""

    query_id: str
    final_answer: str
    selected_chunks: list[SelectedChunk] = field(default_factory=list)
    schema_version: int = OUTPUT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "final_answer": self.final_answer,
            "schema_version": self.schema_version,
            "selected_chunks": [asdict(c) for c in self.selected_chunks],
        }
