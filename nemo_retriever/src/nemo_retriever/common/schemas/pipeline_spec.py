# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-request pipeline configuration shipped from client → server.

``PipelineSpec`` is the wire-format mirror of the fluent state accumulated by
``ServiceIngestor``. Today's worker pipeline is fixed at service startup
(``pipeline_executor._make_work_fn`` bakes ``ExtractParams`` / ``EmbedParams``
into a closure); ``PipelineSpec`` lets a client request **different** stage
configuration on a per-document basis while the server retains absolute
control over trust-sensitive fields (NIM endpoint URLs, API keys, storage
allowlists, webhook destinations, …).

The contract is:

* Clients populate fields they want to *override* — fields left ``None``
  defer to ``ServiceConfig.nim_endpoints`` and the bundled defaults.
* The server merges ``ServiceConfig.nim_endpoints`` (URLs + api_key)
  **after** validating the client spec, so a tenant cannot redirect the
  pipeline's GPU traffic.
* ``stage_order`` controls **post-extraction** stage ordering only;
  extraction is always first.

The spec is transported inside the existing ``metadata`` form field of
``POST /v1/ingest`` so no breaking API change is required.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import ConfigDict, Field

from nemo_retriever.common.schemas.base import RichModel


ExtractionMode = Literal["pdf", "image", "auto", "text", "html", "audio"]
StageName = Literal["extract", "dedup", "caption", "embed", "store", "filter", "webhook"]


class PdfSplitSpec(RichModel):
    """Per-request PDF chunking config (``pages_per_chunk`` only for now).

    Mirrors :meth:`ServiceIngestor.pdf_split_config`. The server uses
    ``pages_per_chunk`` to refine the realtime-vs-batch routing decision
    in :func:`_route_by_page_count`.
    """

    model_config = ConfigDict(extra="forbid")

    pages_per_chunk: int = Field(default=32, ge=1, le=4096)


class PipelineSpec(RichModel):
    """Wire-format representation of fluent pipeline state.

    Each ``*_params`` field is an opaque dict matching the corresponding
    Pydantic params model (``ExtractParams``, ``EmbedParams``, …). The
    worker reconstructs the typed model after server-side validation.

    Fields are intentionally permissive (``dict[str, Any]``) so the wire
    format does not need to track every params-model field change in
    lock-step. The :mod:`nemo_retriever.service.policy` module is the
    layer that decides which keys / values are admissible.
    """

    model_config = ConfigDict(extra="forbid")

    # Extraction stage selector (mirrors GraphIngestor._extraction_mode).
    extraction_mode: ExtractionMode = "auto"

    extract_params: Optional[dict[str, Any]] = None
    embed_params: Optional[dict[str, Any]] = None
    dedup_params: Optional[dict[str, Any]] = None
    caption_params: Optional[dict[str, Any]] = None
    store_params: Optional[dict[str, Any]] = None
    vdb_upload_params: Optional[dict[str, Any]] = None
    webhook_params: Optional[dict[str, Any]] = None

    split_config: Optional[dict[str, Any]] = None
    pdf_split: Optional[PdfSplitSpec] = None

    stage_order: list[StageName] = Field(default_factory=list)

    def is_empty(self) -> bool:
        """``True`` when the client supplied no overrides and no stage_order.

        Used by the worker to short-circuit to the legacy
        baked-at-startup pipeline path.
        """
        return (
            self.extraction_mode in ("pdf", "auto")
            and self.extract_params is None
            and self.embed_params is None
            and self.dedup_params is None
            and self.caption_params is None
            and self.store_params is None
            and self.vdb_upload_params is None
            and self.webhook_params is None
            and self.split_config is None
            and self.pdf_split is None
            and not self.stage_order
        )
