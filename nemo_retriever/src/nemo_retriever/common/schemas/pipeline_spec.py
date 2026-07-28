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


class EndpointOverrides(RichModel):
    """Per-request model-endpoint overrides shipped from client → server.

    Endpoint URLs, model names, and API keys are normally *server-owned*:
    they are baked into the worker pipeline from ``ServiceConfig.nim_endpoints``
    at startup and the :mod:`nemo_retriever.common.policy` denylist rejects
    any attempt to smuggle them in through the ordinary ``*_params`` blocks.

    This model is the **explicit, audited channel** for a client to point a
    submitted job at a *different* model deployment than the cluster default —
    for example a purpose-built VLM for captioning or an alternative embedding
    NIM. It is honored **only** when the operator has opted in via
    ``pipeline_overrides.endpoint_overrides`` in ``retriever-service.yaml``;
    otherwise the server rejects the request with HTTP 403. Because it is a
    dedicated field rather than free-form params, the denylist that protects
    ``embed_params`` / ``caption_params`` stays fully intact.

    Fields left ``None`` fall back to the server-configured default for that
    stage. ``embed_*`` retarget the embedding NIM used by the ``embed`` stage;
    ``caption_*`` retarget the VLM used by the ``caption`` stage.
    ``embed_api_key`` / ``caption_api_key`` are optional credentials for
    those stages; ``api_key`` remains as a legacy fallback when only one
    stage is overridden (never replaces the server key for stages left at
    their defaults).
    """

    model_config = ConfigDict(extra="forbid")

    embed_invoke_url: Optional[str] = Field(
        default=None,
        description="Remote embedding NIM URL to use for the embed stage instead of the cluster default.",
    )
    embed_model_name: Optional[str] = Field(
        default=None,
        description="Model identifier passed to the overridden embedding endpoint.",
    )
    embed_model_provider_prefix: Optional[str] = Field(
        default=None,
        description="Optional LiteLLM provider prefix prepended to embed_model_name.",
    )
    caption_invoke_url: Optional[str] = Field(
        default=None,
        description="Remote VLM (caption) endpoint URL to use for the caption stage.",
    )
    caption_model_name: Optional[str] = Field(
        default=None,
        description="Model identifier passed to the overridden caption (VLM) endpoint.",
    )
    embed_api_key: Optional[str] = Field(
        default=None,
        description="Optional API key for the client-overridden embedding endpoint only.",
    )
    caption_api_key: Optional[str] = Field(
        default=None,
        description="Optional API key for the client-overridden caption (VLM) endpoint only.",
    )
    api_key: Optional[str] = Field(
        default=None,
        description=(
            "Legacy fallback API key when a single overridden stage supplies "
            "no stage-specific key. Prefer embed_api_key / caption_api_key "
            "when both stages need credentials."
        ),
    )

    def is_empty(self) -> bool:
        """``True`` when the client did not set any override field."""
        return not any(
            (
                self.embed_invoke_url,
                self.embed_model_name,
                self.embed_model_provider_prefix,
                self.caption_invoke_url,
                self.caption_model_name,
                self.embed_api_key,
                self.caption_api_key,
                self.api_key,
            )
        )


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

    # Per-request model-endpoint overrides. Honored only when the operator
    # opted in via ``pipeline_overrides.endpoint_overrides``; otherwise the
    # policy layer rejects any non-empty value with HTTP 403.
    endpoint_overrides: Optional[EndpointOverrides] = None

    stage_order: list[StageName] = Field(default_factory=list)
    result_schema: Literal["legacy", "compact"] = Field(
        default="legacy",
        description=(
            "Result row schema for service return_results/save_to_disk. "
            "'legacy' preserves GraphIngestor.ingest() columns with bulky values stripped; "
            "'compact' returns the future compact public schema."
        ),
    )
    return_embeddings: bool = Field(
        default=False,
        description="Include embedding payload values in legacy transport rows.",
    )
    return_images: bool = Field(
        default=False,
        description="Include raw image payload values in legacy transport rows.",
    )

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
            and (self.endpoint_overrides is None or self.endpoint_overrides.is_empty())
            and not self.stage_order
            and self.result_schema == "legacy"
            and not self.return_embeddings
            and not self.return_images
        )
