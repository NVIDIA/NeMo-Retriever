# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server-side trust boundary for per-request pipeline overrides.

The :class:`~nemo_retriever.service.models.pipeline_spec.PipelineSpec`
shipped by a client cannot be applied blindly: it would let a tenant
redirect NIM traffic, exfiltrate to arbitrary HTTP endpoints, or write
files to attacker-chosen paths. This module centralises that
trust-boundary logic so the router, worker, and ``pipeline-config``
introspection endpoint all see the same picture of what is allowed.

Two layers:

1. **Denylist of endpoint / api_key fields**. The server *always* owns
   these — the only way to influence them is to update
   ``ServiceConfig.nim_endpoints``.
2. **Per-stage allowlist of param keys**. Defaults to a conservative set
   of "shape" fields (chunk sizes, batch sizes, output flags). Operators
   can widen the allowlist via the new ``pipeline_overrides`` section of
   ``retriever-service.yaml``.

The module is pure-Python: it takes plain dicts (the wire format) and
returns plain dicts (the validated, server-merged form). No FastAPI
imports, so unit tests can exercise it directly.
"""

from __future__ import annotations

from typing import Any

from nemo_retriever.service.models.pipeline_spec import PipelineSpec


# ----------------------------------------------------------------------
# Trust-sensitive field names — never accept these from a client
# ----------------------------------------------------------------------

_DENYLIST_KEY_SUBSTRINGS: tuple[str, ...] = (
    "invoke_url",
    "endpoint",
    "endpoint_url",
    "api_key",
    "api_base",
    "auth_token",
    "function_id",
    "audio_endpoints",
    "remote_invoke",
    "callback_url",
    "lancedb_uri",
    "storage_uri",
    "vectordb_url",
    "embed_invoke_url",
    "embedding_endpoint",
    "page_elements_invoke_url",
    "ocr_invoke_url",
    "table_structure_invoke_url",
    "graphic_elements_invoke_url",
    "nemotron_parse_invoke_url",
)


def _is_trust_sensitive(key: str) -> bool:
    """Return ``True`` when ``key`` matches a denied substring."""
    lower = key.lower()
    return any(needle in lower for needle in _DENYLIST_KEY_SUBSTRINGS)


# ----------------------------------------------------------------------
# Default per-stage allowlist — overridable from YAML
# ----------------------------------------------------------------------

_DEFAULT_ALLOWED_EXTRACT_KEYS: frozenset[str] = frozenset(
    {
        "extract_text",
        "extract_images",
        "extract_tables",
        "extract_charts",
        "extract_infographics",
        "extract_page_as_image",
        "method",
        "use_table_structure",
        "table_output_format",
        "use_graphic_elements",
        "dpi",
        "image_format",
        "jpeg_quality",
        "render_mode",
        "inference_batch_size",
        "ocr_version",
        "output_column",
        "num_detections_column",
        "counts_by_label_column",
    }
)

_DEFAULT_ALLOWED_EMBED_KEYS: frozenset[str] = frozenset(
    {
        "input_type",
        "embed_modality",
        "embed_granularity",
        "text_elements_modality",
        "structured_elements_modality",
        "text_column",
        "inference_batch_size",
        "output_column",
        "embedding_dim_column",
        "has_embedding_column",
        "embed_output_column",
        "embed_inference_batch_size",
        "query_max_length",
        "dimensions",
    }
)

_DEFAULT_ALLOWED_DEDUP_KEYS: frozenset[str] = frozenset({"content_hash", "bbox_iou", "iou_threshold"})

_DEFAULT_ALLOWED_SPLIT_KEYS: frozenset[str] = frozenset(
    {"max_tokens", "overlap_tokens", "tokenizer_model_id", "encoding"}
)

# Stages we explicitly disallow in Phase 1. Each is wired up in a later
# phase under its own configuration.
_PHASE_1_STAGES: frozenset[str] = frozenset({"extract", "dedup", "embed", "filter"})


class PolicyError(ValueError):
    """Raised by :func:`validate_pipeline_spec` when a client spec is rejected.

    Carries an HTTP-friendly *status_code* (400 for malformed input, 403
    for a policy denial) and a human-readable *detail* string suitable
    for the FastAPI exception handler.
    """

    def __init__(self, detail: str, *, status_code: int = 400) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


# ----------------------------------------------------------------------
# Public configuration object — mirrors ServiceConfig.pipeline_overrides
# ----------------------------------------------------------------------


class PipelineOverridesPolicy:
    """Operator-tunable view of what client overrides are admissible.

    Operators set this via ``pipeline_overrides`` in ``retriever-service.yaml``;
    callers in the router/worker get a single :class:`PipelineOverridesPolicy`
    instance from ``ServiceConfig`` and pass it to :func:`validate_pipeline_spec`.
    """

    def __init__(
        self,
        *,
        mode: str = "allow_list",
        allowed_stages: frozenset[str] | None = None,
        extra_extract_keys: frozenset[str] = frozenset(),
        extra_embed_keys: frozenset[str] = frozenset(),
        extra_dedup_keys: frozenset[str] = frozenset(),
        extra_split_keys: frozenset[str] = frozenset(),
    ) -> None:
        if mode not in {"reject", "allow_list", "allow_all"}:
            raise ValueError(
                f"pipeline_overrides.mode must be one of 'reject', 'allow_list', 'allow_all'; got {mode!r}"
            )
        self.mode = mode
        self.allowed_stages = allowed_stages if allowed_stages is not None else _PHASE_1_STAGES
        self.allowed_extract_keys = _DEFAULT_ALLOWED_EXTRACT_KEYS | extra_extract_keys
        self.allowed_embed_keys = _DEFAULT_ALLOWED_EMBED_KEYS | extra_embed_keys
        self.allowed_dedup_keys = _DEFAULT_ALLOWED_DEDUP_KEYS | extra_dedup_keys
        self.allowed_split_keys = _DEFAULT_ALLOWED_SPLIT_KEYS | extra_split_keys

    def describe(self) -> dict[str, Any]:
        """Render the policy as a JSON-safe dict for the introspection endpoint."""
        return {
            "mode": self.mode,
            "allowed_stages": sorted(self.allowed_stages),
            "allowed_extract_keys": sorted(self.allowed_extract_keys),
            "allowed_embed_keys": sorted(self.allowed_embed_keys),
            "allowed_dedup_keys": sorted(self.allowed_dedup_keys),
            "allowed_split_keys": sorted(self.allowed_split_keys),
            "denied_key_substrings": sorted(_DENYLIST_KEY_SUBSTRINGS),
        }


# ----------------------------------------------------------------------
# Validation entry point
# ----------------------------------------------------------------------


def _scrub_trust_sensitive(params: dict[str, Any] | None, stage: str) -> dict[str, Any] | None:
    """Strip trust-sensitive keys from a params dict, raising on hit.

    We *reject* (rather than silently drop) so the client gets a clear
    error and knows their request will not behave as written.
    """
    if params is None:
        return None
    bad = [k for k in params if _is_trust_sensitive(k)]
    if bad:
        raise PolicyError(
            f"{stage}_params: rejected trust-sensitive keys {bad!r}. "
            "Endpoint URLs and API keys are configured via the server's "
            "nim_endpoints section and cannot be overridden per-request.",
            status_code=403,
        )
    return params


def _enforce_allowlist(
    params: dict[str, Any] | None,
    allowed: frozenset[str],
    stage: str,
    *,
    mode: str,
) -> dict[str, Any] | None:
    """Apply per-stage allowlist enforcement based on the policy mode."""
    if params is None or mode == "allow_all":
        return params
    extras = [k for k in params if k not in allowed]
    if not extras:
        return params
    if mode == "reject":
        raise PolicyError(
            f"{stage}_params: client overrides are disabled (pipeline_overrides.mode='reject'). "
            f"Offending keys: {extras!r}.",
            status_code=403,
        )
    raise PolicyError(
        f"{stage}_params: keys {extras!r} are not in the allow_list. "
        "Ask the service operator to widen pipeline_overrides.extra_*_keys "
        "or remove them from your request.",
        status_code=403,
    )


def validate_pipeline_spec(
    spec: PipelineSpec | None,
    policy: PipelineOverridesPolicy,
) -> PipelineSpec | None:
    """Return a sanitized copy of *spec* or raise :class:`PolicyError`.

    Returns ``None`` when the spec is missing or empty (so the worker can
    short-circuit to the legacy startup-baked pipeline).
    """
    if spec is None or spec.is_empty():
        return None

    if policy.mode == "reject":
        raise PolicyError(
            "Per-request pipeline overrides are disabled on this service "
            "(pipeline_overrides.mode='reject'). Update retriever-service.yaml "
            "to enable them.",
            status_code=403,
        )

    for stage_name in spec.stage_order:
        if stage_name not in policy.allowed_stages:
            raise PolicyError(
                f"stage {stage_name!r} is not in pipeline_overrides.allowed_stages. "
                f"Allowed in this phase: {sorted(policy.allowed_stages)}.",
                status_code=403,
            )

    # Reject groups B/C/D stages outright until their phase lands.
    for forbidden in ("caption_params", "store_params", "vdb_upload_params", "webhook_params"):
        if getattr(spec, forbidden) is not None:
            raise PolicyError(
                f"{forbidden} overrides are not yet supported in service run_mode "
                "(scheduled for a future phase). Configure the corresponding "
                "stage via retriever-service.yaml instead.",
                status_code=501,
            )

    # Endpoint/API-key denylist applies to every params block, even ones
    # we currently accept — defense in depth in case a new field name
    # slips into the allowlist without being audited.
    _scrub_trust_sensitive(spec.extract_params, "extract")
    _scrub_trust_sensitive(spec.embed_params, "embed")
    _scrub_trust_sensitive(spec.dedup_params, "dedup")
    _scrub_trust_sensitive(spec.split_config, "split")

    _enforce_allowlist(spec.extract_params, policy.allowed_extract_keys, "extract", mode=policy.mode)
    _enforce_allowlist(spec.embed_params, policy.allowed_embed_keys, "embed", mode=policy.mode)
    _enforce_allowlist(spec.dedup_params, policy.allowed_dedup_keys, "dedup", mode=policy.mode)
    if spec.split_config is not None:
        for source_type, cfg in spec.split_config.items():
            if not isinstance(cfg, dict):
                continue
            _enforce_allowlist(cfg, policy.allowed_split_keys, f"split[{source_type}]", mode=policy.mode)

    return spec
