# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from nemo_retriever.ingest.execution import execute_ingest_plan
from nemo_retriever.ingest.plan import ResolvedIngestPlan
from nemo_retriever.ingestor.manifest import format_branch_summary

_DRY_RUN_SECRET_FIELD_PATTERNS = ("api_key", "password", "secret", "credential", "bearer")


def _params_to_dry_run_dict(params: Any | None) -> dict[str, Any] | None:
    """Convert params to dry-run JSON data while redacting credential values.

    Dry-run output serializes the resolved Pydantic option models. Those models
    can hold endpoint credentials, so redact before printing them to the user.
    """
    if params is None:
        return None
    if hasattr(params, "model_dump"):
        data = params.model_dump(mode="json")
    elif isinstance(params, dict):
        data = dict(params)
    else:
        return {"value": str(params)}
    return _strip_secret_values(data)


def _is_dry_run_secret_field(key: Any) -> bool:
    normalized_key = str(key).lower().replace("-", "_")
    return normalized_key.endswith("token") or any(
        pattern in normalized_key for pattern in _DRY_RUN_SECRET_FIELD_PATTERNS
    )


def _strip_secret_values(value: Any) -> Any:
    """Redact secrets from dry-run reporting only."""
    if hasattr(value, "model_dump"):
        return _strip_secret_values(value.model_dump(mode="json"))
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, nested in value.items():
            if _is_dry_run_secret_field(key):
                out[key] = "<redacted>" if nested else nested
            else:
                out[key] = _strip_secret_values(nested)
        return out
    if isinstance(value, list):
        return [_strip_secret_values(item) for item in value]
    return value


def _ingest_plan_to_dry_run_data(plan: ResolvedIngestPlan) -> dict[str, Any]:
    """Return the JSON payload printed by ``retriever ingest --dry-run``."""
    return {
        "dry_run": True,
        "profile": plan.profile,
        "documents": list(plan.documents),
        "branches": [
            {
                "family": branch.family,
                "extraction_mode": branch.extraction_mode,
                "count": len(branch.input_paths),
                "input_paths": list(branch.input_paths),
            }
            for branch in plan.branches
        ],
        "branch_summary": format_branch_summary(plan.branches),
        "create_ingestor": dict(plan.create_kwargs),
        "extract": _params_to_dry_run_dict(plan.extract_params),
        "text": _params_to_dry_run_dict(plan.text_params),
        "html": _params_to_dry_run_dict(plan.html_params),
        "audio": _params_to_dry_run_dict(plan.audio_chunk_params),
        "asr": _params_to_dry_run_dict(plan.asr_params),
        "video_frames": _params_to_dry_run_dict(plan.video_frame_params),
        "video_frame_text_dedup": _params_to_dry_run_dict(plan.video_text_dedup_params),
        "audio_visual_fuse": _params_to_dry_run_dict(plan.av_fuse_params),
        "split_config": _params_to_dry_run_dict(plan.split_config),
        "dedup": _params_to_dry_run_dict(plan.dedup_params),
        "caption": _params_to_dry_run_dict(plan.caption_params),
        "embed": _params_to_dry_run_dict(plan.embed_params),
        "store": _params_to_dry_run_dict(plan.store_params),
        "vdb_upload": _params_to_dry_run_dict(plan.vdb_params),
    }


def run_ingest_workflow(
    plan: ResolvedIngestPlan,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Apply root ingest workflow policy to an already-resolved plan."""
    if dry_run:
        return _ingest_plan_to_dry_run_data(plan)

    return execute_ingest_plan(plan).to_summary_dict()
