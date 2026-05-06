# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared parameter coercion and building helpers used by ingest modes."""

from __future__ import annotations

from typing import Any, Dict


def coerce_params[T](params: T | None, model_cls: type[T], kwargs: dict[str, Any]) -> T:
    """Merge *params* and *kwargs* into an instance of *model_cls*.

    - If *params* is ``None``, construct from *kwargs*.
    - If *kwargs* is non-empty, apply them as overrides via ``model_copy``.
    - Otherwise return *params* unchanged.
    """
    if params is None:
        return model_cls(**kwargs)
    if kwargs:
        return params.model_copy(update=kwargs)  # type: ignore[return-value]
    return params


def build_embed_kwargs(resolved: Any, *, include_batch_tuning: bool = False) -> Dict[str, Any]:
    """Flatten an ``EmbedParams`` instance into a dict ready for actor/task kwargs.

    Merges ``runtime`` (always) and optionally ``batch_tuning`` sub-models.
    Also normalises ``embed_invoke_url`` → ``embedding_endpoint``.
    """
    exclude = {"runtime", "batch_tuning", "fused_tuning"}
    kwargs: Dict[str, Any] = {
        **resolved.model_dump(mode="python", exclude=exclude, exclude_none=True),
        **resolved.runtime.model_dump(mode="python", exclude_none=True),
    }
    if include_batch_tuning:
        kwargs.update(resolved.batch_tuning.model_dump(mode="python", exclude_none=True))

    if "embedding_endpoint" not in kwargs and kwargs.get("embed_invoke_url"):
        kwargs["embedding_endpoint"] = kwargs["embed_invoke_url"]

    return kwargs


SPLIT_CONFIG_VALID_KEYS = frozenset({"text", "html", "pdf", "audio", "image", "video"})


def resolve_split_params(
    split_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve a user-supplied split_config dict into per-key effective params.

    Returns a dict keyed by every entry in ``SPLIT_CONFIG_VALID_KEYS``. Each
    value is one of: a ``TextChunkParams`` / ``HtmlChunkParams`` instance
    (chunking enabled for that key), ``None`` (key absent — chunking off
    via the default), or ``False`` (explicit opt-out sentinel).

    Per-key values supplied by the caller may be a plain dict of
    chunk-params fields, a pre-built ``TextChunkParams`` /
    ``HtmlChunkParams`` instance (passed through verbatim), ``None``, or
    ``False``.
    """
    from nemo_retriever.params.models import HtmlChunkParams, TextChunkParams

    cfg = split_config or {}
    unknown = set(cfg) - SPLIT_CONFIG_VALID_KEYS
    if unknown:
        raise ValueError(
            f"Unknown split_config key(s): {sorted(unknown)}; " f"expected one of {sorted(SPLIT_CONFIG_VALID_KEYS)}"
        )

    out: dict[str, Any] = {}
    for key in SPLIT_CONFIG_VALID_KEYS:
        v = cfg.get(key)
        if v is None:
            out[key] = None
            continue
        if v is False:
            out[key] = False  # explicit opt-out (distinct from None / absent)
            continue
        if isinstance(v, TextChunkParams):  # HtmlChunkParams is a TextChunkParams subclass
            out[key] = v
            continue
        if isinstance(v, dict):
            cls = HtmlChunkParams if key == "html" else TextChunkParams
            out[key] = cls(**v)
            continue
        raise TypeError(
            f"split_config['{key}'] must be a TextChunkParams, dict, None, or False; got {type(v).__name__}"
        )
    return out
