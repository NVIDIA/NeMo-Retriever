# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embedding runtime helpers shared by graph-based example pipelines."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import pandas as pd

from nemo_retriever.model import resolve_embed_model
from nemo_retriever.params.models import IMAGE_MODALITIES
from nemo_retriever.text_embed.main_text_embed import TextEmbeddingConfig, create_text_embeddings_for_df


def _embed_group(
    group_df: pd.DataFrame,
    *,
    group_modality: str,
    model: Any,
    endpoint: Optional[str],
    api_key: Optional[str],
    text_column: str,
    inference_batch_size: int,
    output_column: str,
    resolved_model_name: str,
) -> pd.DataFrame:
    """Embed a single modality group via ``create_text_embeddings_for_df``."""
    embedder = None
    multimodal_embedder = None

    if endpoint is None and model is not None:
        if group_modality in IMAGE_MODALITIES:
            multimodal_embedder = model
        else:
            skip_prefix = hasattr(model, "embed_queries")

            def embedder(texts: Sequence[str]) -> Sequence[Sequence[float]]:  # noqa
                batch = texts if skip_prefix else [f"passage: {text}" for text in texts]
                vectors = model.embed(batch, batch_size=int(inference_batch_size))
                tolist = getattr(vectors, "tolist", None)
                if callable(tolist):
                    return tolist()
                return vectors  # type: ignore[return-value]

    default_remote_image_batch_size = 4
    default_local_image_batch_size = 8
    effective_batch_size = inference_batch_size
    if group_modality in IMAGE_MODALITIES:
        if endpoint is not None:
            effective_batch_size = min(inference_batch_size, default_remote_image_batch_size)
        else:
            effective_batch_size = min(inference_batch_size, default_local_image_batch_size)

    cfg = TextEmbeddingConfig(
        text_column=str(text_column),
        output_payload_column=str(output_column) if output_column else None,
        write_embedding_to_metadata=True,
        metadata_column="metadata",
        batch_size=int(effective_batch_size),
        encoding_format="float",
        input_type="passage",
        truncate="END",
        dimensions=None,
        embedding_nim_endpoint=endpoint or "http://localhost:8012/v1",
        embedding_model=resolved_model_name or "nvidia/llama-nemotron-embed-1b-v2",
        embed_modality=group_modality,
    )

    out_df, _ = create_text_embeddings_for_df(
        group_df,
        task_config={
            "api_key": api_key,
            "embedder": embedder,
            "multimodal_embedder": multimodal_embedder,
            "endpoint_url": endpoint,
            "local_batch_size": int(inference_batch_size),
        },
        transform_config=cfg,
    )
    return out_df


def embed_text_main_text_embed(
    batch_df: Any,
    *,
    model: Any = None,
    model_name: Optional[str] = None,
    embedding_endpoint: Optional[str] = None,
    embed_invoke_url: Optional[str] = None,
    api_key: Optional[str] = None,
    text_column: str = "text",
    inference_batch_size: int = 16,
    output_column: str = "text_embeddings_1b_v2",
    embedding_dim_column: str = "text_embeddings_1b_v2_dim",
    has_embedding_column: str = "text_embeddings_1b_v2_has_embedding",
    embed_modality: str = "text",
    **_: Any,
) -> Any:
    """Embed graph batches while preserving the legacy output columns."""
    if not isinstance(batch_df, pd.DataFrame):
        raise NotImplementedError("embed_text_main_text_embed currently only supports pandas.DataFrame input.")
    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be > 0")

    endpoint = (embedding_endpoint or embed_invoke_url or "").strip() or None
    if endpoint is None and model is None:
        raise ValueError("Either a local model or an embedding_endpoint must be provided.")

    resolved_model_name = resolve_embed_model(model_name)

    has_per_row_modality = "_embed_modality" in batch_df.columns
    if has_per_row_modality:
        modalities = batch_df["_embed_modality"].fillna(embed_modality).unique().tolist()
    else:
        modalities = [embed_modality]

    try:
        if len(modalities) == 1:
            out_df = _embed_group(
                batch_df,
                group_modality=modalities[0],
                model=model,
                endpoint=endpoint,
                api_key=api_key,
                text_column=text_column,
                inference_batch_size=inference_batch_size,
                output_column=output_column,
                resolved_model_name=resolved_model_name,
            )
        else:
            parts: List[pd.DataFrame] = []
            for modality in modalities:
                mask = batch_df["_embed_modality"] == modality
                group_df = batch_df.loc[mask]
                if group_df.empty:
                    continue
                part = _embed_group(
                    group_df,
                    group_modality=modality,
                    model=model,
                    endpoint=endpoint,
                    api_key=api_key,
                    text_column=text_column,
                    inference_batch_size=inference_batch_size,
                    output_column=output_column,
                    resolved_model_name=resolved_model_name,
                )
                parts.append(part)
            out_df = pd.concat(parts).sort_index()
    except BaseException as exc:
        import traceback as traceback_module

        print(f"Warning: embedding failed: {type(exc).__name__}: {exc}")
        traceback_module.print_exc()
        err_payload = {
            "embedding": None,
            "error": {"stage": "embed", "type": exc.__class__.__name__, "message": str(exc)},
        }
        out_df = batch_df.copy()
        if output_column:
            out_df[output_column] = [err_payload for _ in range(len(out_df.index))]
        out_df[embedding_dim_column] = [0 for _ in range(len(out_df.index))]
        out_df[has_embedding_column] = [False for _ in range(len(out_df.index))]
        out_df["_contains_embeddings"] = [False for _ in range(len(out_df.index))]
        return out_df

    if embedding_dim_column:

        def dim(row: pd.Series) -> int:
            metadata = row.get("metadata")
            if isinstance(metadata, dict):
                embedding = metadata.get("embedding")
                if isinstance(embedding, list):
                    return int(len(embedding))
            payload = row.get(output_column) if output_column else None
            if isinstance(payload, dict) and isinstance(payload.get("embedding"), list):
                return int(len(payload.get("embedding") or []))
            return 0

        out_df[embedding_dim_column] = out_df.apply(dim, axis=1)
    else:
        out_df[embedding_dim_column] = [0 for _ in range(len(out_df.index))]

    out_df[has_embedding_column] = [bool(int(dim) > 0) for dim in out_df[embedding_dim_column].tolist()]

    for column in ("_image_b64", "_embed_modality"):
        if column in out_df.columns:
            out_df = out_df.drop(columns=[column])

    return out_df
