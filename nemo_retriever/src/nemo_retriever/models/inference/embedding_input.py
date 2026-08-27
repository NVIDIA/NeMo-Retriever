# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic admission and overflow splitting for text embedding rows."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from nemo_retriever.common.modality.txt.tokenizer_provider import (
    ChunkTokenizer,
    load_chunk_tokenizer,
)
from nemo_retriever.models import resolve_embed_model
from nemo_retriever.models.embed_model_spec import resolve_embed_model_spec
from nemo_retriever.models.hf_model_registry import HF_MODEL_REVISIONS

_OVERLENGTH_COLUMN = "_embedding_input_overlength"
_SPLIT_PARENT_COLUMN = "_embedding_input_split_parent"
_SPLIT_CHILD_COLUMN = "_embedding_input_split_child"
_INTERNAL_ACCOUNTING_COLUMNS = (
    _OVERLENGTH_COLUMN,
    _SPLIT_PARENT_COLUMN,
    _SPLIT_CHILD_COLUMN,
)


def _deep_copy_row(row: pd.Series) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value) if isinstance(value, (dict, list)) else value for key, value in row.to_dict().items()
    }


def _embedding_text(row: pd.Series, text_column: str) -> tuple[str, str] | None:
    for column in dict.fromkeys((text_column, "text", "content", "chunk", "page_text")):
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            return column, value
    return None


def _stable_json(value: Any) -> str:
    def normalize(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): normalize(nested) for key, nested in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(nested) for nested in item]
        if hasattr(item, "tolist") and callable(item.tolist):
            return normalize(item.tolist())
        if hasattr(item, "item") and callable(item.item):
            return normalize(item.item())
        if isinstance(item, float) and item.is_integer():
            return int(item)
        return item

    return json.dumps(
        normalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if pd.api.types.is_scalar(value) and pd.isna(value):
            continue
        return value
    return None


def _parent_id(row: dict[str, Any], text: str) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    content_metadata = metadata.get("content_metadata") if isinstance(metadata.get("content_metadata"), dict) else {}
    identity = {
        "source": _first_present(
            metadata.get("source_path"),
            row.get("path"),
            row.get("source_id"),
            row.get("source"),
        ),
        "page": _first_present(row.get("_page_number"), row.get("page_number"), metadata.get("page_number")),
        "content_type": _first_present(
            row.get("_content_type"),
            row.get("content_type"),
            content_metadata.get("type"),
        ),
        "element_id": _first_present(
            row.get("element_id"),
            row.get("id"),
            metadata.get("id"),
            content_metadata.get("id"),
        ),
        "bbox": _first_present(
            row.get("_bbox_xyxy_norm"),
            row.get("bbox_xyxy_norm"),
            content_metadata.get("bbox_xyxy_norm"),
        ),
        "source_chunk_index": metadata.get("chunk_index"),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    return hashlib.sha256(_stable_json(identity).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EmbeddingInputPolicy:
    """Prepare text rows so every formatted prompt fits the model input limit."""

    tokenizer: ChunkTokenizer
    max_tokens: int
    prefix: str
    overlap_tokens: int = 0

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("Embedding input max_tokens must be positive")
        if self.overlap_tokens < 0:
            raise ValueError("Embedding input overlap_tokens must be nonnegative")

    def _formatted_token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(f"{self.prefix}{text}", add_special_tokens=True))

    def _formatted_token_counts(self, texts: list[str]) -> list[int]:
        encode_batch = getattr(self.tokenizer, "encode_batch", None)
        if callable(encode_batch):
            return [
                len(token_ids)
                for token_ids in encode_batch(
                    [f"{self.prefix}{text}" for text in texts],
                    add_special_tokens=True,
                )
            ]
        return [self._formatted_token_count(text) for text in texts]

    def _largest_fitting_end(self, token_ids: list[int], start: int) -> int:
        low = start + 1
        high = min(len(token_ids), start + self.max_tokens)
        best = start
        while low <= high:
            candidate = (low + high) // 2
            text = self.tokenizer.decode(token_ids[start:candidate], skip_special_tokens=False)
            if text and self._formatted_token_count(text) <= self.max_tokens:
                best = candidate
                low = candidate + 1
            else:
                high = candidate - 1
        if best == start:
            raise ValueError(
                "Embedding input cannot fit one content token after applying the model prefix and special tokens"
            )
        return best

    def _split(self, text: str) -> list[tuple[str, int, int, int]]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunks: list[tuple[str, int, int, int]] = []
        start = 0
        while start < len(token_ids):
            end = self._largest_fitting_end(token_ids, start)
            chunk = self.tokenizer.decode(token_ids[start:end], skip_special_tokens=False)
            formatted_tokens = self._formatted_token_count(chunk)
            chunks.append((chunk, start, end, formatted_tokens))
            if end >= len(token_ids):
                break
            next_start = end - self.overlap_tokens
            start = next_start if next_start > start else end
        return chunks

    def prepare(
        self,
        frame: pd.DataFrame,
        *,
        text_column: str = "text",
        default_modality: str = "text",
    ) -> pd.DataFrame:
        """Return rows ready for embedding, expanding only overlength text inputs."""
        if frame.empty:
            return frame.copy()

        if "_embed_modality" in frame.columns:
            modalities = [str(_first_present(value, default_modality)) for value in frame["_embed_modality"].tolist()]
            if all(modality != "text" for modality in modalities):
                return frame.copy().reset_index(drop=True)
        elif default_modality != "text":
            return frame.copy().reset_index(drop=True)

        if text_column in frame.columns:
            texts = frame[text_column].tolist()
            all_text = all(isinstance(text, str) and text.strip() for text in texts)
            all_text_modalities = "_embed_modality" not in frame.columns or all(
                modality == "text" for modality in modalities
            )
            if all_text and all_text_modalities:
                parent_token_counts = self._formatted_token_counts(texts)
                if all(token_count <= self.max_tokens for token_count in parent_token_counts):
                    return frame.copy().reset_index(drop=True)

        rows = [row for _, row in frame.iterrows()]
        selected_inputs: list[tuple[str, str] | None] = []
        text_positions: list[int] = []
        texts: list[str] = []
        for position, row in enumerate(rows):
            modality = str(_first_present(row.get("_embed_modality"), default_modality))
            selected = _embedding_text(row, text_column) if modality == "text" else None
            selected_inputs.append(selected)
            if selected is not None:
                text_positions.append(position)
                texts.append(selected[1])

        parent_token_counts: list[int | None] = [None] * len(rows)
        for position, token_count in zip(text_positions, self._formatted_token_counts(texts)):
            parent_token_counts[position] = token_count

        if not any(token_count is not None and token_count > self.max_tokens for token_count in parent_token_counts):
            return frame.copy().reset_index(drop=True)

        prepared: list[dict[str, Any]] = []
        for row, selected, parent_tokens in zip(rows, selected_inputs, parent_token_counts):
            row_copy = _deep_copy_row(row)
            if selected is None:
                row_copy["embedding_input_action"] = "not_applicable"
                row_copy[_OVERLENGTH_COLUMN] = False
                row_copy[_SPLIT_PARENT_COLUMN] = False
                row_copy[_SPLIT_CHILD_COLUMN] = False
                prepared.append(row_copy)
                continue

            selected_column, text = selected
            assert parent_tokens is not None
            if parent_tokens <= self.max_tokens:
                row_copy["embedding_input_action"] = "none"
                row_copy["embedding_input_token_count"] = parent_tokens
                row_copy["embedding_input_parent_token_count"] = parent_tokens
                row_copy[_OVERLENGTH_COLUMN] = False
                row_copy[_SPLIT_PARENT_COLUMN] = False
                row_copy[_SPLIT_CHILD_COLUMN] = False
                prepared.append(row_copy)
                continue

            chunks = self._split(text)
            parent_id = _parent_id(row_copy, text)
            for chunk_index, (chunk, start, end, formatted_tokens) in enumerate(chunks):
                child = copy.deepcopy(row_copy)
                child[selected_column] = chunk
                if selected_column == "text" and "content" in child:
                    child["content"] = chunk
                metadata = child.get("metadata")
                if not isinstance(metadata, dict):
                    metadata = {}
                    child["metadata"] = metadata
                chunk_id = hashlib.sha256(f"{parent_id}\0{start}\0{end}".encode("utf-8")).hexdigest()
                metadata.update(
                    {
                        "content": chunk,
                        "embedding_parent_id": parent_id,
                        "embedding_chunk_id": chunk_id,
                        "embedding_chunk_index": chunk_index,
                        "embedding_chunk_count": len(chunks),
                        "embedding_chunk_start_token": start,
                        "embedding_chunk_end_token": end,
                    }
                )
                child["embedding_input_action"] = "split"
                child["embedding_input_token_count"] = formatted_tokens
                child["embedding_input_parent_token_count"] = parent_tokens
                child[_OVERLENGTH_COLUMN] = chunk_index == 0
                child[_SPLIT_PARENT_COLUMN] = chunk_index == 0
                child[_SPLIT_CHILD_COLUMN] = True
                prepared.append(child)

        return pd.DataFrame(prepared).reset_index(drop=True)


def resolve_embedding_input_policy(
    model_id: str,
    *,
    configured_max_tokens: int,
    input_type: str,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> EmbeddingInputPolicy:
    """Resolve a model-pinned input policy shared by local and remote adapters."""
    if configured_max_tokens <= 0:
        raise ValueError("Configured embedding max length must be positive")
    spec = resolve_embed_model_spec(model_id, revision=revision, hf_cache_dir=cache_dir)
    supported_max = spec.max_input_tokens or configured_max_tokens
    effective_max = min(configured_max_tokens, supported_max)
    prefix = spec.query_prefix if str(input_type).strip().lower() == "query" else spec.document_prefix
    tokenizer = load_chunk_tokenizer(
        spec.model_id,
        cache_dir=cache_dir,
        revision=spec.revision,
    )
    return EmbeddingInputPolicy(tokenizer=tokenizer, max_tokens=effective_max, prefix=prefix)


def resolve_known_embedding_input_policy(
    *,
    model_name: str | None,
    configured_max_tokens: int,
    input_type: str,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> EmbeddingInputPolicy:
    """Resolve input protection for a pinned or local model, failing closed otherwise."""
    model_id = resolve_embed_model(model_name)
    if model_id not in HF_MODEL_REVISIONS and not Path(model_id).expanduser().is_dir() and revision is None:
        raise ValueError(
            f"Embedding model {model_id!r} is not revision-pinned, so the embedding stage cannot enforce "
            "its tokenizer, prefix, and input limit. Use a registered model, a local checkpoint, or set an "
            "immutable embed_model_revision."
        )
    return resolve_embedding_input_policy(
        model_id,
        configured_max_tokens=configured_max_tokens,
        input_type=input_type,
        revision=revision,
        cache_dir=cache_dir,
    )


def configure_embedding_input_policy(kwargs: dict[str, Any]) -> EmbeddingInputPolicy:
    """Resolve and install the shared input policy for an embedding actor."""
    input_type = str(kwargs.get("input_type", "passage"))
    configured_max_tokens = (
        int(kwargs.get("query_max_length", 128))
        if input_type.strip().lower() == "query"
        else int(kwargs.get("max_length", 8192))
    )
    policy = resolve_known_embedding_input_policy(
        model_name=kwargs.get("embed_model_name") or kwargs.get("model_name"),
        configured_max_tokens=configured_max_tokens,
        input_type=input_type,
        revision=kwargs.get("embed_model_revision"),
        cache_dir=kwargs.get("hf_cache_dir"),
    )
    kwargs["embedding_input_policy"] = policy
    return policy


__all__ = [
    "EmbeddingInputPolicy",
    "configure_embedding_input_policy",
    "resolve_embedding_input_policy",
    "resolve_known_embedding_input_policy",
    "_INTERNAL_ACCOUNTING_COLUMNS",
]
