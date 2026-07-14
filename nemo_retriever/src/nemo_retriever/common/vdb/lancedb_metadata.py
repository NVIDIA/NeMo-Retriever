# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned NeMo Retriever metadata stored on LanceDB table schemas."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from nemo_retriever.version import get_version_info


INDEX_FORMAT_VERSION = "1"

_INDEX_FORMAT_VERSION_KEY = b"nemo_retriever.index_format_version"
_PRODUCER_VERSION_KEY = b"nemo_retriever.producer_version"
_EMBEDDING_MODEL_NAME_KEY = b"nemo_retriever.embedding_model_name"
_RETRIEVAL_MODE_KEY = b"nemo_retriever.retrieval_mode"
_LEGACY_RETRIEVAL_MODE_KEY = b"retrieval_mode"


@dataclass(frozen=True)
class LanceIndexMetadata:
    index_format_version: str | None
    producer_version: str | None
    embedding_model_name: str | None
    retrieval_mode: str | None


def with_index_metadata(
    schema: pa.Schema,
    *,
    retrieval_mode: str,
    embedding_model_name: str | None,
) -> pa.Schema:
    """Return ``schema`` with current, non-secret index provenance metadata."""
    metadata = dict(schema.metadata or {})
    metadata[_INDEX_FORMAT_VERSION_KEY] = INDEX_FORMAT_VERSION.encode("utf-8")
    metadata[_PRODUCER_VERSION_KEY] = get_version_info()["full_version"].encode("utf-8")
    if embedding_model_name:
        metadata[_EMBEDDING_MODEL_NAME_KEY] = str(embedding_model_name).encode("utf-8")
    encoded_mode = str(retrieval_mode).encode("utf-8")
    metadata[_RETRIEVAL_MODE_KEY] = encoded_mode
    metadata[_LEGACY_RETRIEVAL_MODE_KEY] = encoded_mode
    return schema.with_metadata(metadata)


def read_index_metadata(schema: pa.Schema) -> LanceIndexMetadata:
    """Read NeMo Retriever index metadata without rejecting legacy schemas."""
    metadata = schema.metadata or {}

    def _text(key: bytes) -> str | None:
        value = metadata.get(key)
        if value is None:
            return None
        decoded = value.decode("utf-8", errors="replace").strip()
        return decoded or None

    return LanceIndexMetadata(
        index_format_version=_text(_INDEX_FORMAT_VERSION_KEY),
        producer_version=_text(_PRODUCER_VERSION_KEY),
        embedding_model_name=_text(_EMBEDDING_MODEL_NAME_KEY),
        retrieval_mode=_text(_RETRIEVAL_MODE_KEY) or _text(_LEGACY_RETRIEVAL_MODE_KEY),
    )
