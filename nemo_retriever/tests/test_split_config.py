# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for resolve_split_params."""

from __future__ import annotations

import pytest

from nemo_retriever.params import HtmlChunkParams, TextChunkParams
from nemo_retriever.params.utils import resolve_split_params


def test_resolve_split_params_behavior():
    """Single omnibus test: defaults, dict overrides, False off-switch, unknown key validation."""
    # Chunking is opt-in on every key — None / missing keys resolve to None.
    out = resolve_split_params(None)
    assert all(out[key] is None for key in ("text", "html", "pdf", "audio", "image", "video"))

    # Dict override builds the appropriate chunk-params instance per key.
    out = resolve_split_params({"pdf": {"max_tokens": 256}, "html": {"max_tokens": 333}, "text": False})
    assert isinstance(out["pdf"], TextChunkParams)
    assert out["pdf"].max_tokens == 256
    assert isinstance(out["html"], HtmlChunkParams)
    assert out["html"].max_tokens == 333
    # Explicit False is preserved as a sentinel for "off".
    assert out["text"] is False

    # Pre-built params instances pass through verbatim.
    prebuilt = TextChunkParams(max_tokens=512)
    out = resolve_split_params({"pdf": prebuilt})
    assert out["pdf"] is prebuilt

    # Unknown top-level key raises.
    with pytest.raises(ValueError, match="Unknown split_config key"):
        resolve_split_params({"pptx": {"max_tokens": 256}})
