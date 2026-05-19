# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `pdf/stage.py` JSON-sidecar helpers — specifically the
recursive strip used by ``--compact-json`` to remove placeholder fields
that bloat sidecars for programmatic consumers."""

from __future__ import annotations

from nemo_retriever.pdf.stage import _strip_null_empty


def test_strip_drops_nulls_and_empty_collections():
    assert _strip_null_empty({"a": None, "b": "", "c": [], "d": {}, "e": "keep"}) == {"e": "keep"}


def test_strip_drops_negative_one_placeholders():
    # `-1` is the conventional "not applicable" sentinel across
    # content_metadata.hierarchy, start_time/end_time, partition_id, etc.
    assert _strip_null_empty({"partition_id": -1, "page": 3}) == {"page": 3}


def test_strip_keeps_zero_and_false():
    # 0 and False are real values, not placeholders.
    assert _strip_null_empty({"count": 0, "flag": False, "x": -1}) == {"count": 0, "flag": False}


def test_strip_recurses_into_nested_dicts():
    obj = {
        "outer": {"inner_keep": "v", "inner_drop": None},
        "all_empty": {"a": None, "b": ""},
    }
    # Inner dict that becomes empty after recursion gets dropped from parent.
    assert _strip_null_empty(obj) == {"outer": {"inner_keep": "v"}}


def test_strip_filters_lists_and_drops_emptied_lists():
    # The classic text_location placeholder [-1, -1, -1, -1] collapses to [].
    assert _strip_null_empty({"text_location": [-1, -1, -1, -1], "bbox": [10, 20, 30, 40]}) == {
        "bbox": [10, 20, 30, 40]
    }


def test_strip_preserves_scalar_strings_and_numbers():
    assert _strip_null_empty({"text": "hello", "n": 42}) == {"text": "hello", "n": 42}
