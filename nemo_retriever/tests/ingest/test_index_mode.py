# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_retriever.ingest.index_mode import resolve_ingest_index_mode


@pytest.mark.parametrize(
    ("requested", "existing", "expected"),
    [
        ("auto", None, "hybrid"),
        ("dense", None, "dense"),
        ("hybrid", None, "hybrid"),
        ("sparse", None, "sparse"),
        ("auto", "dense", "dense"),
        ("dense", "dense", "dense"),
        ("hybrid", "dense", "hybrid"),
        ("auto", "hybrid", "hybrid"),
        ("hybrid", "hybrid", "hybrid"),
        ("auto", "sparse", "sparse"),
        ("sparse", "sparse", "sparse"),
    ],
)
def test_resolve_ingest_index_mode_compatible_transitions(requested, existing, expected) -> None:
    assert resolve_ingest_index_mode(requested, overwrite=False, existing_mode=existing) == expected


@pytest.mark.parametrize("requested", ["auto", "dense", "hybrid", "sparse"])
def test_resolve_ingest_index_mode_overwrite_ignores_existing_mode(requested) -> None:
    expected = "hybrid" if requested == "auto" else requested
    assert resolve_ingest_index_mode(requested, overwrite=True, existing_mode="sparse") == expected


@pytest.mark.parametrize(
    ("requested", "existing"),
    [
        ("dense", "hybrid"),
        ("sparse", "hybrid"),
        ("dense", "sparse"),
        ("hybrid", "sparse"),
        ("sparse", "dense"),
    ],
)
def test_resolve_ingest_index_mode_rejects_incompatible_append(requested, existing) -> None:
    with pytest.raises(ValueError, match="Cannot append"):
        resolve_ingest_index_mode(requested, overwrite=False, existing_mode=existing)
