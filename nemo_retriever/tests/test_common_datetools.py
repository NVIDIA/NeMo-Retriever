# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_retriever.common.api.util.converters.datetools import normalize_timezone_aware_iso8601_to_utc


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2030-01-01T00:00:00Z", "2030-01-01T00:00:00+00:00"),
        ("2030-01-01T01:00:00+01:00", "2030-01-01T00:00:00+00:00"),
        ("2030-01-01T00:00:00.123456+00:00", "2030-01-01T00:00:00.123456+00:00"),
    ],
)
def test_normalize_timezone_aware_iso8601_to_utc(value: str, expected: str) -> None:
    assert normalize_timezone_aware_iso8601_to_utc(value) == expected


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("not-a-timestamp", "ISO-8601"),
        ("2030-01-01T00:00:00", "timezone"),
    ],
)
def test_normalize_timezone_aware_iso8601_to_utc_rejects_invalid_input(value: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_timezone_aware_iso8601_to_utc(value)
