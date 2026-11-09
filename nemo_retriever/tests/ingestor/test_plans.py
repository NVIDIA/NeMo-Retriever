# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_retriever.common.params import CaptionParams, DedupParams, ExtractParams, StoreParams
from nemo_retriever.ingest.plan import profile_extract_defaults
from nemo_retriever.ingestor.plans import resolve_effective_extract_params


def _resolve(extract_params, *, caption=None, dedup=None, store=None):
    resolved = resolve_effective_extract_params(
        extract_params, caption_params=caption, dedup_params=dedup, store_params=store
    )
    return resolved.extract_images


def test_default_extract_images_turns_off_without_image_consumers():
    assert _resolve(ExtractParams()) is False


@pytest.mark.parametrize(
    "consumer",
    [
        {"caption": CaptionParams()},
        {"dedup": DedupParams()},
        {"store": StoreParams()},
    ],
    ids=["caption", "dedup", "store"],
)
def test_default_extract_images_stays_on_for_image_consumers(consumer):
    assert _resolve(ExtractParams(), **consumer) is True


def test_disabled_dedup_is_not_an_image_consumer():
    assert _resolve(ExtractParams(), dedup=DedupParams(content_hash=False, bbox_iou=False)) is False


@pytest.mark.parametrize("value", [True, False])
def test_explicit_extract_images_wins(value):
    assert _resolve(ExtractParams(extract_images=value)) is value


def test_nested_images_keep_extract_images():
    assert _resolve(ExtractParams(extract_nested_images=True)) is True


def test_fast_text_profile_keeps_explicit_false():
    assert _resolve(ExtractParams(**profile_extract_defaults("fast-text"))) is False


def test_none_extract_params_pass_through():
    assert resolve_effective_extract_params(None, caption_params=None, dedup_params=None, store_params=None) is None
