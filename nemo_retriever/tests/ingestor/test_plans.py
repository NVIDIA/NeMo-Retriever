# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_retriever.common.params import CaptionParams, DedupParams, ExtractParams, StoreParams
from nemo_retriever.ingest.plan import profile_extract_defaults
from nemo_retriever.ingestor.plans import resolve_effective_extract_params

_CONSUMERS = {
    "none": {},
    "caption": {"caption_params": CaptionParams()},
    "dedup": {"dedup_params": DedupParams()},
    "store": {"store_params": StoreParams()},
}


def _resolve(extract_params, **stages):
    stages = {"caption_params": None, "dedup_params": None, "store_params": None, **stages}
    return resolve_effective_extract_params(extract_params, **stages).extract_images


@pytest.mark.parametrize("consumer", list(_CONSUMERS))
@pytest.mark.parametrize("value", [None, True, False])
def test_auto_resolves_from_consumers_and_explicit_values_win(value, consumer):
    expected = consumer != "none" if value is None else value
    assert _resolve(ExtractParams(extract_images=value), **_CONSUMERS[consumer]) is expected


def test_default_is_auto():
    assert ExtractParams().extract_images is None


def test_disabled_dedup_is_not_an_image_consumer():
    assert _resolve(ExtractParams(), dedup_params=DedupParams(content_hash=False, bbox_iou=False)) is False


def test_nested_images_resolve_auto_to_true():
    assert _resolve(ExtractParams(extract_nested_images=True)) is True


def test_nested_images_reject_explicit_false():
    with pytest.raises(ValueError, match="extract_nested_images=True requires extract_images=True"):
        ExtractParams(extract_images=False, extract_nested_images=True)


def test_fast_text_profile_keeps_explicit_false():
    assert _resolve(ExtractParams(**profile_extract_defaults("fast-text"))) is False


def test_none_extract_params_pass_through():
    assert resolve_effective_extract_params(None, caption_params=None, dedup_params=None, store_params=None) is None
