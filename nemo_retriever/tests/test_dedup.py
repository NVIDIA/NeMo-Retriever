# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nemo_retriever.dedup.dedup."""

import pandas as pd

from nemo_retriever.dedup.dedup import calculate_iou, dedup_images


def _make_image(b64: str, bbox: list[float] | None = None) -> dict:
    d: dict = {"image_b64": b64}
    if bbox is not None:
        d["bbox_xyxy_norm"] = bbox
    return d


def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ── calculate_iou ──────────────────────────────────────────────────────────


class TestCalculateIou:
    def test_identical_boxes(self):
        assert calculate_iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1.0

    def test_no_overlap(self):
        assert calculate_iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0

    def test_partial_overlap(self):
        iou = calculate_iou((0, 0, 2, 2), (1, 1, 3, 3))
        assert 0.1 < iou < 0.2  # intersection=1, union=7

    def test_zero_area(self):
        assert calculate_iou((0, 0, 0, 0), (0, 0, 1, 1)) == 0.0


# ── content-hash dedup ─────────────────────────────────────────────────────


class TestContentHashDedup:
    def test_removes_exact_duplicate_images(self):
        df = _make_df(
            [
                {
                    "images": [_make_image("AAA"), _make_image("BBB"), _make_image("AAA")],
                }
            ]
        )
        result = dedup_images(df, content_hash=True, bbox_iou=False)
        assert len(result.iloc[0]["images"]) == 2

    def test_keeps_unique_images(self):
        df = _make_df(
            [
                {
                    "images": [_make_image("AAA"), _make_image("BBB")],
                }
            ]
        )
        result = dedup_images(df, content_hash=True, bbox_iou=False)
        assert len(result.iloc[0]["images"]) == 2


# ── bbox IoU dedup ─────────────────────────────────────────────────────────


class TestBboxIouDedup:
    def test_drops_image_overlapping_with_table(self):
        df = _make_df(
            [
                {
                    "images": [_make_image("X", bbox=[0.0, 0.0, 1.0, 1.0])],
                    "tables": [{"bbox_xyxy_norm": [0.0, 0.0, 1.0, 1.0], "text": "t"}],
                    "charts": [],
                    "infographics": [],
                }
            ]
        )
        result = dedup_images(df, content_hash=False, bbox_iou=True, iou_threshold=0.45)
        assert len(result.iloc[0]["images"]) == 0

    def test_keeps_image_below_threshold(self):
        df = _make_df(
            [
                {
                    "images": [_make_image("X", bbox=[0.0, 0.0, 0.1, 0.1])],
                    "tables": [{"bbox_xyxy_norm": [0.9, 0.9, 1.0, 1.0], "text": "t"}],
                    "charts": [],
                    "infographics": [],
                }
            ]
        )
        result = dedup_images(df, content_hash=False, bbox_iou=True, iou_threshold=0.45)
        assert len(result.iloc[0]["images"]) == 1

    def test_keeps_image_when_no_structured_content(self):
        df = _make_df(
            [
                {
                    "images": [_make_image("X", bbox=[0.0, 0.0, 1.0, 1.0])],
                    "tables": [],
                    "charts": [],
                    "infographics": [],
                }
            ]
        )
        result = dedup_images(df, content_hash=False, bbox_iou=True)
        assert len(result.iloc[0]["images"]) == 1

    def test_image_without_bbox_is_kept(self):
        df = _make_df(
            [
                {
                    "images": [_make_image("X")],  # no bbox
                    "tables": [{"bbox_xyxy_norm": [0.0, 0.0, 1.0, 1.0], "text": "t"}],
                    "charts": [],
                    "infographics": [],
                }
            ]
        )
        result = dedup_images(df, content_hash=False, bbox_iou=True)
        assert len(result.iloc[0]["images"]) == 1


# ── edge cases ─────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = dedup_images(df)
        assert result.empty

    def test_no_images_column(self):
        df = _make_df([{"text": "hello"}])
        result = dedup_images(df)
        assert len(result) == 1

    def test_empty_images_list(self):
        df = _make_df([{"images": []}])
        result = dedup_images(df)
        assert len(result.iloc[0]["images"]) == 0
