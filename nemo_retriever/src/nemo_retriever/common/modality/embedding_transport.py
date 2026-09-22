# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fields needed after built-in content reshape in terminal VDB uploads."""

import pandas as pd

from nemo_retriever.common.stage_errors import iter_stage_errors_from_value

CONTENT_COUNTS_FIELD = "_embedding_transport_content_counts"

# Embedding inputs and canonical VDB content, provenance, and diagnostics.
# Metadata stays intact because it is part of the retrieval contract.
EMBEDDING_TRANSPORT_FIELDS = (
    "text",
    "content",
    "metadata",
    "path",
    "source_id",
    "source",
    "filename",
    "page_number",
    "_page_number",
    "document_type",
    "_embed_modality",
    "_image_b64",
    "_content_type",
    "content_type",
    "_stored_image_uri",
    "stored_image_uri",
    "_bbox_xyxy_norm",
    "bbox_xyxy_norm",
    "page_elements_v3_num_detections",
    "page_elements_v3_counts_by_label",
)


def project_embedding_transport(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop extraction payloads after their text and image inputs are resolved."""
    columns = [field for field in EMBEDDING_TRANSPORT_FIELDS if field in frame.columns]
    for column in frame.columns:
        if column in columns:
            continue
        # Retain failing stage payloads unchanged so error paths and messages
        # remain available to the existing validators.
        for value in frame[column]:
            if any(iter_stage_errors_from_value(value)):
                columns.append(column)
                break

    result = frame.loc[:, columns].copy()
    counts = [{} for _ in frame.index]
    for kind in ("table", "chart", "infographic"):
        if kind in frame:
            for row_counts, detections in zip(counts, frame[kind]):
                if isinstance(detections, list):
                    row_counts[kind] = len(detections)
    result[CONTENT_COUNTS_FIELD] = pd.Series(counts, index=frame.index, dtype=object)

    if "page_image" in frame and "page_image" not in result:
        # VDB conversion needs the original URI to distinguish page images
        # from uploaded crops. Keep its existing shape without the raster.
        result["page_image"] = pd.Series(
            [
                {"stored_image_uri": value.get("stored_image_uri")} if isinstance(value, dict) else None
                for value in frame["page_image"]
            ],
            index=frame.index,
            dtype=object,
        )
    return result
