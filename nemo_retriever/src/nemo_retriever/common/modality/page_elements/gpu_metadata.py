# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Batch detection metadata readback without transferring page images."""

from __future__ import annotations

from collections.abc import Sequence

import nvtx
import torch


@nvtx.annotate("nrl.cuda_rdt::detector_prediction_metadata", color="purple")
def prediction_batch_to_cpu(predictions: Sequence[dict[str, torch.Tensor]]) -> Sequence[dict[str, torch.Tensor]]:
    """Copy homogeneous CUDA detections together, preserving page order and dtype.

    Empty predictions and heterogeneous dtypes keep the upstream representation.
    In particular, integer labels must never be rounded by a floating-point cast.
    """
    nonempty = [prediction for prediction in predictions if prediction["boxes"].shape[0]]
    if not nonempty:
        return predictions
    first = nonempty[0]["boxes"]
    if not first.is_cuda or not all(
        prediction[key].device == first.device and prediction[key].dtype == first.dtype
        for prediction in nonempty
        for key in ("boxes", "labels", "scores")
    ):
        return predictions

    packed = torch.cat(
        [
            torch.cat((prediction["boxes"], prediction["labels"][:, None], prediction["scores"][:, None]), dim=1)
            for prediction in nonempty
        ],
        dim=0,
    ).cpu()
    result = []
    offset = 0
    for prediction in predictions:
        count = prediction["boxes"].shape[0]
        if not count:
            result.append(prediction)
            continue
        block = packed[offset : offset + count]
        result.append({"boxes": block[:, :4], "labels": block[:, 4], "scores": block[:, 5]})
        offset += count
    return result
