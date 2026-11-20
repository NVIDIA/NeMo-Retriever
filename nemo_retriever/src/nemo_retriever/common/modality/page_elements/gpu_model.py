# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Detector output geometry using original CPU page dimensions."""

import ast
from functools import lru_cache
import hashlib
import inspect
import textwrap

import torch


_AUDITED_GEOMETRY_SHA256 = "bc96ed6757adddd881d583de626bee633e5b23af96b571e83a6c96ae3a3bbc24"


@lru_cache(maxsize=8)
def _supports_forward(wrapper_type):
    """Enable the optimization only for the audited provider geometry contract."""
    try:
        functions = [
            ast.parse(textwrap.dedent(inspect.getsource(getattr(wrapper_type, name)))).body[0]
            for name in ("forward", "reformat_input")
        ]
        for function in functions:
            if ast.get_docstring(function, clean=False) is not None:
                function.body = function.body[1:]
        source = "\n".join(ast.dump(function, include_attributes=False) for function in functions)
    except (AttributeError, OSError, TypeError, SyntaxError):
        return False
    return hashlib.sha256(source.encode()).hexdigest() == _AUDITED_GEOMETRY_SHA256


def forward_with_cpu_shapes(wrapper, x, orig_sizes):
    """Keep upstream inference/NMS while avoiding GPU scalar reads for geometry.

    Original dimensions are metadata supplied by the renderer, not values read
    from CUDA tensors. Scale ratios and normalization retain upstream GPU
    arithmetic; Python only needs the original dimensions for clipping limits.
    """
    from nemotron_page_elements_v3.yolox.boxes import postprocess

    cpu_sizes = torch.as_tensor(orig_sizes, device="cpu")
    if cpu_sizes.ndim == 1:
        cpu_sizes = cpu_sizes.unsqueeze(0)
    cpu_sizes = cpu_sizes[:, :2].tolist()
    x, sizes = wrapper.reformat_input(x, orig_sizes)
    if x.max() <= 1:
        x *= 255
    pred_boxes = wrapper.model(x.to(wrapper.device))
    pred_boxes = postprocess(
        pred_boxes,
        wrapper.config.num_classes,
        wrapper.conf_thresh,
        wrapper.iou_thresh,
        class_agnostic=wrapper.class_agnostic,
    )
    # Preserve reciprocal/division precision while replacing Python min on
    # CUDA scalars with one batched GPU operation.
    height_ratios = wrapper.img_size[0] / sizes[:, 0]
    width_ratios = wrapper.img_size[1] / sizes[:, 1]
    ratios = torch.minimum(height_ratios, width_ratios)
    preds = []
    for p, size, cpu_size, ratio in zip(pred_boxes, sizes, cpu_sizes, ratios):
        if p is None:
            preds.append({"labels": torch.empty(0), "boxes": torch.empty((0, 4)), "scores": torch.empty(0)})
            continue
        p = p.view(-1, p.size(-1))
        boxes = p[:, :4] / ratio
        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, cpu_size[1])
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, cpu_size[0])
        kept = (boxes[:, 2] - boxes[:, 0] > wrapper.min_bbox_size) & (boxes[:, 3] - boxes[:, 1] > wrapper.min_bbox_size)
        boxes = boxes[kept]
        p = p[kept]
        if wrapper.normalize_boxes:
            boxes[:, [0, 2]] /= size[1]
            boxes[:, [1, 3]] /= size[0]
        scores = p[:, 4] * p[:, 5]
        labels = p[:, 6]
        preds.append({"labels": labels, "boxes": boxes, "scores": scores})
    return preds
