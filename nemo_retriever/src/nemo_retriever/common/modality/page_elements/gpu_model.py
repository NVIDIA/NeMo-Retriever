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


def _geometry_reference(wrapper, pred_boxes, sizes, cpu_sizes):
    """Retain the audited per-page arithmetic for other metadata representations."""
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


def _geometry_bulk(wrapper, pred_boxes, sizes, cpu_sizes):
    """Apply geometry across pages while preserving page and detection order."""
    counts = [0 if p is None else int(p.shape[0]) for p in pred_boxes]
    nonempty = [p for p in pred_boxes if p is not None and p.shape[0]]
    if (
        not nonempty
        or sizes.dtype != torch.int64
        or any(
            (
                p.dtype != torch.float32 or p.device != sizes.device or p.ndim != 2 or (p.shape[1] != 7)
                for p in pred_boxes
                if p is not None
            )
        )
    ):
        return _geometry_reference(wrapper, pred_boxes, sizes, cpu_sizes)
    packed = torch.cat(nonempty, dim=0)
    ids = torch.repeat_interleave(
        torch.arange(len(counts), device=sizes.device),
        torch.tensor(counts, device=sizes.device),
        output_size=sum(counts),
    )
    height_ratios = wrapper.img_size[0] / sizes[:, 0]
    width_ratios = wrapper.img_size[1] / sizes[:, 1]
    ratios = torch.minimum(height_ratios, width_ratios)
    boxes = packed[:, :4] / ratios[ids, None]
    upper = torch.stack((sizes[:, 1], sizes[:, 0], sizes[:, 1], sizes[:, 0]), dim=1)[ids]
    boxes = torch.clamp(boxes, min=torch.zeros((), dtype=boxes.dtype, device=boxes.device), max=upper)
    kept = (boxes[:, 2] - boxes[:, 0] > wrapper.min_bbox_size) & (boxes[:, 3] - boxes[:, 1] > wrapper.min_bbox_size)
    if wrapper.normalize_boxes:
        boxes[:, [0, 2]] /= sizes[ids, 1, None]
        boxes[:, [1, 3]] /= sizes[ids, 0, None]
    selected = torch.cat((boxes, packed[:, 4:]), dim=1)[kept]
    # Read only ragged detection counts to form per-page GPU tensor views.
    valid_counts = (
        torch.zeros(len(counts), device=sizes.device, dtype=torch.int64)
        .scatter_add_(0, ids, kept.to(torch.int64))
        .cpu()
        .tolist()
    )
    scores = selected[:, 4] * selected[:, 5]
    result = []
    offset = 0
    for original, count in zip(pred_boxes, valid_counts):
        if original is None:
            result.append({"labels": torch.empty(0), "boxes": torch.empty((0, 4)), "scores": torch.empty(0)})
        else:
            block = selected[offset : offset + count]
            result.append({"labels": block[:, 6], "boxes": block[:, :4], "scores": scores[offset : offset + count]})
        offset += count
    return result


def forward_with_cpu_shapes(wrapper, x, orig_sizes):
    """Keep upstream inference/NMS while batching its output geometry.

    Original dimensions are metadata supplied by the renderer, not values read
    from CUDA tensors. Scale ratios and normalization retain upstream GPU
    arithmetic, with reference handling for other metadata representations.
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
    return _geometry_bulk(wrapper, pred_boxes, sizes, cpu_sizes)
