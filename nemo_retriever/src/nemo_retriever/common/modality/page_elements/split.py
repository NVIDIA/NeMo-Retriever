# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU/GPU split helpers for local page-element detection.

``split_preprocess`` (CPU) -> ``split_detect`` (GPU) -> ``split_postprocess`` (CPU) produces the same
``page_elements_v3`` columns as :func:`detect_page_elements_v3` on the local-model path.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from nemo_retriever.common.modality.page_elements.local import postprocess_preds_page_element
from nemo_retriever.common.modality.page_elements.shared import (
    YOLOX_PAGE_V3_FINAL_SCORE,
    _RETRIEVER_TO_API,
    _apply_final_score_filter,
    _apply_page_elements_v3_postprocess,
    _counts_by_label,
    _decode_b64_image_to_np_array,
    _error_payload,
    _labels_from_model,
    _postprocess_to_per_image_detections,
)

logger = logging.getLogger(__name__)

PE_INPUT_SHAPE = (1024, 1024)
PRE_COLUMNS = ("_pe_pre", "_pe_pre_shape", "_pe_orig_shape")
GPU_COLUMNS = ("_pe_boxes", "_pe_labels", "_pe_scores", "_pe_elapsed")
ERR_COLUMN = "_pe_err"


def _dump_err(payload: Optional[Dict[str, Any]]) -> Optional[str]:
    # JSON keeps the Arrow schema stable across blocks with and without errors.
    return None if payload is None else json.dumps(payload)


def _is_missing_scalar(value: Any) -> bool:
    missing = pd.isna(value)
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _load_err(value: Any) -> Optional[Dict[str, Any]]:
    if _is_missing_scalar(value):
        return None
    return json.loads(value) if isinstance(value, str) else dict(value)


def _hwc_to_chw_float(arr: np.ndarray) -> torch.Tensor:
    if int(arr.shape[-1]) == 3 and int(arr.shape[0]) != 3:
        x = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
    else:
        x = torch.from_numpy(np.ascontiguousarray(arr))
    return x.to(dtype=torch.float32)


def split_preprocess(pages_df: pd.DataFrame) -> pd.DataFrame:
    """Decode page images and create the CPU-to-GPU split intermediates.

    Args:
        pages_df: Page rows containing ``page_image["image_b64"]``.

    Returns:
        A copy with ``_pe_pre``, ``_pe_pre_shape``, ``_pe_orig_shape``, and
        ``_pe_err`` added to the original columns.

    Raises:
        None. Per-row decode and resize failures are recorded in ``_pe_err``.
    """
    from nemo_retriever.models.local.nemotron_page_elements_v3 import resize_pad_chw_to_uint8

    pre: List[Optional[bytes]] = []
    pre_shape: List[Optional[List[int]]] = []
    orig_shape: List[Optional[List[int]]] = []
    errs: List[Optional[str]] = []
    for _, row in pages_df.iterrows():
        try:
            b64 = row.get("page_image")["image_b64"]
            if not b64:
                raise ValueError("No usable image_b64 found in row.")
            arr, (h, w) = _decode_b64_image_to_np_array(b64)
            y = resize_pad_chw_to_uint8(_hwc_to_chw_float(arr), PE_INPUT_SHAPE).contiguous()
            pre.append(y.numpy().tobytes())
            pre_shape.append([int(d) for d in y.shape])
            orig_shape.append([int(h), int(w)])
            errs.append(None)
        except BaseException as exc:
            pre.append(None)
            pre_shape.append(None)
            orig_shape.append(None)
            errs.append(_dump_err(_error_payload(stage="decode_image", exc=exc)))
    out = pages_df.copy()
    out["_pe_pre"], out["_pe_pre_shape"], out["_pe_orig_shape"], out[ERR_COLUMN] = pre, pre_shape, orig_shape, errs
    return out


def _postprocess_raw(model: Any, preds_list: List[Any]) -> tuple[list, list, list]:
    if hasattr(model, "postprocess"):
        return model.postprocess(preds_list)
    label_names = _labels_from_model(model)
    thresholds = getattr(model, "thresholds_per_class", None) or [
        YOLOX_PAGE_V3_FINAL_SCORE.get(_RETRIEVER_TO_API.get(n, n), 0.0) for n in label_names
    ]
    boxes, labels, scores = [], [], []
    for p in preds_list:
        if not isinstance(p, dict):
            boxes.append(torch.empty((0, 4), dtype=torch.float32))
            labels.append(torch.empty((0,), dtype=torch.int64))
            scores.append(torch.empty((0,), dtype=torch.float32))
            continue
        b_np, l_np, s_np = postprocess_preds_page_element(p, thresholds, label_names)
        boxes.append(torch.as_tensor(b_np, dtype=torch.float32))
        labels.append(torch.as_tensor(l_np, dtype=torch.int64))
        scores.append(torch.as_tensor(s_np, dtype=torch.float32))
    return boxes, labels, scores


def _per_image(x: Any, n: int) -> List[Any]:
    if isinstance(x, torch.Tensor) and x.ndim >= 2 and int(x.shape[0]) == n and n > 1:
        return [x[i] for i in range(n)]
    return list(x) if isinstance(x, list) else [x]


def _as_plain(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    return None


def split_detect(pages_df: pd.DataFrame, *, model: Any, inference_batch_size: int = 8, **_: Any) -> pd.DataFrame:
    """Run local PE inference on preprocessed page batches.

    Args:
        pages_df: Rows carrying ``_pe_pre``, ``_pe_pre_shape``, ``_pe_orig_shape``,
            and ``_pe_err``.
        model: Local page-elements model.
        inference_batch_size: Maximum number of pages per inference call.

    Returns:
        A copy without ``_pe_pre``, ``_pe_pre_shape``, or ``_pe_orig_shape``, with
        ``_pe_boxes``, ``_pe_labels``, ``_pe_scores``, ``_pe_elapsed``, and
        ``_pe_err`` added.

    Raises:
        ValueError: If ``inference_batch_size`` is not positive.
    """
    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be > 0")
    n_rows = len(pages_df.index)
    errs = [_load_err(v) for v in pages_df[ERR_COLUMN].tolist()]
    pre = pages_df["_pe_pre"].tolist()
    pre_shape = pages_df["_pe_pre_shape"].tolist()
    orig = pages_df["_pe_orig_shape"].tolist()
    boxes_out: List[Any] = [None] * n_rows
    labels_out: List[Any] = [None] * n_rows
    scores_out: List[Any] = [None] * n_rows
    elapsed_out: List[Optional[float]] = [None] * n_rows
    valid = [
        i
        for i in range(n_rows)
        if errs[i] is None and not any(_is_missing_scalar(value) for value in (pre[i], pre_shape[i], orig[i]))
    ]
    for start in range(0, len(valid), int(inference_batch_size)):
        chunk = valid[start : start + int(inference_batch_size)]
        tensors = [
            torch.from_numpy(
                np.frombuffer(pre[i], dtype=np.uint8).reshape([int(d) for d in pre_shape[i]]).astype(np.float32)
            )
            for i in chunk
        ]
        orig_shapes = [(int(orig[i][0]), int(orig[i][1])) for i in chunk]
        batch = torch.stack(tensors, dim=0)
        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                with torch.autocast(device_type="cuda"):
                    preds = model(batch, orig_shapes) if len(chunk) > 1 else model(batch, orig_shapes[0])
            if len(chunk) > 1:
                if isinstance(preds, dict):
                    raise RuntimeError("Model returned a single pred dict for batched input.")
                if isinstance(preds, list) and len(preds) != len(chunk):
                    raise RuntimeError(
                        f"Model returned {len(preds)} preds for batch size {len(chunk)}; falling back to per-image."
                    )
        except Exception:
            logger.exception("Error invoking page-elements model; retrying per image")
            preds = [model(batch[j : j + 1], orig_shapes[j]) for j in range(int(batch.shape[0]))]
        elapsed = time.perf_counter() - t0
        preds_list = [preds] if isinstance(preds, dict) else preds if isinstance(preds, list) else [preds]
        try:
            boxes, labels, scores = _postprocess_raw(model, preds_list)
            bl, ll, sl = (_per_image(v, len(chunk)) for v in (boxes, labels, scores))
            for local_i, row_i in enumerate(chunk):
                if local_i < min(len(bl), len(ll), len(sl)):
                    boxes_out[row_i] = _as_plain(bl[local_i])
                    labels_out[row_i] = _as_plain(ll[local_i])
                    scores_out[row_i] = _as_plain(sl[local_i])
                elapsed_out[row_i] = float(elapsed)
        except BaseException as exc:
            for row_i in chunk:
                errs[row_i] = _error_payload(stage="postprocess", exc=exc) | {"timing": {"seconds": float(elapsed)}}
    out = pages_df.drop(columns=list(PRE_COLUMNS))
    out["_pe_boxes"], out["_pe_labels"], out["_pe_scores"], out["_pe_elapsed"] = (
        boxes_out,
        labels_out,
        scores_out,
        elapsed_out,
    )
    out[ERR_COLUMN] = [_dump_err(e) for e in errs]
    return out


def split_postprocess(
    pages_df: pd.DataFrame,
    *,
    output_column: str = "page_elements_v3",
    num_detections_column: str = "page_elements_v3_num_detections",
    counts_by_label_column: str = "page_elements_v3_counts_by_label",
    **_: Any,
) -> pd.DataFrame:
    """Build the final page-elements payloads and remove split intermediates.

    Args:
        pages_df: Rows carrying ``_pe_boxes``, ``_pe_labels``, ``_pe_scores``,
            ``_pe_elapsed``, and ``_pe_err``.
        output_column: Name for the page-elements payload column.
        num_detections_column: Name for the detection-count column.
        counts_by_label_column: Name for per-label counts.

    Returns:
        A copy without the ``_pe_*`` columns, with the requested output columns added.

    Raises:
        ValueError: If an ``_pe_err`` string contains invalid JSON.
    """
    label_names = _labels_from_model(None)
    payloads: List[Dict[str, Any]] = []
    for err, b, lab, s, el in zip(
        pages_df[ERR_COLUMN].tolist(),
        pages_df["_pe_boxes"].tolist(),
        pages_df["_pe_labels"].tolist(),
        pages_df["_pe_scores"].tolist(),
        pages_df["_pe_elapsed"].tolist(),
    ):
        err_payload = _load_err(err)
        if err_payload is not None:
            payloads.append(err_payload)
            continue
        if _is_missing_scalar(el):
            payloads.append({"detections": []})
            continue
        try:
            if any(_is_missing_scalar(value) for value in (b, lab, s)):
                dets: List[Dict[str, Any]] = []
            else:
                bt = torch.as_tensor(np.asarray(list(b), dtype=np.float32).reshape(-1, 4))
                lt = torch.as_tensor(np.asarray(list(lab), dtype=np.int64).reshape(-1))
                st = torch.as_tensor(np.asarray(list(s), dtype=np.float32).reshape(-1))
                dets = _postprocess_to_per_image_detections(
                    boxes=[bt], labels=[lt], scores=[st], batch_size=1, label_names=label_names
                )[0]
            dets = _apply_final_score_filter(_apply_page_elements_v3_postprocess(dets))
            payloads.append({"detections": dets, "timing": {"seconds": float(el)}, "error": None})
        except BaseException as exc:
            payloads.append(_error_payload(stage="postprocess", exc=exc) | {"timing": {"seconds": float(el)}})
    out = pages_df.drop(columns=[c for c in out_columns() if c in pages_df.columns])
    out[output_column] = payloads
    out[num_detections_column] = [int(len(p.get("detections") or [])) for p in payloads]
    out[counts_by_label_column] = [_counts_by_label(p.get("detections") or []) for p in payloads]
    return out


def out_columns() -> tuple[str, ...]:
    """Return names of the intermediate columns used by the split stages.

    Args:
        None.

    Returns:
        The ``_pe_pre``, ``_pe_pre_shape``, ``_pe_orig_shape``, ``_pe_boxes``,
        ``_pe_labels``, ``_pe_scores``, ``_pe_elapsed``, and ``_pe_err`` columns.

    Raises:
        None.
    """
    return PRE_COLUMNS + GPU_COLUMNS + (ERR_COLUMN,)
