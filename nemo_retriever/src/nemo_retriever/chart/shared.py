# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import base64
import io
import time
import traceback

import pandas as pd
from nemo_retriever.nim.error_reporter import report_error
from nemo_retriever.nim.nim import NIMClient, invoke_image_inference_batches
from nemo_retriever.params import RemoteRetryParams

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]

try:
    from nv_ingest_api.internal.primitives.nim.model_interface.yolox import (
        YOLOX_GRAPHIC_MIN_SCORE,
    )
except ImportError:
    YOLOX_GRAPHIC_MIN_SCORE = 0.1  # type: ignore[assignment]


def _error_payload(*, stage: str, exc: BaseException) -> Dict[str, Any]:
    return {
        "detections": [],
        "error": {
            "stage": str(stage),
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        },
    }


def _decode_b64_image_to_chw_tensor(image_b64: str) -> Tuple["torch.Tensor", Tuple[int, int]]:
    if torch is None or Image is None or np is None:  # pragma: no cover
        raise ImportError("chart detection requires torch, pillow, and numpy.")

    raw = base64.b64decode(image_b64)
    with Image.open(io.BytesIO(raw)) as im0:
        im = im0.convert("RGB")
        w, h = im.size
        arr = np.array(im, dtype=np.uint8)  # (H,W,3)

    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,H,W) uint8
    t = t.to(dtype=torch.float32)
    return t, (int(h), int(w))


def _crop_b64_image_by_norm_bbox(
    page_image_b64: str,
    *,
    bbox_xyxy_norm: Sequence[float],
    image_format: str = "png",
) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
    """
    Crop a base64-encoded RGB image by a normalized xyxy bbox.

    Returns:
      - cropped_image_b64 (png) or None
      - cropped_shape_hw (H,W) or None
    """
    if Image is None:  # pragma: no cover
        raise ImportError("Cropping requires pillow.")
    if not isinstance(page_image_b64, str) or not page_image_b64:
        return None, None
    try:
        x1n, y1n, x2n, y2n = [float(x) for x in bbox_xyxy_norm]
    except Exception:
        return None, None

    try:
        raw = base64.b64decode(page_image_b64)
        with Image.open(io.BytesIO(raw)) as im0:
            im = im0.convert("RGB")
            w, h = im.size
            if w <= 1 or h <= 1:
                return None, None

            def _clamp_int(v: float, lo: int, hi: int) -> int:
                if v != v:  # NaN
                    return lo
                return int(min(max(v, float(lo)), float(hi)))

            x1 = _clamp_int(x1n * w, 0, w)
            x2 = _clamp_int(x2n * w, 0, w)
            y1 = _clamp_int(y1n * h, 0, h)
            y2 = _clamp_int(y2n * h, 0, h)

            if x2 <= x1 or y2 <= y1:
                return None, None

            crop = im.crop((x1, y1, x2, y2))
            cw, ch = crop.size
            if cw <= 1 or ch <= 1:
                return None, None

            buf = io.BytesIO()
            fmt = str(image_format or "png").lower()
            if fmt not in {"png"}:
                fmt = "png"
            crop.save(buf, format=fmt.upper())
            return base64.b64encode(buf.getvalue()).decode("ascii"), (int(ch), int(cw))
    except Exception:
        return None, None


def _labels_from_model(model: Any) -> List[str]:
    try:
        labels = getattr(getattr(model, "_model", None), "labels", None)
        if isinstance(labels, (list, tuple)) and all(isinstance(x, str) for x in labels):
            return [str(x) for x in labels]
    except Exception:
        pass

    try:
        out = getattr(model, "output", None)
        if isinstance(out, dict):
            classes = out.get("classes")
            if isinstance(classes, (list, tuple)) and all(isinstance(x, str) for x in classes):
                return [str(x) for x in classes]
    except Exception:
        pass

    return []


def _prediction_to_detections(pred: Any, *, label_names: List[str]) -> List[Dict[str, Any]]:
    if torch is None:  # pragma: no cover
        raise ImportError("torch required for prediction parsing.")

    boxes = labels = scores = None
    if isinstance(pred, dict):
        # IMPORTANT: do not use `or` chains here. torch.Tensor truthiness is ambiguous and raises.
        def _get_any(d: Dict[str, Any], *keys: str) -> Any:
            for k in keys:
                if k in d:
                    v = d.get(k)
                    if v is not None:
                        return v
            return None

        boxes = _get_any(pred, "boxes", "bboxes", "bbox", "box")
        labels = _get_any(pred, "labels", "classes", "class_ids", "class")
        scores = _get_any(pred, "scores", "conf", "confidences", "score")
    elif isinstance(pred, (list, tuple)) and len(pred) >= 3:
        boxes, labels, scores = pred[0], pred[1], pred[2]

    if boxes is None or labels is None:
        return []

    def _to_tensor(x: Any) -> Optional["torch.Tensor"]:
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        try:
            return torch.as_tensor(x).detach().cpu()
        except Exception:
            return None

    # Handle string labels (e.g. NIM returns ["chart_title", "xlabel", ...]).
    # torch.as_tensor cannot convert strings, so handle them before tensor conversion.
    _string_labels: Optional[List[str]] = None
    if isinstance(labels, (list, tuple)) and labels and isinstance(labels[0], str):
        _string_labels = [str(x) for x in labels]

    b = _to_tensor(boxes)
    labels_t = _to_tensor(labels) if _string_labels is None else None
    s = _to_tensor(scores) if scores is not None else None
    if b is None:
        return []
    if labels_t is None and _string_labels is None:
        return []

    if b.ndim != 2 or int(b.shape[-1]) != 4:
        return []
    if labels_t is not None:
        if labels_t.ndim == 2 and int(labels_t.shape[-1]) == 1:
            labels_t = labels_t.squeeze(-1)
        if labels_t.ndim != 1:
            return []

    n_labels = len(_string_labels) if _string_labels is not None else int(labels_t.shape[0])
    n = int(min(b.shape[0], n_labels))
    dets: List[Dict[str, Any]] = []
    for i in range(n):
        try:
            x1, y1, x2, y2 = [float(x) for x in b[i].tolist()]
        except Exception:
            continue

        if _string_labels is not None:
            label_i = i
            label_name = _string_labels[i]
        else:
            label_i: Optional[int]
            try:
                label_i = int(labels_t[i].item())
            except Exception:
                label_i = None

            label_name = None
            if label_i is not None and 0 <= label_i < len(label_names):
                label_name = label_names[label_i]
            if not label_name:
                label_name = f"label_{label_i}" if label_i is not None else "unknown"

        score_f: Optional[float]
        if s is not None and s.ndim >= 1 and int(s.shape[0]) > i:
            try:
                score_f = float(s[i].item())
            except Exception:
                score_f = None
        else:
            score_f = None

        dets.append(
            {
                "bbox_xyxy_norm": [x1, y1, x2, y2],
                "label": label_i,
                "label_name": str(label_name),
                "score": score_f,
            }
        )
    return dets


def _counts_by_label(detections: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for d in detections:
        if not isinstance(d, dict):
            continue
        name = d.get("label_name")
        if not isinstance(name, str) or not name.strip():
            name = f"label_{d.get('label')}"
        k = str(name)
        out[k] = int(out.get(k, 0) + 1)
    return out


def _remote_response_to_ge_detections(response_json: Any) -> List[Dict[str, Any]]:
    """Parse a NIM graphic-elements response into the standard detection list.

    The NIM returns either:
    * ``{"label_name": [[x0, y0, x1, y1, conf], ...], ...}`` (annotation dict), or
    * ``{"bounding_boxes": {"label_name": [{"x_min":..., ...}]}}`` (NIM v2), or
    * a dict with ``boxes``/``labels``/``scores`` tensors (model-pred style).
    """
    if not isinstance(response_json, dict):
        return []

    # Unwrap common NIM envelopes.
    candidates: List[Any] = [response_json]
    for key in ("data", "output", "predictions"):
        nested = response_json.get(key)
        if isinstance(nested, list) and nested:
            candidates.append(nested[0])

    for cand in candidates:
        if not isinstance(cand, dict):
            continue

        # NIM v2 bounding_boxes format.
        bb = cand.get("bounding_boxes")
        if isinstance(bb, dict):
            dets: List[Dict[str, Any]] = []
            for label_name, items in bb.items():
                if not isinstance(items, list):
                    continue
                for item in items:
                    if isinstance(item, dict):
                        dets.append(
                            {
                                "bbox_xyxy_norm": [
                                    float(item.get("x_min", 0)),
                                    float(item.get("y_min", 0)),
                                    float(item.get("x_max", 0)),
                                    float(item.get("y_max", 0)),
                                ],
                                "label": None,
                                "label_name": str(label_name),
                                "score": float(item.get("confidence", 0)),
                            }
                        )
            if dets:
                return dets

        # Annotation dict: {"chart_title": [[x0, y0, x1, y1, conf], ...]}
        if all(isinstance(v, list) for v in cand.values()):
            dets = []
            for label_name, boxes in cand.items():
                if not isinstance(boxes, list):
                    continue
                for box in boxes:
                    if isinstance(box, (list, tuple)) and len(box) >= 4:
                        dets.append(
                            {
                                "bbox_xyxy_norm": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                                "label": None,
                                "label_name": str(label_name),
                                "score": float(box[4]) if len(box) > 4 else None,
                            }
                        )
            if dets:
                return dets

    return []


# ---------------------------------------------------------------------------
# Combined graphic-elements + OCR core function
# ---------------------------------------------------------------------------


def graphic_elements_ocr_page_elements(
    batch_df: Any,
    *,
    graphic_elements_model: Any = None,
    ocr_model: Any = None,
    graphic_elements_invoke_url: str = "",
    ocr_invoke_url: str = "",
    api_key: str = "",
    request_timeout_s: float = 120.0,
    remote_retry: RemoteRetryParams | None = None,
    nim_client: NIMClient | None = None,
    **kwargs: Any,
) -> Any:
    """
    Run graphic-elements + OCR on chart crops and produce structure-aware text.

    For each row (page) in ``batch_df``:
    1. Read ``page_elements_v3`` detections and ``page_image["image_b64"]``.
    2. Crop all chart detections from the page image.
    3. Run graphic-elements model on each crop to get element bboxes.
    4. Run OCR on each crop to get text with bboxes.
    5. Join the two outputs using ``join_graphic_elements_and_ocr_output()``
       to produce semantically structured chart text.
    6. Fall back to OCR-only text if graphic-elements returns no detections.

    Returns
    -------
    pandas.DataFrame
        Original columns plus ``chart`` and ``graphic_elements_ocr_v1``.
    """
    from nemo_retriever.ocr.ocr import (
        _blocks_to_text,
        _crop_all_from_page,
        _extract_remote_ocr_item,
        _np_rgb_to_b64_png,
        _parse_ocr_result,
    )
    from nemo_retriever.utils.table_and_chart import join_graphic_elements_and_ocr_output

    retry = remote_retry or RemoteRetryParams(
        remote_max_pool_workers=int(kwargs.get("remote_max_pool_workers", 16)),
        remote_max_retries=int(kwargs.get("remote_max_retries", 10)),
        remote_max_429_retries=int(kwargs.get("remote_max_429_retries", 5)),
    )

    if not isinstance(batch_df, pd.DataFrame):
        raise NotImplementedError("graphic_elements_ocr_page_elements currently only supports pandas.DataFrame input.")

    ge_url = (graphic_elements_invoke_url or kwargs.get("graphic_elements_invoke_url") or "").strip()
    ocr_url = (ocr_invoke_url or kwargs.get("ocr_invoke_url") or "").strip()
    use_remote_ge = bool(ge_url)
    use_remote_ocr = bool(ocr_url)

    if not use_remote_ge and graphic_elements_model is None:
        raise ValueError("A local `graphic_elements_model` is required when `graphic_elements_invoke_url` is not set.")
    if not use_remote_ocr and ocr_model is None:
        raise ValueError("A local `ocr_model` is required when `ocr_invoke_url` is not set.")

    label_names = _labels_from_model(graphic_elements_model) if graphic_elements_model is not None else []
    inference_batch_size = int(kwargs.get("inference_batch_size", 8))

    num_rows = len(batch_df)
    all_chart: List[List[Dict[str, Any]]] = [[] for _ in range(num_rows)]
    all_meta: List[Dict[str, Any]] = [{"timing": None, "error": None} for _ in range(num_rows)]

    t0_total = time.perf_counter()

    # ---------------------------------------------------------------
    # Pass 1: Collect all chart crops across every row (page) in the
    # batch.  Track which row each crop belongs to so we can stitch
    # results back later.
    # ---------------------------------------------------------------
    flat_crops: List[Any] = []
    flat_crop_b64s: List[str] = []
    crop_row_indices: List[int] = []

    for row_i, row in enumerate(batch_df.itertuples(index=False)):
        try:
            pe = getattr(row, "page_elements_v3", None)
            dets: List[Dict[str, Any]] = []
            if isinstance(pe, dict):
                dets = pe.get("detections") or []
            if not isinstance(dets, list):
                dets = []

            page_image = getattr(row, "page_image", None) or {}
            page_image_b64 = page_image.get("image_b64") if isinstance(page_image, dict) else None

            if not isinstance(page_image_b64, str) or not page_image_b64:
                continue

            crops = _crop_all_from_page(page_image_b64, dets, {"chart"})
            if not crops:
                continue

            for crop in crops:
                flat_crops.append(crop)
                crop_row_indices.append(row_i)
                if use_remote_ge or use_remote_ocr:
                    flat_crop_b64s.append(_np_rgb_to_b64_png(crop[2]))

        except BaseException as e:
            print(f"Warning: chart crop collection failed for row {row_i}: {type(e).__name__}: {e}")
            report_error("graphic_elements_ocr_page_elements:crop", e, row_index=row_i)
            all_meta[row_i]["error"] = {
                "stage": "graphic_elements_ocr_page_elements:crop",
                "type": e.__class__.__name__,
                "message": str(e),
                "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
            }

    if not flat_crops:
        elapsed = time.perf_counter() - t0_total
        for meta in all_meta:
            meta["timing"] = {"seconds": float(elapsed)}
        out = batch_df.copy()
        out["chart"] = all_chart
        out["graphic_elements_ocr_v1"] = all_meta
        out["graphic_elements_v1_num_detections"] = [0 for _ in range(num_rows)]
        out["graphic_elements_v1_counts_by_label"] = [{} for _ in range(num_rows)]
        return out

    n_crops = len(flat_crops)

    # ---------------------------------------------------------------
    # Pass 2: Run graphic-elements on ALL crops in one batched call.
    # ---------------------------------------------------------------
    ge_results: List[List[Dict[str, Any]]] = [[] for _ in range(n_crops)]
    try:
        if use_remote_ge:
            _ge_kw = dict(
                invoke_url=ge_url,
                image_b64_list=flat_crop_b64s,
                api_key=api_key or None,
                timeout_s=float(request_timeout_s),
                max_batch_size=inference_batch_size,
                max_retries=int(retry.remote_max_retries),
                max_429_retries=int(retry.remote_max_429_retries),
            )
            if nim_client is not None:
                response_items = nim_client.invoke_image_inference_batches(**_ge_kw)
            else:
                response_items = invoke_image_inference_batches(
                    **_ge_kw,
                    max_pool_workers=int(retry.remote_max_pool_workers),
                )
            if len(response_items) != n_crops:
                raise RuntimeError(f"Expected {n_crops} GE responses, got {len(response_items)}")
            for ci, resp in enumerate(response_items):
                ge_results[ci] = [
                    d
                    for d in _remote_response_to_ge_detections(resp)
                    if (d.get("score") or 0.0) >= YOLOX_GRAPHIC_MIN_SCORE
                ]
        else:
            for ci, (_, _, crop_array) in enumerate(flat_crops):
                chw = torch.from_numpy(crop_array).permute(2, 0, 1).contiguous().to(dtype=torch.float32)
                h, w = crop_array.shape[:2]
                x = chw.unsqueeze(0)
                try:
                    pre = graphic_elements_model.preprocess(x)
                except Exception:
                    pre = x
                if isinstance(pre, torch.Tensor) and pre.ndim == 3:
                    pre = pre.unsqueeze(0)
                pred = graphic_elements_model.invoke(pre, (h, w))
                ge_dets = _prediction_to_detections(pred, label_names=label_names)
                ge_results[ci] = [d for d in ge_dets if (d.get("score") or 0.0) >= YOLOX_GRAPHIC_MIN_SCORE]
    except BaseException as e:
        print(f"Warning: graphic-elements batch inference failed: {type(e).__name__}: {e}")
        report_error("graphic_elements_ocr_page_elements:graphic_elements", e)
        err_payload = {
            "stage": "graphic_elements_ocr_page_elements:graphic_elements",
            "type": e.__class__.__name__,
            "message": str(e),
            "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
        }
        for row_i in set(crop_row_indices):
            all_meta[row_i]["error"] = err_payload

    # ---------------------------------------------------------------
    # Pass 3: Run OCR on ALL crops in one batched call.
    # ---------------------------------------------------------------
    ocr_results: List[Any] = [None] * n_crops
    try:
        if use_remote_ocr:
            _ocr_kw = dict(
                invoke_url=ocr_url,
                image_b64_list=flat_crop_b64s,
                api_key=api_key or None,
                timeout_s=float(request_timeout_s),
                max_batch_size=inference_batch_size,
                max_retries=int(retry.remote_max_retries),
                max_429_retries=int(retry.remote_max_429_retries),
            )
            if nim_client is not None:
                ocr_response_items = nim_client.invoke_image_inference_batches(**_ocr_kw)
            else:
                ocr_response_items = invoke_image_inference_batches(
                    **_ocr_kw,
                    max_pool_workers=int(retry.remote_max_pool_workers),
                )
            if len(ocr_response_items) != n_crops:
                raise RuntimeError(f"Expected {n_crops} OCR responses, got {len(ocr_response_items)}")
            for ci, resp in enumerate(ocr_response_items):
                ocr_results[ci] = _extract_remote_ocr_item(resp)
        else:
            for ci, (_, _, crop_array) in enumerate(flat_crops):
                ocr_results[ci] = ocr_model.invoke(crop_array, merge_level="word")
    except BaseException as e:
        print(f"Warning: chart OCR batch inference failed: {type(e).__name__}: {e}")
        report_error("graphic_elements_ocr_page_elements:ocr", e)
        err_payload = {
            "stage": "graphic_elements_ocr_page_elements:ocr",
            "type": e.__class__.__name__,
            "message": str(e),
            "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
        }
        for row_i in set(crop_row_indices):
            if all_meta[row_i]["error"] is None:
                all_meta[row_i]["error"] = err_payload

    # ---------------------------------------------------------------
    # Pass 4: Stitch results back to the correct rows.
    # ---------------------------------------------------------------
    for ci in range(n_crops):
        row_i = crop_row_indices[ci]
        label_name, bbox, crop_array = flat_crops[ci]
        crop_hw = (int(crop_array.shape[0]), int(crop_array.shape[1]))
        ge_dets = ge_results[ci]
        ocr_preds = ocr_results[ci]

        text = join_graphic_elements_and_ocr_output(ge_dets, ocr_preds, crop_hw)

        if not text:
            blocks = _parse_ocr_result(ocr_preds)
            text = _blocks_to_text(blocks)

        all_chart[row_i].append({"bbox_xyxy_norm": bbox, "text": text})

    elapsed = time.perf_counter() - t0_total
    for meta in all_meta:
        meta["timing"] = {"seconds": float(elapsed)}

    row_det_counts = [0] * num_rows
    for ci in range(n_crops):
        row_det_counts[crop_row_indices[ci]] += 1

    out = batch_df.copy()
    out["chart"] = all_chart
    out["graphic_elements_ocr_v1"] = all_meta
    out["graphic_elements_v1_num_detections"] = row_det_counts
    out["graphic_elements_v1_counts_by_label"] = [{"chart": c} if c > 0 else {} for c in row_det_counts]
    return out


# ---------------------------------------------------------------------------
# Combined graphic-elements + OCR Ray Actor
# ---------------------------------------------------------------------------
