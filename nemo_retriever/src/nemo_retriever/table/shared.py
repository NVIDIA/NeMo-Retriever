# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, List, Optional

import time
import traceback

import pandas as pd
from nemo_retriever.params import RemoteRetryParams

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from nv_ingest_api.internal.primitives.nim.model_interface.yolox import (
        YOLOX_TABLE_MIN_SCORE,
    )
except ImportError:
    YOLOX_TABLE_MIN_SCORE = 0.1  # type: ignore[assignment]

_DEFAULT_TABLE_STRUCTURE_LABELS: List[str] = ["cell", "row", "column"]


def _labels_from_model(model: Any) -> List[str]:
    # Prefer underlying model labels if present.
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
    """
    Best-effort conversion of model output into a standard detection list.

    Produces dicts of the form:
      {"bbox_xyxy_norm": [...], "label": int|None, "label_name": str, "score": float|None}
    """
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

    # Normalize to torch tensors.
    def _to_tensor(x: Any) -> Optional["torch.Tensor"]:
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        try:
            return torch.as_tensor(x).detach().cpu()
        except Exception:
            return None

    # Handle string labels (e.g. NIM returns ["cell", "row", "column", ...]).
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

    # Expect boxes (N,4), labels (N,)
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


def _parse_nim_bounding_boxes(response_item: Any) -> List[Dict[str, Any]]:
    """Parse the ``bounding_boxes`` NIM response format.

    NIM table-structure endpoints return::

        {"index": 0, "bounding_boxes": {
            "cell": [{"x_min": ..., "y_min": ..., "x_max": ..., "y_max": ..., "confidence": ...}, ...],
            "row":  [...],
            "column": [...]
        }}

    Returns a flat list of detection dicts compatible with
    ``_structure_dets_to_class_boxes``.
    """
    bb = None
    if isinstance(response_item, dict):
        bb = response_item.get("bounding_boxes")
    if not isinstance(bb, dict):
        return []

    dets: List[Dict[str, Any]] = []
    for label_name, items in bb.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                bbox = [float(item["x_min"]), float(item["y_min"]), float(item["x_max"]), float(item["y_max"])]
            except (KeyError, TypeError, ValueError):
                continue
            score = None
            try:
                score = float(item["confidence"])
            except (KeyError, TypeError, ValueError):
                pass
            dets.append(
                {
                    "bbox_xyxy_norm": bbox,
                    "label_name": str(label_name),
                    "score": score,
                }
            )
    return dets


def _extract_remote_pred_item(response_item: Any) -> Any:
    if isinstance(response_item, dict):
        for k in ("prediction", "predictions", "output", "outputs", "data"):
            v = response_item.get(k)
            if isinstance(v, list) and v:
                return v[0]
            if v is not None:
                return v
    return response_item


def _count_structure_labels(structure_dets: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"cell": 0, "row": 0, "column": 0}
    for det in structure_dets:
        label_name = str(det.get("label_name") or "")
        if label_name in counts:
            counts[label_name] += 1
    return counts


def _render_structure_only_text(structure_dets: List[Dict[str, Any]], *, table_output_format: Optional[str]) -> str:
    counts = _count_structure_labels(structure_dets)
    output_format = str(table_output_format or "markdown").strip().lower()

    if output_format == "markdown" and counts["row"] > 0 and counts["column"] > 0:
        from nemo_retriever.utils.table_and_chart import display_markdown

        blank_matrix = [["" for _ in range(counts["column"])] for _ in range(counts["row"])]
        return display_markdown(blank_matrix, use_header=True)

    parts: List[str] = []
    for label_name in ("row", "column", "cell"):
        count = counts[label_name]
        if count:
            suffix = "" if count == 1 else "s"
            parts.append(f"{count} {label_name}{suffix}")

    if not parts:
        return ""

    return "Table structure detected: " + ", ".join(parts) + "."


# ---------------------------------------------------------------------------
# Combined table-structure core function
# ---------------------------------------------------------------------------


def table_structure_ocr_page_elements(
    batch_df: Any,
    *,
    table_structure_model: Any = None,
    table_structure_invoke_url: str = "",
    api_key: str = "",
    request_timeout_s: float = 120.0,
    remote_retry: RemoteRetryParams | None = None,
    **kwargs: Any,
) -> Any:
    """
    Run table-structure inference on table crops and produce structure-only output.

    For each row (page) in ``batch_df``:
    1. Read ``page_elements_v3`` detections and ``page_image["image_b64"]``.
    2. Crop all table detections from the page image.
    3. Run table-structure model on each crop to get cell/row/column bboxes.
    4. Render a structure-only summary or blank markdown skeleton from the
       detected rows/columns/cells.

    Parameters
    ----------
    batch_df : pandas.DataFrame
        Ray Data batch with ``page_elements_v3`` and ``page_image`` columns.
    table_structure_model : NemotronTableStructureV1 | None
        Local table-structure model, or None for remote inference.
    table_structure_invoke_url : str
        Remote NIM endpoint for table-structure inference.

    Returns
    -------
    pandas.DataFrame
        Original columns plus ``table`` and ``table_structure_ocr_v1``.
    """
    from nemo_retriever.nim.nim import invoke_image_inference_batches
    from nemo_retriever.ocr.ocr import _crop_all_from_page, _np_rgb_to_b64_png

    retry = remote_retry or RemoteRetryParams(
        remote_max_pool_workers=int(kwargs.get("remote_max_pool_workers", 16)),
        remote_max_retries=int(kwargs.get("remote_max_retries", 10)),
        remote_max_429_retries=int(kwargs.get("remote_max_429_retries", 5)),
    )

    if not isinstance(batch_df, pd.DataFrame):
        raise NotImplementedError("table_structure_ocr_page_elements currently only supports pandas.DataFrame input.")

    ts_url = (table_structure_invoke_url or kwargs.get("table_structure_invoke_url") or "").strip()
    use_remote_ts = bool(ts_url)
    table_output_format = kwargs.get("table_output_format")

    if not use_remote_ts and table_structure_model is None:
        raise ValueError("A local `table_structure_model` is required when `table_structure_invoke_url` is not set.")

    label_names = _labels_from_model(table_structure_model) if table_structure_model is not None else []
    if not label_names:
        label_names = _DEFAULT_TABLE_STRUCTURE_LABELS
    inference_batch_size = int(kwargs.get("inference_batch_size", 8))

    # Per-row accumulators.
    all_table: List[List[Dict[str, Any]]] = []
    all_meta: List[Dict[str, Any]] = []

    t0_total = time.perf_counter()

    for row in batch_df.itertuples(index=False):
        table_items: List[Dict[str, Any]] = []
        row_error: Any = None

        try:
            # --- get page elements detections ---
            pe = getattr(row, "page_elements_v3", None)
            dets: List[Dict[str, Any]] = []
            if isinstance(pe, dict):
                dets = pe.get("detections") or []
            if not isinstance(dets, list):
                dets = []

            # --- get page image ---
            page_image = getattr(row, "page_image", None) or {}
            page_image_b64 = page_image.get("image_b64") if isinstance(page_image, dict) else None

            if not isinstance(page_image_b64, str) or not page_image_b64:
                all_table.append(table_items)
                all_meta.append({"timing": None, "error": None})
                continue

            # --- Pass 1: Collect table crops ---
            crops = _crop_all_from_page(page_image_b64, dets, {"table"})

            if not crops:
                all_table.append(table_items)
                all_meta.append({"timing": None, "error": None})
                continue

            # Pre-compute base64 encodings once for remote paths.
            crop_b64s = [_np_rgb_to_b64_png(crop_array) for _, _, crop_array in crops] if use_remote_ts else []

            # --- Pass 2: Run table-structure on all crops ---
            structure_results: List[List[Dict[str, Any]]] = []
            if use_remote_ts:
                response_items = invoke_image_inference_batches(
                    invoke_url=ts_url,
                    image_b64_list=crop_b64s,
                    api_key=api_key or None,
                    timeout_s=float(request_timeout_s),
                    max_batch_size=inference_batch_size,
                    max_pool_workers=int(retry.remote_max_pool_workers),
                    max_retries=int(retry.remote_max_retries),
                    max_429_retries=int(retry.remote_max_429_retries),
                )
                if len(response_items) != len(crops):
                    raise RuntimeError(f"Expected {len(crops)} table-structure responses, got {len(response_items)}")
                for resp in response_items:
                    # Try NIM bounding_boxes format first, fall back to generic parser.
                    parsed = _parse_nim_bounding_boxes(resp)
                    if not parsed:
                        pred_item = _extract_remote_pred_item(resp)
                        parsed = _prediction_to_detections(pred_item, label_names=label_names)
                    structure_results.append([d for d in parsed if (d.get("score") or 0.0) >= YOLOX_TABLE_MIN_SCORE])
            else:
                # Local batched inference.
                for _, _, crop_array in crops:
                    chw = torch.from_numpy(crop_array).permute(2, 0, 1).contiguous().to(dtype=torch.float32)
                    h, w = crop_array.shape[:2]
                    x = chw.unsqueeze(0)  # BCHW
                    try:
                        pre = table_structure_model.preprocess(x, (h, w))
                    except TypeError:
                        pre = table_structure_model.preprocess(x)
                    if isinstance(pre, torch.Tensor) and pre.ndim == 3:
                        pre = pre.unsqueeze(0)
                    pred = table_structure_model.invoke(pre, (h, w))
                    dets = _prediction_to_detections(pred, label_names=label_names)
                    structure_results.append([d for d in dets if (d.get("score") or 0.0) >= YOLOX_TABLE_MIN_SCORE])

            # --- Pass 3: Build structure-only output per crop ---
            for crop_i, (_, bbox, _) in enumerate(crops):
                structure_dets = structure_results[crop_i]
                table_items.append(
                    {
                        "bbox_xyxy_norm": bbox,
                        "text": _render_structure_only_text(
                            structure_dets,
                            table_output_format=table_output_format,
                        ),
                        "structure_detections": structure_dets,
                        "structure_counts": _count_structure_labels(structure_dets),
                    }
                )

        except BaseException as e:
            print(f"Warning: table-structure failed: {type(e).__name__}: {e}")
            row_error = {
                "stage": "table_structure_ocr_page_elements",
                "type": e.__class__.__name__,
                "message": str(e),
                "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
            }

        all_table.append(table_items)
        all_meta.append({"timing": None, "error": row_error})

    elapsed = time.perf_counter() - t0_total
    for meta in all_meta:
        meta["timing"] = {"seconds": float(elapsed)}

    out = batch_df.copy()
    out["table"] = all_table
    out["table_structure_ocr_v1"] = all_meta
    return out


# ---------------------------------------------------------------------------
# Combined table-structure + OCR Ray Actor
# ---------------------------------------------------------------------------
