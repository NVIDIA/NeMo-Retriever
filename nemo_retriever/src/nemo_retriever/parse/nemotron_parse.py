# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Nemotron Parse v1.2 pipeline stage.

Runs the Nemotron Parse model on full page images to extract structured
document content (text, tables, charts, infographics) in a single pass,
replacing the page-elements → OCR multi-stage pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import base64
import io
import time
import traceback

import numpy as np
import pandas as pd

from nemo_retriever.model.local.nemotron_parse_postprocessing import (
    extract_classes_bboxes,
    postprocess_text as _postprocess_element_text,
)
from nemo_retriever.nim.nim import invoke_chat_completions, invoke_image_inference_batches
from nemo_retriever.params import RemoteRetryParams

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NEMOTRON_PARSE_DEFAULT_MODEL = "nvidia/nemotron-parse-v1.2"

# Map Nemotron Parse class labels to the pipeline content channels.
_PARSE_CLASS_TO_CHANNEL: Dict[str, str] = {
    "Table": "table",
    "Chart": "chart",
    "Picture": "infographic",
    "Infographic": "infographic",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error_payload(*, stage: str, exc: BaseException) -> Dict[str, Any]:
    return {
        "timing": None,
        "error": {
            "stage": str(stage),
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        },
    }


def _extract_parse_text(response_item: Any) -> str:
    """Extract text from a Nemotron Parse NIM response item."""
    if response_item is None:
        return ""
    if isinstance(response_item, str):
        return response_item.strip()
    if isinstance(response_item, dict):
        for key in ("generated_text", "text", "output_text", "prediction", "output", "data"):
            value = response_item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list) and value:
                first = value[0]
                if isinstance(first, str) and first.strip():
                    return first.strip()
                if isinstance(first, dict):
                    inner = _extract_parse_text(first)
                    if inner:
                        return inner
    if isinstance(response_item, list):
        for item in response_item:
            text = _extract_parse_text(item)
            if text:
                return text
    try:
        return str(response_item).strip()
    except Exception:
        return ""


def _route_parsed_elements(
    raw_text: str,
    *,
    extract_tables: bool,
    extract_charts: bool,
    extract_infographics: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    """Split raw Nemotron Parse output by element class into pipeline channels.

    Returns ``(table_items, chart_items, infographic_items, page_text)``
    where each ``*_items`` list contains ``{"bbox_xyxy_norm": ..., "text": ...}``
    dicts and ``page_text`` is the concatenated text of non-structured elements
    (or ``None`` if there are none).
    """
    classes, bboxes, texts = extract_classes_bboxes(raw_text)
    table_items: List[Dict[str, Any]] = []
    chart_items: List[Dict[str, Any]] = []
    infographic_items: List[Dict[str, Any]] = []
    text_parts: List[str] = []

    for cls, bbox, text in zip(classes, bboxes, texts):
        bbox_list = list(bbox)
        processed = _postprocess_element_text(text, cls=cls, table_format="markdown")
        if not processed:
            continue
        channel = _PARSE_CLASS_TO_CHANNEL.get(cls)
        entry = {"bbox_xyxy_norm": bbox_list, "text": processed}
        if channel == "table" and extract_tables:
            table_items.append(entry)
        elif channel == "chart" and extract_charts:
            chart_items.append(entry)
        elif channel == "infographic" and extract_infographics:
            infographic_items.append(entry)
        else:
            # Text, Title, Header_footer, Formula, etc. → page text
            text_parts.append(processed)

    page_text = "\n\n".join(text_parts) if text_parts else None
    return table_items, chart_items, infographic_items, page_text


def _decode_page_image(page_image_b64: str) -> np.ndarray:
    """Decode a base64 page image to an HWC uint8 numpy array."""
    raw = base64.b64decode(page_image_b64)
    with Image.open(io.BytesIO(raw)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


# ---------------------------------------------------------------------------
# Main stage function
# ---------------------------------------------------------------------------


def nemotron_parse_pages(
    batch_df: Any,
    *,
    model: Any = None,
    invoke_url: Optional[str] = None,
    api_key: Optional[str] = None,
    request_timeout_s: float = 120.0,
    extract_text: bool = False,
    extract_tables: bool = False,
    extract_charts: bool = False,
    extract_infographics: bool = False,
    nemotron_parse_model: Optional[str] = None,
    task_prompt: str = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>",
    remote_retry: RemoteRetryParams | None = None,
    **kwargs: Any,
) -> Any:
    """Run Nemotron Parse v1.2 on full page images.

    Each page is parsed in a single model call.  The structured output is
    split by element class (Text, Table, Chart, Picture, …) and routed to
    the corresponding pipeline content columns (``table``, ``chart``,
    ``infographic``).  Non-structured elements (headings, body text, …) are
    concatenated into the ``text`` column, replacing the upstream pdfium
    extraction when ``extract_text`` is ``True``.
    """
    retry = remote_retry or RemoteRetryParams(
        remote_max_pool_workers=int(kwargs.get("remote_max_pool_workers", 16)),
        remote_max_retries=int(kwargs.get("remote_max_retries", 10)),
        remote_max_429_retries=int(kwargs.get("remote_max_429_retries", 5)),
    )
    if not isinstance(batch_df, pd.DataFrame):
        raise NotImplementedError("nemotron_parse_pages currently only supports pandas.DataFrame input.")

    invoke_url = (invoke_url or kwargs.get("nemotron_parse_invoke_url") or "").strip()
    use_remote = bool(invoke_url)
    if not use_remote and model is None:
        raise ValueError("A local `model` is required when `invoke_url` is not provided.")

    n_rows = len(batch_df)
    all_table: List[List[Dict[str, Any]]] = [[] for _ in range(n_rows)]
    all_chart: List[List[Dict[str, Any]]] = [[] for _ in range(n_rows)]
    all_infographic: List[List[Dict[str, Any]]] = [[] for _ in range(n_rows)]
    all_text: List[Optional[str]] = [None] * n_rows
    all_meta: List[Dict[str, Any]] = [{"timing": None, "error": None} for _ in range(n_rows)]

    t0_total = time.perf_counter()

    # -- Phase 1: collect page images that need inference ----------------
    batch_indices: List[int] = []  # index into batch_df
    batch_images: List[Any] = []  # numpy arrays (local) or b64 strings (remote)

    for idx, row in enumerate(batch_df.itertuples(index=False)):
        page_image = getattr(row, "page_image", None) or {}
        page_image_b64 = page_image.get("image_b64") if isinstance(page_image, dict) else None
        if not isinstance(page_image_b64, str) or not page_image_b64:
            continue
        try:
            if use_remote:
                batch_images.append(page_image_b64)
            else:
                batch_images.append(_decode_page_image(page_image_b64))
            batch_indices.append(idx)
        except Exception as e:
            all_meta[idx] = {
                "timing": None,
                "error": {
                    "stage": "nemotron_parse_pages_decode",
                    "type": e.__class__.__name__,
                    "message": str(e),
                    "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
                },
            }

    # -- Phase 2: run model inference in a single batch ------------------
    raw_texts: List[str] = [""] * len(batch_indices)
    if batch_images:
        try:
            if use_remote:
                if "/v1/chat/completions" in invoke_url:
                    raw_texts = invoke_chat_completions(
                        invoke_url=invoke_url,
                        image_b64_list=batch_images,
                        model=nemotron_parse_model or NEMOTRON_PARSE_DEFAULT_MODEL,
                        api_key=api_key,
                        timeout_s=float(request_timeout_s),
                        task_prompt=task_prompt,
                        max_pool_workers=int(retry.remote_max_pool_workers),
                        max_retries=int(retry.remote_max_retries),
                        max_429_retries=int(retry.remote_max_429_retries),
                    )
                else:
                    response_items = invoke_image_inference_batches(
                        invoke_url=invoke_url,
                        image_b64_list=batch_images,
                        api_key=api_key,
                        timeout_s=float(request_timeout_s),
                        max_batch_size=int(kwargs.get("inference_batch_size", 8)),
                        max_pool_workers=int(retry.remote_max_pool_workers),
                        max_retries=int(retry.remote_max_retries),
                        max_429_retries=int(retry.remote_max_429_retries),
                    )
                    raw_texts = [_extract_parse_text(item) for item in response_items]
            else:
                invoke_batch = getattr(model, "invoke_batch", None)
                if invoke_batch is not None:
                    raw_texts = [str(t or "").strip() for t in invoke_batch(batch_images, task_prompt=task_prompt)]
                else:
                    raw_texts = [str(model.invoke(img, task_prompt=task_prompt) or "").strip() for img in batch_images]
        except BaseException as e:
            print(f"Warning: Nemotron Parse batch failed: {type(e).__name__}: {e}")
            err = {
                "stage": "nemotron_parse_pages",
                "type": e.__class__.__name__,
                "message": str(e),
                "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
            }
            for i in batch_indices:
                all_meta[i] = {"timing": None, "error": err}
            raw_texts = []

    # -- Phase 3: route parsed elements into content channels ------------
    for pos, raw_text in enumerate(raw_texts):
        idx = batch_indices[pos]
        try:
            fp_tables, fp_charts, fp_infographics, fp_text = _route_parsed_elements(
                raw_text,
                extract_tables=extract_tables,
                extract_charts=extract_charts,
                extract_infographics=extract_infographics,
            )
            all_table[idx] = fp_tables
            all_chart[idx] = fp_charts
            all_infographic[idx] = fp_infographics
            if fp_text is not None:
                all_text[idx] = fp_text
        except BaseException as e:
            all_meta[idx] = {
                "timing": None,
                "error": {
                    "stage": "nemotron_parse_pages_route",
                    "type": e.__class__.__name__,
                    "message": str(e),
                    "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
                },
            }

    elapsed = time.perf_counter() - t0_total
    for meta in all_meta:
        meta["timing"] = {"seconds": float(elapsed)}

    out = batch_df.copy()
    if extract_text and "text" in out.columns:
        for i, parse_text in enumerate(all_text):
            if parse_text is not None:
                out.iat[i, out.columns.get_loc("text")] = parse_text
    elif extract_text:
        out["text"] = [t if t is not None else "" for t in all_text]
    out["table"] = all_table
    out["chart"] = all_chart
    out["infographic"] = all_infographic
    out["table_parse"] = all_table
    out["chart_parse"] = all_chart
    out["infographic_parse"] = all_infographic
    out["nemotron_parse_v1_2"] = all_meta
    return out


# ---------------------------------------------------------------------------
# Ray actor
# ---------------------------------------------------------------------------


class NemotronParseActor:
    """Ray-friendly callable that initialises Nemotron Parse v1.2 once per actor."""

    __slots__ = (
        "_model",
        "_extract_text",
        "_extract_tables",
        "_extract_charts",
        "_extract_infographics",
        "_invoke_url",
        "_nemotron_parse_model",
        "_api_key",
        "_request_timeout_s",
        "_task_prompt",
        "_remote_retry",
    )

    def __init__(
        self,
        *,
        extract_text: bool = False,
        extract_tables: bool = False,
        extract_charts: bool = False,
        extract_infographics: bool = False,
        nemotron_parse_invoke_url: Optional[str] = None,
        nemotron_parse_model: Optional[str] = None,
        invoke_url: Optional[str] = None,
        api_key: Optional[str] = None,
        request_timeout_s: float = 120.0,
        task_prompt: str = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>",
        remote_max_pool_workers: int = 16,
        remote_max_retries: int = 10,
        remote_max_429_retries: int = 5,
    ) -> None:
        self._invoke_url = (nemotron_parse_invoke_url or invoke_url or "").strip()
        self._nemotron_parse_model = nemotron_parse_model
        self._api_key = api_key
        self._request_timeout_s = float(request_timeout_s)
        self._task_prompt = str(task_prompt)
        self._remote_retry = RemoteRetryParams(
            remote_max_pool_workers=int(remote_max_pool_workers),
            remote_max_retries=int(remote_max_retries),
            remote_max_429_retries=int(remote_max_429_retries),
        )
        if self._invoke_url:
            self._model = None
        else:
            from nemo_retriever.model.local import NemotronParseV12

            self._model = NemotronParseV12(task_prompt=self._task_prompt)
        self._extract_text = bool(extract_text)
        self._extract_tables = bool(extract_tables)
        self._extract_charts = bool(extract_charts)
        self._extract_infographics = bool(extract_infographics)

    def __call__(self, batch_df: Any, **override_kwargs: Any) -> Any:
        try:
            return nemotron_parse_pages(
                batch_df,
                model=self._model,
                invoke_url=self._invoke_url,
                nemotron_parse_model=self._nemotron_parse_model,
                api_key=self._api_key,
                request_timeout_s=self._request_timeout_s,
                task_prompt=self._task_prompt,
                extract_text=self._extract_text,
                extract_tables=self._extract_tables,
                extract_charts=self._extract_charts,
                extract_infographics=self._extract_infographics,
                remote_retry=self._remote_retry,
                **override_kwargs,
            )
        except BaseException as e:
            if isinstance(batch_df, pd.DataFrame):
                out = batch_df.copy()
                payload = _error_payload(stage="nemotron_parse_actor_call", exc=e)
                n = len(out.index)
                out["table"] = [[] for _ in range(n)]
                out["chart"] = [[] for _ in range(n)]
                out["infographic"] = [[] for _ in range(n)]
                out["table_parse"] = [[] for _ in range(n)]
                out["chart_parse"] = [[] for _ in range(n)]
                out["infographic_parse"] = [[] for _ in range(n)]
                out["nemotron_parse_v1_2"] = [payload for _ in range(n)]
                return out
            return [{"nemotron_parse_v1_2": _error_payload(stage="nemotron_parse_actor_call", exc=e)}]
