# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replace detected chart crops with Parse 3.5 content."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from nemo_retriever.common.modality.ocr.shared import _crop_all_from_page
from nemo_retriever.common.modality.parse.nemotron_parse35_postprocessing import _postprocess_parse35_blocks
from nemo_retriever.common.params import ParseParams, RemoteRetryParams
from nemo_retriever.common.params.utils import NEMOTRON_PARSE35_MODEL, validate_nemotron_parse_endpoint_list
from nemo_retriever.models.nim.nim import NIMClient

_CHART_ROUTING_SAMPLING = {
    "temperature": 0.1,
    "repetition_penalty": 1.0,
    "top_k": 50,
    "top_p": 1.0,
    "seed": 0,
    "max_tokens": 8744,
}
_CHART_ROUTING_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"


def _items(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple, np.ndarray)) else []


def _same_box(left: Any, right: Any) -> bool:
    try:
        return len(left) == len(right) == 4 and bool(np.allclose(left, right, rtol=0, atol=1e-7))
    except (TypeError, ValueError):
        return False


def route_chart_regions(
    batch_df: pd.DataFrame,
    *,
    model: Any = None,
    invoke_url: str | None = None,
    api_key: str | None = None,
    request_timeout_s: float = 120.0,
    remote_retry: RemoteRetryParams | None = None,
    nim_client: NIMClient | None = None,
    table_format: str | None = None,
    sampling_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Parse Page Elements chart crops and replace only their chart content.

    Args:
        batch_df: Page rows with ``page_image.image_b64`` and
            ``page_elements_v3.detections``. Detection entries require
            ``label_name`` and normalized ``bbox_xyxy_norm`` coordinates.
            Existing ``chart`` entries use the same coordinates and a ``text`` field.
        model: Local Parse 3.5 adapter configured with the routing sampling settings.
            Required for pages containing charts when no remote URL is supplied.
        invoke_url: Optional comma-separated Parse 3.5 chat-completions URLs.
        api_key: Optional remote service credential.
        request_timeout_s: Timeout for each remote request, in seconds.
        remote_retry: Remote concurrency and retry limits.
        nim_client: Optional reusable remote client. Clients created by this function
            are closed before it returns.
        table_format: Table block rendering: HTML, simple_html (default), markdown,
            or latex. Other block classes are unchanged.
        sampling_params: Optional generation overrides merged with crop defaults:
            temperature=0.1, repetition_penalty=1.0, top_k=50, top_p=1.0, seed=0,
            max_tokens=8744. A supplied local adapter must already use these settings.

    Returns:
        A copy with chart texts replaced at their original list positions and a
        ``chart_routing`` column containing per-crop raw outputs and errors.
        Non-chart content, row order, and original boxes are preserved. The returned
        ``ordered_regions`` and ``ordered_page_text`` fields contain the updated chart text.
        New chart entries are appended only when a detection has no existing OCR entry.
        All Parse blocks remain in model order; only Table blocks use the selected rendering format.
        Successful empty output clears the old chart text. Inference or formatting
        failures preserve that crop's OCR text and record the error in ``chart_routing``.
        Crops use RGB page pixels, clipped integer boxes, and no added margin.
        Invalid crops and crops smaller than two pixels in either dimension are skipped.

    Raises:
        TypeError: If batch_df is not a pandas DataFrame.
        ValueError: If no local model is supplied for chart inference, or remote
            URLs do not use the chat-completions contract.
    """
    if not isinstance(batch_df, pd.DataFrame):
        raise TypeError("route_chart_regions requires a pandas.DataFrame of extracted page rows.")
    options = ParseParams(table_format=table_format, **(sampling_params or {}))
    sampling = {**_CHART_ROUTING_SAMPLING, **options._sampling_overrides()}
    endpoints = validate_nemotron_parse_endpoint_list(invoke_url)
    if any(not url.rstrip("/").endswith("/v1/chat/completions") for url in endpoints):
        raise ValueError("Chart routing requires Parse 3.5 /v1/chat/completions endpoints.")
    remote = bool(endpoints)
    charts = [deepcopy(_items(value)) for value in batch_df.get("chart", [None] * len(batch_df))]
    metadata: list[dict[str, Any]] = [{"regions": []} for _ in range(len(batch_df))]
    jobs: list[tuple[int, list[float], Any]] = []
    for position, (_, row) in enumerate(batch_df.iterrows()):
        layout = row.get("page_elements_v3")
        if not isinstance(layout, dict):
            continue
        detections = _items(layout.get("detections"))
        selected = [
            {**d, "bbox_xyxy_norm": _items(d.get("bbox_xyxy_norm"))}
            for d in detections
            if isinstance(d, dict) and d.get("label_name") == "chart"
        ]
        if not selected:
            continue
        page = row.get("page_image")
        image = page.get("image_b64") if isinstance(page, dict) else None
        crops = _crop_all_from_page(image, selected, {"chart"}, as_b64=remote)
        jobs.extend((position, box, crop) for _, box, crop in crops)
        if len(crops) < len(selected):
            metadata[position]["skipped_crops"] = len(selected) - len(crops)
    out = batch_df.copy()
    if not jobs:
        out["chart_routing"] = metadata
        return out
    if not remote and model is None:
        raise ValueError("Chart routing requires a local Parse 3.5 model or nemotron_parse_invoke_url.")
    retry = remote_retry or RemoteRetryParams()
    owned = remote and nim_client is None
    client = nim_client or (NIMClient(max_pool_workers=retry.remote_max_pool_workers) if remote else None)
    try:
        if remote:
            raw_outputs = client.invoke_chat_completions_images(
                invoke_url=invoke_url,
                image_b64_list=[crop for _, _, crop in jobs],
                model=NEMOTRON_PARSE35_MODEL,
                task_prompt=_CHART_ROUTING_PROMPT,
                api_key=api_key,
                timeout_s=request_timeout_s,
                temperature=sampling["temperature"],
                repetition_penalty=sampling["repetition_penalty"],
                extra_body={
                    **{k: v for k, v in sampling.items() if k not in ("temperature", "repetition_penalty")},
                    "skip_special_tokens": False,
                },
                max_retries=retry.remote_max_retries,
                max_429_retries=retry.remote_max_429_retries,
            )
        else:
            raw_outputs = model.invoke_batch([crop for _, _, crop in jobs], task_prompt=_CHART_ROUTING_PROMPT)
        if len(raw_outputs) != len(jobs):
            raise ValueError(
                f"Parse returned {len(raw_outputs)} outputs for {len(jobs)} chart crops; check the endpoint."
            )
    except Exception as exc:
        for position, box, _ in jobs:
            metadata[position]["regions"].append({"bbox_xyxy_norm": box, "error": str(exc), "type": type(exc).__name__})
        out["chart_routing"] = metadata
        return out
    finally:
        if owned:
            client.shutdown()
    used_entries: dict[int, set[int]] = {}
    for (position, box, _), raw in zip(jobs, raw_outputs):
        record: dict[str, Any] = {"bbox_xyxy_norm": box, "raw_output": raw, "error": None}
        try:
            blocks = _postprocess_parse35_blocks(raw, table_format=options.table_format or "simple_html")
            text = "\n".join(block["content"] for block in blocks if block.get("content"))
            used = used_entries.setdefault(position, set())
            index = next(
                (
                    i
                    for i, item in enumerate(charts[position])
                    if i not in used and isinstance(item, dict) and _same_box(item.get("bbox_xyxy_norm"), box)
                ),
                None,
            )
            entry = charts[position][index] if index is not None else None
            if index is not None:
                used.add(index)
            if entry is None:
                charts[position].append({"bbox_xyxy_norm": box, "text": text})
            else:
                entry["text"] = text
        except Exception as exc:
            record.update(error=str(exc), type=type(exc).__name__)
        metadata[position]["regions"].append(record)
    if "ordered_regions" in out.columns:
        updated = []
        for position, value in enumerate(out["ordered_regions"]):
            regions = deepcopy(_items(value))
            for region in regions:
                if isinstance(region, dict) and region.get("label_name") == "chart":
                    match = next(
                        (
                            item
                            for item in charts[position]
                            if isinstance(item, dict)
                            and _same_box(item.get("bbox_xyxy_norm"), region.get("bbox_xyxy_norm"))
                        ),
                        None,
                    )
                    if match is not None:
                        region["text"] = match["text"]
            updated.append(regions)
        out["ordered_regions"] = updated
        if "ordered_page_text" in out.columns:
            out["ordered_page_text"] = [
                "\n".join(
                    r["text"] for r in regions if isinstance(r, dict) and isinstance(r.get("text"), str) and r["text"]
                )
                for regions in updated
            ]
    out["chart"] = charts
    out["chart_routing"] = metadata
    return out
