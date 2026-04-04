# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

logger = logging.getLogger(__name__)


def _chunk_ranges(total: int, chunk_size: int) -> List[Tuple[int, int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    return [(i, min(i + chunk_size, total)) for i in range(0, total, chunk_size)]


def _parse_invoke_urls(invoke_url: str) -> List[str]:
    parts = [p.strip() for p in str(invoke_url or "").split(",")]
    urls = [p for p in parts if p]
    if not urls:
        raise ValueError("invoke_url is required")
    return urls


def _normalize_batch_response(response_json: Any, expected_count: int) -> List[Any]:
    """
    Normalize common NIM HTTP response shapes to one item per input image.
    """
    if isinstance(response_json, list):
        items = response_json
    elif isinstance(response_json, dict):
        data = response_json.get("data")
        output = response_json.get("output")
        predictions = response_json.get("predictions")
        if isinstance(data, list):
            items = data
        elif isinstance(output, list):
            items = output
        elif isinstance(predictions, list):
            items = predictions
        else:
            items = [response_json]
    else:
        raise RuntimeError(f"Unexpected response type: {type(response_json)!r}")

    out: List[Any] = []
    for item in items:
        out.append(item)

    if len(out) != int(expected_count):
        raise RuntimeError(f"Remote response count mismatch: expected {expected_count}, got {len(out)}")
    return out


def _post_with_retries(
    *,
    invoke_url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout_s: float,
    max_retries: int,
    max_429_retries: int,
) -> Any:
    base_delay = 2.0
    attempt = 0
    retries_429 = 0

    while attempt < int(max_retries):
        try:
            response = requests.post(invoke_url, headers=headers, json=payload, timeout=float(timeout_s))
            status_code = response.status_code

            if status_code == 429:
                retries_429 += 1
                if retries_429 >= int(max_429_retries):
                    response.raise_for_status()
                backoff_time = base_delay * (2**retries_429)
                logger.warning(
                    "NIM endpoint %s returned 429 (rate limited). Retry %d/%d after %.1fs backoff.",
                    invoke_url,
                    retries_429,
                    max_429_retries,
                    backoff_time,
                )
                time.sleep(backoff_time)
                continue

            if status_code == 503 or (500 <= status_code < 600):
                if attempt == int(max_retries) - 1:
                    response.raise_for_status()
                backoff_time = base_delay * (2**attempt)
                logger.warning(
                    "NIM endpoint %s returned %d. Retry %d/%d after %.1fs backoff.",
                    invoke_url,
                    status_code,
                    attempt + 1,
                    max_retries,
                    backoff_time,
                )
                time.sleep(backoff_time)
                attempt += 1
                continue

            if 400 <= status_code < 500:
                raise requests.HTTPError(
                    f"HTTP {status_code} from {invoke_url}: {response.text}",
                    response=response,
                )
            response.raise_for_status()
            return response.json()

        except requests.Timeout as exc:
            if attempt == int(max_retries) - 1:
                raise TimeoutError(f"Request timed out after {attempt + 1} attempts.") from exc
            backoff_time = base_delay * (2**attempt)
            logger.warning(
                "NIM endpoint %s timed out (%.1fs). Retry %d/%d after %.1fs backoff.",
                invoke_url,
                timeout_s,
                attempt + 1,
                max_retries,
                backoff_time,
            )
            time.sleep(backoff_time)
            attempt += 1
        except requests.RequestException as exc:
            resp = getattr(exc, "response", None)
            if resp is not None and 400 <= resp.status_code < 500:
                raise  # client errors are not retryable
            if attempt == int(max_retries) - 1:
                raise
            backoff_time = base_delay * (2**attempt)
            logger.warning(
                "NIM endpoint %s request failed: %s. Retry %d/%d after %.1fs backoff.",
                invoke_url,
                exc,
                attempt + 1,
                max_retries,
                backoff_time,
            )
            time.sleep(backoff_time)
            attempt += 1

    raise RuntimeError(f"Failed to get a successful response after {max_retries} retries.")


def _mime_from_b64(b64: str) -> str:
    """Detect MIME type from the first bytes of a base64-encoded image."""
    if b64.startswith("/9j"):  # JPEG magic bytes FF D8 FF
        return "image/jpeg"
    return "image/png"


def invoke_image_inference_batches(
    *,
    invoke_url: str,
    image_b64_list: Sequence[str],
    api_key: Optional[str] = None,
    timeout_s: float = 60.0,
    max_batch_size: int = 8,
    max_pool_workers: int = 8,
    max_retries: int = 5,
    max_429_retries: int = 3,
) -> List[Any]:
    """
    Invoke one or more image NIM HTTP endpoints with batched concurrent requests.

    `invoke_url` may be a single URL or a comma-separated URL list.
    When multiple URLs are provided, batches are distributed round-robin.

    Returns one response item per input image, in the same order.
    """
    invoke_urls = _parse_invoke_urls(invoke_url)

    token = (api_key or "").strip()
    headers: Dict[str, str] = {"Accept": "application/json", "Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    n = len(image_b64_list)
    if n == 0:
        return []

    ranges = _chunk_ranges(n, int(max_batch_size))
    flattened: List[Optional[Any]] = [None] * n

    def _invoke_one_batch(start: int, end: int, endpoint_url: str) -> Tuple[int, int, List[Any]]:
        inputs = [
            {
                "type": "image_url",
                "url": f"data:{_mime_from_b64(b64)};base64,{b64}",
            }
            for b64 in image_b64_list[start:end]
        ]
        payload = {"input": inputs}
        response_json = _post_with_retries(
            invoke_url=endpoint_url,
            payload=payload,
            headers=headers,
            timeout_s=float(timeout_s),
            max_retries=int(max_retries),
            max_429_retries=int(max_429_retries),
        )
        per_image = _normalize_batch_response(response_json, end - start)
        return start, end, per_image

    with ThreadPoolExecutor(max_workers=max(1, int(max_pool_workers))) as executor:
        futures = {
            executor.submit(
                _invoke_one_batch,
                start,
                end,
                invoke_urls[idx % len(invoke_urls)],
            ): (start, end)
            for idx, (start, end) in enumerate(ranges)
        }
        for future in as_completed(futures):
            start, end = futures[future]
            _s, _e, per_image = future.result()
            if _s != start or _e != end:
                raise RuntimeError("Internal batch ordering mismatch.")
            for i, item in enumerate(per_image):
                flattened[start + i] = item

    out: List[Any] = []
    for idx, item in enumerate(flattened):
        if item is None:
            raise RuntimeError(f"Missing response for item index {idx}")
        out.append(item)
    return out


def extract_chat_completion_text(response_json: Any) -> str:
    """Extract generated text from an OpenAI-compatible chat completions response."""
    try:
        choice = response_json["choices"][0]["message"]
        # Some models return output via tool_calls
        tool_calls = choice.get("tool_calls")
        if tool_calls:
            return str(tool_calls[0]["function"]["arguments"]).strip()
        content = choice.get("content")
        if content:
            return str(content).strip()
    except (KeyError, IndexError, TypeError):
        pass
    return ""


def invoke_chat_completions(
    *,
    invoke_url: str,
    messages_list: Sequence[List[Dict[str, Any]]],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_s: float = 120.0,
    temperature: float = 0.0,
    extra_body: Optional[Dict[str, Any]] = None,
    max_pool_workers: int = 16,
    max_retries: int = 10,
    max_429_retries: int = 5,
) -> List[str]:
    """Invoke an OpenAI-compatible chat completions endpoint concurrently.

    Parameters
    ----------
    messages_list
        A sequence of OpenAI-format message lists, one per request.
        Each entry is passed as the ``"messages"`` field in the payload.
    model
        Optional model identifier included in each request payload.
    extra_body
        Additional top-level keys merged into every request payload
        (e.g. ``{"repetition_penalty": 1.1, "max_tokens": 9000}``).

    Returns one extracted text string per entry in *messages_list*, in order.
    """
    if not messages_list:
        return []

    token = (api_key or "").strip()
    headers: Dict[str, str] = {"Accept": "application/json", "Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    invoke_urls = _parse_invoke_urls(invoke_url)
    results: List[Optional[str]] = [None] * len(messages_list)

    def _invoke_one(idx: int, messages: List[Dict[str, Any]], endpoint_url: str) -> Tuple[int, str]:
        payload: Dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
        }
        if model:
            payload["model"] = model
        if extra_body:
            payload.update(extra_body)
        response_json = _post_with_retries(
            invoke_url=endpoint_url,
            payload=payload,
            headers=headers,
            timeout_s=float(timeout_s),
            max_retries=int(max_retries),
            max_429_retries=int(max_429_retries),
        )
        return idx, extract_chat_completion_text(response_json)

    with ThreadPoolExecutor(max_workers=max(1, int(max_pool_workers))) as executor:
        futures = {
            executor.submit(_invoke_one, i, msgs, invoke_urls[i % len(invoke_urls)]): i
            for i, msgs in enumerate(messages_list)
        }
        for future in as_completed(futures):
            i, text = future.result()
            results[i] = text

    return [r if r is not None else "" for r in results]


def invoke_chat_completions_images(
    *,
    invoke_url: str,
    image_b64_list: Sequence[str],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_s: float = 120.0,
    task_prompt: Optional[str] = None,
    temperature: float = 0.0,
    repetition_penalty: float = 1.1,
    extra_body: Optional[Dict[str, Any]] = None,
    max_pool_workers: int = 16,
    max_retries: int = 10,
    max_429_retries: int = 5,
) -> List[str]:
    """Convenience wrapper: one chat completion request per base64 image.

    Builds an OpenAI-format ``image_url`` message for each image and
    delegates to :func:`invoke_chat_completions`.
    """
    if not image_b64_list:
        return []

    messages_list: List[List[Dict[str, Any]]] = []
    for b64 in image_b64_list:
        mime = _mime_from_b64(b64)
        content: List[Dict[str, Any]] = []
        if task_prompt:
            content.append({"type": "text", "text": task_prompt})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            }
        )
        messages_list.append([{"role": "user", "content": content}])

    merged_extra: Dict[str, Any] = {"repetition_penalty": repetition_penalty}
    if extra_body:
        merged_extra.update(extra_body)

    return invoke_chat_completions(
        invoke_url=invoke_url,
        messages_list=messages_list,
        model=model,
        api_key=api_key,
        timeout_s=timeout_s,
        temperature=temperature,
        extra_body=merged_extra,
        max_pool_workers=max_pool_workers,
        max_retries=max_retries,
        max_429_retries=max_429_retries,
    )


def invoke_page_elements_batches(
    *,
    invoke_url: str,
    image_b64_list: Sequence[str],
    api_key: Optional[str] = None,
    timeout_s: float = 60.0,
    max_batch_size: int = 8,
    max_pool_workers: int = 8,
    max_retries: int = 5,
    max_429_retries: int = 3,
) -> List[Any]:
    """Backward-compatible alias for page-elements callers."""
    return invoke_image_inference_batches(
        invoke_url=invoke_url,
        image_b64_list=image_b64_list,
        api_key=api_key,
        timeout_s=timeout_s,
        max_batch_size=max_batch_size,
        max_pool_workers=max_pool_workers,
        max_retries=max_retries,
        max_429_retries=max_429_retries,
    )
