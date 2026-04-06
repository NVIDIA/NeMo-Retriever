# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""General-purpose OpenAI-compatible chat completions client.

This module is model-agnostic and can be used with any endpoint that
implements the ``/v1/chat/completions`` contract (build.nvidia.com,
self-hosted NIMs, OpenAI, vLLM, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from nemo_retriever.nim.nim import _parse_invoke_urls, _post_with_retries, _mime_from_b64


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
