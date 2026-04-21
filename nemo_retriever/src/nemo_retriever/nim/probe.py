# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time
import urllib.parse
from typing import Optional

import requests

logger = logging.getLogger(__name__)


def probe_endpoint(
    url: str,
    *,
    name: str,
    prefix: str,
    api_key: Optional[str] = None,
    timeout: float = 5.0,
    post_url: Optional[str] = None,
    post_body: Optional[dict] = None,
) -> None:
    """Probe a NIM endpoint at actor startup to catch auth failures early.

    1. GET /v1/health/ready (no auth) — local NIMs expose this and return 2xx
       when healthy; return immediately.
    2. If the health path returns non-2xx (e.g. remote cloud endpoints at
       ai.api.nvidia.com where /v1/health/ready doesn't exist), and an api_key
       is provided, POST with an empty body to the actual endpoint URL. Remote
       endpoints validate the Bearer token before the body, so a revoked or
       missing key returns 401 without consuming inference quota.
    3. Raise RuntimeError on 401/403 so actor __init__ fails immediately rather
       than silently producing zero results later.

    All other errors (connection refused, timeout, other 4xx/5xx) are logged
    and tolerated — transient issues shouldn't block startup.
    """
    parsed = urllib.parse.urlparse(url)
    health_url = f"{parsed.scheme}://{parsed.netloc}/v1/health/ready"

    # Step 1: unauthenticated health check
    try:
        t0 = time.perf_counter()
        resp = requests.get(health_url, timeout=timeout)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "%s: %s endpoint %s responded %d in %.0fms",
            prefix, name, health_url, resp.status_code, elapsed_ms,
        )
        if resp.ok:
            return
    except requests.ConnectionError:
        logger.warning(
            "%s: %s endpoint %s is UNREACHABLE (connection refused). "
            "Processing will stall until this endpoint becomes available.",
            prefix, name, health_url,
        )
        return
    except requests.Timeout:
        logger.warning(
            "%s: %s endpoint %s timed out after %.1fs. "
            "The endpoint may be overloaded or not ready.",
            prefix, name, health_url, timeout,
        )
        return
    except Exception as exc:
        logger.debug("%s: %s endpoint probe %s failed: %s", prefix, name, health_url, exc)

    # Step 2: authenticated probe of the actual endpoint URL.
    # Only reached when the health path returned non-2xx (e.g. 404 on
    # ai.api.nvidia.com where /v1/health/ready doesn't exist).
    # Local NIMs authenticate at container startup, not per-request, so they
    # expose /v1/health/ready and we return early above. This step only applies
    # to remote cloud endpoints (ai.api.nvidia.com) that require a Bearer token.
    if not api_key:
        return

    # POST to the endpoint with the Bearer token. Remote endpoints validate auth
    # before the body, so a bad key returns 401 without triggering inference.
    # Callers can override post_url (e.g. append /embeddings) and post_body
    # (e.g. include model name) when the base URL alone doesn't route correctly.
    target = post_url or url
    body = post_body if post_body is not None else {}
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        t0 = time.perf_counter()
        resp = requests.post(target, headers=headers, json=body, timeout=timeout)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "%s: %s endpoint %s responded %d in %.0fms",
            prefix, name, target, resp.status_code, elapsed_ms,
        )
        if resp.status_code in (401, 403):
            raise RuntimeError(
                f"{prefix}: authentication failed for {name} endpoint {target} "
                f"(HTTP {resp.status_code}) — verify the API key is valid."
            )
    except RuntimeError:
        raise
    except requests.ConnectionError:
        logger.warning(
            "%s: %s endpoint %s is UNREACHABLE (connection refused).",
            prefix, name, target,
        )
    except requests.Timeout:
        logger.warning(
            "%s: %s endpoint %s timed out after %.1fs.",
            prefix, name, target, timeout,
        )
    except Exception as exc:
        logger.debug("%s: %s endpoint probe %s failed: %s", prefix, name, target, exc)
