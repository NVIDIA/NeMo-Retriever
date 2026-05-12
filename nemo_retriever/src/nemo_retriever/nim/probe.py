# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import logging
import time
import urllib.parse
from contextlib import contextmanager
from typing import Optional

import requests

from opentelemetry import trace as _trace_api

from nemo_retriever.observability import attributes as _otel_attrs
from nemo_retriever.observability.tracer import get_tracer

logger = logging.getLogger(__name__)


@contextmanager
def _probe_span(name: str, kind: str, url: str):
    """Open a span around one NIM probe.

    Status / detail are filled in *after* the probe finishes (the caller
    sets ``probe_status`` on the span before exit).  When no provider is
    registered the span is a no-op.
    """
    tracer = get_tracer()
    attrs = {
        _otel_attrs.NIM_KIND: kind,
        _otel_attrs.NIM_NAME: name,
        _otel_attrs.NIM_URL: url,
    }
    with tracer.start_as_current_span(
        "nim.probe",
        attributes=attrs,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        yield span


def _finalize_probe_span(span, status: str, error: str | None = None) -> None:
    """Tag *span* with its terminal ``probe_status``, optionally with an error."""
    span.set_attribute(_otel_attrs.NIM_PROBE_STATUS, status)
    if error is not None:
        span.set_status(_trace_api.Status(_trace_api.StatusCode.ERROR, error))


def _kind_from_name(name: str) -> str:
    """Normalize the canonical NIM role passed via ``name`` to a snake_case kind label.

    Callers pass roles like ``"ocr"``, ``"page-elements"``, ``"graphic-elements"``,
    ``"table-structure"``, ``"embed"`` — the prefix arg holds the actor class
    name, not the role, so we cannot derive kind from it.
    """
    if not name:
        return "unknown"
    return name.replace("-", "_").lower()


@dataclasses.dataclass(frozen=True)
class ProbeResult:
    """Outcome of a single NIM endpoint probe."""

    url: str
    name: str
    prefix: str
    status: str  # "ok", "unreachable", "timeout", "error"
    detail: str = ""


_probe_results: list[ProbeResult] = []


def drain_probe_results() -> list[ProbeResult]:
    """Pop and return all collected probe results since last drain.

    Designed to be called once per worker process after operator-chain
    initialisation so the caller can persist failures to the event log.
    """
    results = list(_probe_results)
    _probe_results.clear()
    return results


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
    nim_kind = _kind_from_name(name)

    # Step 1: unauthenticated health check
    with _probe_span(name=name, kind=nim_kind, url=health_url) as span:
        try:
            t0 = time.perf_counter()
            resp = requests.get(health_url, timeout=timeout)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "%s: %s endpoint %s responded %d in %.0fms",
                prefix,
                name,
                health_url,
                resp.status_code,
                elapsed_ms,
            )
            span.set_attribute("http.status_code", int(resp.status_code))
            if resp.ok:
                _finalize_probe_span(span, "ok")
                _probe_results.append(ProbeResult(url=health_url, name=name, prefix=prefix, status="ok"))
                return
        except requests.ConnectionError:
            logger.warning(
                "%s: %s endpoint %s is UNREACHABLE (connection refused). "
                "Processing will stall until this endpoint becomes available.",
                prefix,
                name,
                health_url,
            )
            _finalize_probe_span(span, "unreachable", "connection refused")
            _probe_results.append(
                ProbeResult(
                    url=health_url,
                    name=name,
                    prefix=prefix,
                    status="unreachable",
                    detail=f"{name} endpoint {health_url} is UNREACHABLE (connection refused). "
                    "Processing will stall until this endpoint becomes available.",
                )
            )
            return
        except requests.Timeout:
            logger.warning(
                "%s: %s endpoint %s timed out after %.1fs. " "The endpoint may be overloaded or not ready.",
                prefix,
                name,
                health_url,
                timeout,
            )
            _finalize_probe_span(span, "timeout", f"timeout after {timeout:.1f}s")
            _probe_results.append(
                ProbeResult(
                    url=health_url,
                    name=name,
                    prefix=prefix,
                    status="timeout",
                    detail=f"{name} endpoint {health_url} timed out after {timeout:.1f}s. "
                    "The endpoint may be overloaded or not ready.",
                )
            )
            return
        except Exception as exc:
            _finalize_probe_span(span, "error", type(exc).__name__)
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
    with _probe_span(name=name, kind=nim_kind, url=target) as span:
        try:
            t0 = time.perf_counter()
            resp = requests.post(target, headers=headers, json=body, timeout=timeout)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "%s: %s endpoint %s responded %d in %.0fms",
                prefix,
                name,
                target,
                resp.status_code,
                elapsed_ms,
            )
            span.set_attribute("http.status_code", int(resp.status_code))
            if resp.status_code in (401, 403):
                _finalize_probe_span(span, "auth_failed", f"http {resp.status_code}")
                raise RuntimeError(
                    f"{prefix}: authentication failed for {name} endpoint {target} "
                    f"(HTTP {resp.status_code}) — verify the API key is valid."
                )
            _finalize_probe_span(span, "ok")
        except RuntimeError:
            raise
        except requests.ConnectionError:
            logger.warning(
                "%s: %s endpoint %s is UNREACHABLE (connection refused).",
                prefix,
                name,
                target,
            )
            _finalize_probe_span(span, "unreachable", "connection refused")
            _probe_results.append(
                ProbeResult(
                    url=target,
                    name=name,
                    prefix=prefix,
                    status="unreachable",
                    detail=f"{name} endpoint {target} is UNREACHABLE (connection refused).",
                )
            )
        except requests.Timeout:
            logger.warning(
                "%s: %s endpoint %s timed out after %.1fs.",
                prefix,
                name,
                target,
                timeout,
            )
            _finalize_probe_span(span, "timeout", f"timeout after {timeout:.1f}s")
            _probe_results.append(
                ProbeResult(
                    url=target,
                    name=name,
                    prefix=prefix,
                    status="timeout",
                    detail=f"{name} endpoint {target} timed out after {timeout:.1f}s.",
                )
            )
        except Exception as exc:
            _finalize_probe_span(span, "error", type(exc).__name__)
            logger.debug("%s: %s endpoint probe %s failed: %s", prefix, name, target, exc)
