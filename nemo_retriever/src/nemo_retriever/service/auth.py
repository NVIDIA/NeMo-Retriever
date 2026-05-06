# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional bearer-token authentication middleware.

Activated only when ``ServiceConfig.auth.api_token`` is non-empty.  The
middleware compares the incoming header value to the configured token in
constant time to avoid timing-attack leakage.

Bypass paths are matched as **prefixes** so OpenAPI / docs / health all
work with just the four defaults in ``AuthConfig``.
"""

from __future__ import annotations

import hmac
import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from nemo_retriever.service.config import AuthConfig

logger = logging.getLogger(__name__)


_BEARER_PREFIX = "bearer "


def _strip_bearer(value: str) -> str:
    if value.lower().startswith(_BEARER_PREFIX):
        return value[len(_BEARER_PREFIX) :].strip()
    return value.strip()


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Reject requests that don't carry the configured token.

    A no-op when ``config.api_token`` is None or empty so deployments that
    don't enable auth pay no overhead beyond the middleware dispatch.
    """

    def __init__(self, app, *, config: AuthConfig) -> None:
        super().__init__(app)
        self._token = (config.api_token or "").strip()
        self._header = config.header_name
        self._bypass = tuple(config.bypass_paths)

    async def dispatch(self, request: Request, call_next):
        if not self._token:
            return await call_next(request)

        path = request.url.path
        if any(path == p or path.startswith(p.rstrip("/") + "/") for p in self._bypass):
            return await call_next(request)

        provided = request.headers.get(self._header, "")
        provided_token = _strip_bearer(provided)
        if not provided_token or not hmac.compare_digest(provided_token, self._token):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid bearer token."},
                headers={"WWW-Authenticate": "Bearer"},
            )

        return await call_next(request)
