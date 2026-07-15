# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bearer authentication and token-to-workspace scope authorization."""

from __future__ import annotations

import hmac
import json
import logging
from pathlib import Path
from typing import Any

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


def internal_auth_headers(token: str | None) -> dict[str, str]:
    """Build headers for gateway/worker calls to the VectorDB service."""
    token = (token or "").strip()
    if not token:
        return {}
    return {"X-NRL-Internal-Token": token}


def auth_headers(config: AuthConfig) -> dict[str, str]:
    """Build the configured service credential for trusted pod callbacks."""
    token = (config.api_token or "").strip()
    if not token:
        return {}
    value = f"Bearer {token}" if config.header_name.lower() == "authorization" else token
    return {config.header_name: value}


def authorized_scope(request: Request) -> str:
    """Return the middleware-authorized scope; never trust a raw header here."""
    return str(getattr(request.state, "authorized_scope", "default"))


class ScopeAuthorizer:
    """Resolve credentials to allowed logical scopes without logging secrets."""

    def __init__(self, config: AuthConfig) -> None:
        self.default_scope = config.default_scope.strip() or "default"
        self.allow_unscoped_dev = config.allow_unscoped_dev
        self._records: list[tuple[str, frozenset[str]]] = []
        if token := (config.api_token or "").strip():
            self._records.append((token, frozenset({self.default_scope})))
        if config.scope_token_file:
            self._load_file(config.scope_token_file)

    @property
    def credentials_required(self) -> bool:
        return bool(self._records)

    def _load_file(self, path: str) -> None:
        try:
            payload: Any = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Unable to load scope-token secret file {path!r}: {exc}") from exc
        records = payload.get("tokens") if isinstance(payload, dict) else None
        if not isinstance(records, list):
            raise ValueError("scope-token secret file must contain a 'tokens' list")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("each scope-token record must be an object")
            token = str(record.get("token") or "").strip()
            scopes = record.get("scopes")
            if not token or not isinstance(scopes, list) or not scopes:
                raise ValueError("scope-token records require a non-empty token and scopes list")
            self._records.append((token, frozenset(str(scope).strip() for scope in scopes if str(scope).strip())))

    def authorize(self, provided_token: str, requested_scope: str | None) -> tuple[str | None, int | None]:
        requested = (requested_scope or self.default_scope).strip() or self.default_scope
        if not self._records:
            if self.allow_unscoped_dev:
                return requested, None
            return None, 401
        allowed: frozenset[str] | None = None
        for token, scopes in self._records:
            if hmac.compare_digest(provided_token, token):
                allowed = scopes
        if allowed is None:
            return None, 401
        if requested not in allowed:
            return None, 404
        return requested, None


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Reject requests that don't carry the configured token.

    A no-op when ``config.api_token`` is None or empty so deployments that
    don't enable auth pay no overhead beyond the middleware dispatch.
    """

    def __init__(self, app, *, config: AuthConfig) -> None:
        super().__init__(app)
        self._header = config.header_name
        self._bypass = tuple(config.bypass_paths)
        self._authorizer = ScopeAuthorizer(config)

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if any(path == p or path.startswith(p.rstrip("/") + "/") for p in self._bypass):
            request.state.authorized_scope = self._authorizer.default_scope
            return await call_next(request)

        provided = request.headers.get(self._header, "")
        provided_token = _strip_bearer(provided)
        scope, failure = self._authorizer.authorize(
            provided_token, request.headers.get("X-NRL-Scope")
        )
        if failure == 401:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid bearer token."},
                headers={"WWW-Authenticate": "Bearer"},
            )
        if failure == 404:
            return JSONResponse(status_code=404, content={"detail": "Resource not found."})

        request.state.authorized_scope = scope
        return await call_next(request)
