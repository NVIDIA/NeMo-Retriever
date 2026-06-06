from __future__ import annotations

import os
from typing import Iterable, Optional

REMOTE_AUTH_RUNTIME_ENV_KEYS: tuple[str, ...] = ("NVIDIA_API_KEY",)


def resolve_remote_api_key(explicit_api_key: Optional[str] = None) -> Optional[str]:
    """Resolve bearer token for hosted NIM endpoints."""
    token = explicit_api_key or os.getenv("NVIDIA_API_KEY") or os.getenv("NGC_API_KEY")
    token = (token or "").strip()
    return token or None


def collect_remote_auth_runtime_env(
    *,
    extra_keys: Iterable[str] = (),
) -> dict[str, str]:
    """Collect non-HF remote auth env vars historically forwarded to Ray workers."""
    env_vars: dict[str, str] = {}
    for key in (*REMOTE_AUTH_RUNTIME_ENV_KEYS, *tuple(extra_keys)):
        if key in env_vars:
            continue
        value = os.environ.get(key)
        if value is not None:
            env_vars[key] = value
    return env_vars
