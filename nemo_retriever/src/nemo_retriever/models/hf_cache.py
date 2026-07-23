from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Iterable, Optional

ENV_HF_CACHE_BASE_DIR = "NEMO_RETRIEVER_HF_CACHE_DIR"

HF_RUNTIME_ENV_KEYS: tuple[str, ...] = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HF_HOME",
    "HF_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    ENV_HF_CACHE_BASE_DIR,
    "HF_ENDPOINT",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN",
    "HF_HUB_ENABLE_HF_TRANSFER",
    "HF_HUB_ETAG_TIMEOUT",
    "HF_HUB_DOWNLOAD_TIMEOUT",
    "HF_HUB_DISABLE_TELEMETRY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "SSL_CERT_FILE",
)


def _has_legacy_default_cache(cache_base: Path) -> bool:
    """Return whether the pre-standard-layout cache contains model snapshots."""
    return any(path.is_dir() for path in cache_base.glob("models--*"))


def resolve_hf_cache_dir(explicit_hf_cache_dir: Optional[str] = None) -> str:
    """Resolve the cache directory used for Hugging Face model snapshots.

    NeMo Retriever's explicit cache settings keep precedence for backwards
    compatibility. Otherwise, follow the standard Hugging Face cache contract:
    ``HF_HUB_CACHE`` names the Hub cache directly, while ``HF_HOME`` is its
    parent directory.
    """
    candidate = explicit_hf_cache_dir or os.getenv(ENV_HF_CACHE_BASE_DIR)
    if candidate:
        return str(Path(candidate).expanduser())

    hub_cache = os.getenv("HF_HUB_CACHE")
    if hub_cache:
        return str(Path(hub_cache).expanduser())

    hf_home = os.getenv("HF_HOME")
    cache_base = Path(hf_home).expanduser() if hf_home else Path.home() / ".cache" / "huggingface"
    return str(cache_base / "hub")


def configure_global_hf_cache_base(explicit_hf_cache_dir: Optional[str] = None) -> str:
    """Configure Hugging Face cache env vars and return the model cache dir."""
    override = explicit_hf_cache_dir or os.getenv(ENV_HF_CACHE_BASE_DIR)
    configured_home = os.getenv("HF_HOME")
    configured_hub = os.getenv("HF_HUB_CACHE")
    if override:
        cache_base = str(Path(override).expanduser())
        hub_cache = str(Path(cache_base) / "hub")
    else:
        cache_base = (
            str(Path(configured_home).expanduser()) if configured_home else str(Path.home() / ".cache" / "huggingface")
        )
        hub_cache = resolve_hf_cache_dir()

    os.environ.setdefault("HF_HOME", cache_base)
    os.environ.setdefault("HF_HUB_CACHE", hub_cache)
    if override:
        os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(cache_base) / "transformers"))

    if not override and not configured_home and not configured_hub and _has_legacy_default_cache(Path(cache_base)):
        warnings.warn(
            f"Using legacy Hugging Face model cache at {cache_base}; "
            f"move its models--* directories to {hub_cache} to adopt the standard layout.",
            UserWarning,
            stacklevel=2,
        )
        return cache_base

    return str(Path(override).expanduser()) if override else hub_cache


def collect_hf_runtime_env(
    *,
    default_hf_hub_offline: str = "0",
    extra_keys: Iterable[str] = (),
) -> dict[str, str]:
    """Collect HF-related environment variables to forward to Ray workers.

    Parameters
    ----------
    default_hf_hub_offline:
        Value to emit for ``HF_HUB_OFFLINE`` when it is not set in the parent
        process environment.  The default keeps online Hub checks enabled.
    extra_keys:
        Additional environment variable names to forward if they are set.
        Duplicates of built-in keys are ignored after their first occurrence.

    Returns
    -------
    dict[str, str]
        Environment variables for Ray ``runtime_env["env_vars"]``.  Explicitly
        blank environment values are preserved.
    """
    env_vars: dict[str, str] = {}
    for key in (*HF_RUNTIME_ENV_KEYS, *tuple(extra_keys)):
        if key in env_vars:
            continue
        value = os.environ.get(key)
        if value is not None:
            env_vars[key] = value

    # HF_HUB_OFFLINE is emitted explicitly so every Ray worker gets a default;
    # passing it through extra_keys is intentionally overridden here.
    env_vars["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", default_hf_hub_offline)
    return env_vars
