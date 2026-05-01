from __future__ import annotations

import os
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
    "NVIDIA_API_KEY",
    "NGC_API_KEY",
)


def resolve_hf_cache_dir(explicit_hf_cache_dir: Optional[str] = None) -> str:
    """Resolve Hugging Face cache dir from explicit arg, env, then default."""
    candidate = explicit_hf_cache_dir or os.getenv(ENV_HF_CACHE_BASE_DIR)
    if candidate:
        return str(Path(candidate).expanduser())
    return str(Path.home() / ".cache" / "huggingface")


def configure_global_hf_cache_base(explicit_hf_cache_dir: Optional[str] = None) -> str:
    """Apply resolved HF cache base to standard Hugging Face env vars."""
    cache_base = resolve_hf_cache_dir(explicit_hf_cache_dir)
    os.environ.setdefault("HF_HOME", cache_base)
    os.environ.setdefault("HF_HUB_CACHE", str(Path(cache_base) / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(cache_base) / "transformers"))
    return cache_base


def collect_hf_runtime_env(
    *,
    default_hf_hub_offline: str = "0",
    extra_keys: Iterable[str] = (),
) -> dict[str, str]:
    """Collect HF-related environment variables to forward to Ray workers."""
    env_vars: dict[str, str] = {}
    for key in (*HF_RUNTIME_ENV_KEYS, *tuple(extra_keys)):
        if key in env_vars:
            continue
        value = os.environ.get(key)
        if value:
            env_vars[key] = value

    env_vars["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", default_hf_hub_offline)
    return env_vars
