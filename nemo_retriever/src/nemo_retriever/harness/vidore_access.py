# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Fast authenticated access checks for remote ViDoRe evaluation data."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
import os
from typing import Any
from urllib.parse import urlparse

from nemo_retriever.harness.benchmark_registry import get_benchmark, get_runset

_EVALUATION_PARTITIONS = ("queries", "qrels", "corpus")
_SUCCESS_STATUSES = {200, 206}


class VidoreAccessError(RuntimeError):
    """Raised when a ViDoRe repository or one of its data objects is inaccessible."""


@dataclass(frozen=True)
class VidoreAccessResult:
    dataset: str
    revision: str
    checked_partitions: tuple[str, ...]
    authenticated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_vidore_dataset_names() -> tuple[str, ...]:
    """Return the registry-owned ViDoRe v3 dataset names used by the nightly."""

    runset = get_runset("vidore_v3_all")
    return tuple(get_benchmark(benchmark).dataset for benchmark in runset.runs)


def _exception_status(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def _safe_api_error(*, operation: str, dataset: str, exc: BaseException) -> VidoreAccessError:
    status = _exception_status(exc)
    status_suffix = f" (HTTP {status})" if status is not None else ""
    return VidoreAccessError(f"{operation} failed for {dataset}{status_suffix}: {type(exc).__name__}")


def _partition_files(info: Any, *, dataset: str) -> dict[str, str]:
    files = [str(sibling.rfilename) for sibling in getattr(info, "siblings", ())]
    selected: dict[str, str] = {}
    for partition in _EVALUATION_PARTITIONS:
        matches = sorted(path for path in files if path.startswith(f"{partition}/") and path.endswith(".parquet"))
        if matches:
            selected[partition] = matches[0]
    missing = [partition for partition in _EVALUATION_PARTITIONS if partition not in selected]
    if missing:
        raise VidoreAccessError(f"{dataset} is missing evaluation partitions: {', '.join(missing)}")
    return selected


def check_vidore_access(
    *,
    dataset_names: Sequence[str] | None = None,
    token: str | None = None,
    timeout_seconds: float = 20.0,
    require_token: bool = True,
    api: Any | None = None,
    get: Callable[..., Any] | None = None,
) -> list[VidoreAccessResult]:
    """Probe one byte from each ViDoRe queries/qrels/corpus partition.

    The range requests follow Hugging Face redirects but stream the response,
    so a successful check does not download the underlying parquet objects.
    Error messages intentionally exclude tokens and signed URLs.
    """

    effective_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if require_token and not effective_token:
        raise VidoreAccessError(
            "HF_TOKEN is not set; add it to the mode-600 nightly configuration before checking ViDoRe access"
        )

    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=effective_token)
    if get is None:
        import requests

        get = requests.get

    authenticated = bool(effective_token)
    if authenticated:
        try:
            api.whoami()
        except Exception as exc:
            raise _safe_api_error(operation="HF_TOKEN validation", dataset="Hugging Face", exc=exc) from exc

    from huggingface_hub import hf_hub_url

    results: list[VidoreAccessResult] = []
    for dataset in tuple(dataset_names or default_vidore_dataset_names()):
        repo_id = f"vidore/{dataset}"
        try:
            info = api.dataset_info(repo_id, files_metadata=True)
        except Exception as exc:
            raise _safe_api_error(operation="dataset metadata request", dataset=dataset, exc=exc) from exc

        revision = str(getattr(info, "sha", None) or "main")
        partition_files = _partition_files(info, dataset=dataset)
        for partition, filename in partition_files.items():
            url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision)
            headers = {"Range": "bytes=0-0"}
            if effective_token:
                headers["Authorization"] = f"Bearer {effective_token}"
            try:
                with get(
                    url,
                    headers=headers,
                    allow_redirects=True,
                    stream=True,
                    timeout=timeout_seconds,
                ) as response:
                    status_chain = [item.status_code for item in response.history] + [response.status_code]
                    if response.status_code not in _SUCCESS_STATUSES:
                        chain = " -> ".join(str(status) for status in status_chain)
                        final_host = urlparse(response.url).hostname or "unknown host"
                        raise VidoreAccessError(
                            f"{dataset} {filename} failed with status chain {chain} at {final_host}"
                        )
                    if not next(response.iter_content(chunk_size=1), b""):
                        raise VidoreAccessError(f"{dataset} {filename} returned an empty response")
            except VidoreAccessError:
                raise
            except Exception as exc:
                raise VidoreAccessError(f"{dataset} {partition} range request failed: {type(exc).__name__}") from exc

        results.append(
            VidoreAccessResult(
                dataset=dataset,
                revision=revision,
                checked_partitions=_EVALUATION_PARTITIONS,
                authenticated=authenticated,
            )
        )
    return results
