# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import urlparse

from nemo_retriever.agent_mcp.models import AgentMcpError, AgentMcpErrorCode

EXTENSION_MEDIA_TYPES: dict[str, str] = {
    ".pdf": "document",
    ".docx": "document",
    ".pptx": "document",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".tif": "image",
    ".tiff": "image",
    ".txt": "text",
    ".md": "text",
    ".html": "html",
    ".htm": "html",
    ".mp3": "audio",
    ".wav": "audio",
    ".flac": "audio",
    ".m4a": "audio",
    ".mp4": "video",
    ".mov": "video",
    ".mkv": "video",
    ".webm": "video",
}

_CLOUD_URI_PREFIXES = (
    "s3://",
    "gs://",
    "az://",
    "azure://",
    "abfs://",
    "abfss://",
    "adl://",
    "dbfs:/",
)


@dataclass(frozen=True)
class PathPolicy:
    allowed_roots: list[Path]
    max_files: int = 1000
    max_total_bytes: int | None = None

    def resolved_roots(self) -> list[Path]:
        return [root.expanduser().resolve() for root in self.allowed_roots]


@dataclass
class ExpandedPaths:
    files: list[Path]
    skipped: list[dict[str, str]]
    total_bytes: int


def _is_supported(path: Path) -> bool:
    return path.suffix.lower() in EXTENSION_MEDIA_TYPES


def _looks_like_uri(value: str) -> bool:
    lower_value = value.lower()
    if lower_value.startswith(_CLOUD_URI_PREFIXES):
        return True

    parsed = urlparse(value)
    if not parsed.scheme:
        return False
    if len(parsed.scheme) == 1 and value[1:3] in {":/", ":\\"}:
        return False
    return True


def _assert_allowed(path: Path, roots: list[Path]) -> Path:
    resolved_path = path.expanduser().resolve()
    for root in roots:
        if resolved_path == root or root in resolved_path.parents:
            return resolved_path

    raise AgentMcpError(
        AgentMcpErrorCode.PATH_OUTSIDE_ALLOWED_ROOT,
        f"Path '{resolved_path}' is outside the allowed roots.",
        details={"path": str(resolved_path), "allowed_roots": [str(root) for root in roots]},
    )


def _iter_candidates(path: Path) -> Iterator[Path]:
    if path.is_dir():
        yield from path.rglob("*")
        return

    yield path


def expand_local_paths(inputs: Iterable[str | Path], *, policy: PathPolicy) -> ExpandedPaths:
    roots = policy.resolved_roots()
    files: set[Path] = set()
    skipped_by_path: dict[Path, dict[str, str]] = {}
    total_bytes = 0

    for input_path in inputs:
        if isinstance(input_path, str) and _looks_like_uri(input_path):
            raise AgentMcpError(
                AgentMcpErrorCode.PATH_NOT_FOUND,
                f"Remote URI '{input_path}' is not a local path.",
                details={"path": input_path},
            )

        resolved_path = _assert_allowed(Path(input_path), roots)
        if not resolved_path.exists():
            raise AgentMcpError(
                AgentMcpErrorCode.PATH_NOT_FOUND,
                f"Path '{resolved_path}' was not found.",
                details={"path": str(resolved_path)},
            )

        for candidate in _iter_candidates(resolved_path):
            if not candidate.exists():
                raise AgentMcpError(
                    AgentMcpErrorCode.PATH_NOT_FOUND,
                    f"Path '{candidate}' was not found.",
                    details={"path": str(candidate)},
                )
            if not candidate.is_file():
                continue

            resolved_file = _assert_allowed(candidate, roots)
            if not _is_supported(resolved_file):
                skipped_by_path[resolved_file] = {
                    "path": str(resolved_file),
                    "code": AgentMcpErrorCode.UNSUPPORTED_MEDIA_TYPE.value,
                }
                continue

            if resolved_file in files:
                continue

            file_size = resolved_file.stat().st_size
            if len(files) + 1 > policy.max_files:
                raise AgentMcpError(
                    AgentMcpErrorCode.BACKEND_ERROR,
                    f"Expanded local paths exceed max_files={policy.max_files}.",
                    details={"max_files": policy.max_files},
                )
            if policy.max_total_bytes is not None and total_bytes + file_size > policy.max_total_bytes:
                raise AgentMcpError(
                    AgentMcpErrorCode.BACKEND_ERROR,
                    f"Expanded local paths exceed max_total_bytes={policy.max_total_bytes}.",
                    details={"max_total_bytes": policy.max_total_bytes},
                )

            files.add(resolved_file)
            total_bytes += file_size

    return ExpandedPaths(
        files=sorted(files),
        skipped=[skipped_by_path[path] for path in sorted(skipped_by_path)],
        total_bytes=total_bytes,
    )


def media_type_for_path(path: str | Path) -> str:
    extension = Path(path).suffix.lower()
    try:
        return EXTENSION_MEDIA_TYPES[extension]
    except KeyError as exc:
        raise AgentMcpError(
            AgentMcpErrorCode.UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported media type for path '{path}'.",
            details={"path": str(path), "extension": extension},
        ) from exc


def group_paths_by_media_type(paths: Iterable[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        grouped.setdefault(media_type_for_path(path), []).append(path)
    return grouped
