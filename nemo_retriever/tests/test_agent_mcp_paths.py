# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from nemo_retriever.agent_mcp.models import AgentMcpError, AgentMcpErrorCode
from nemo_retriever.agent_mcp.paths import (
    PathPolicy,
    expand_local_paths,
    group_paths_by_media_type,
    media_type_for_path,
)


def test_rejects_files_outside_allowed_root(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    outside_root = tmp_path / "outside"
    allowed_root.mkdir()
    outside_root.mkdir()
    outside_file = outside_root / "escape.pdf"
    outside_file.write_text("pdf", encoding="utf-8")

    with pytest.raises(AgentMcpError) as exc:
        expand_local_paths([outside_file], policy=PathPolicy(allowed_roots=[allowed_root]))

    assert exc.value.code is AgentMcpErrorCode.PATH_OUTSIDE_ALLOWED_ROOT


def test_missing_outside_allowed_root_is_rejected_before_not_found(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    missing_outside_file = tmp_path / "outside" / "missing.pdf"

    with pytest.raises(AgentMcpError) as exc:
        expand_local_paths([missing_outside_file], policy=PathPolicy(allowed_roots=[allowed_root]))

    assert exc.value.code is AgentMcpErrorCode.PATH_OUTSIDE_ALLOWED_ROOT


def test_resolves_symlinks_before_allowed_root_checks(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    outside_root = tmp_path / "outside"
    allowed_root.mkdir()
    outside_root.mkdir()
    outside_file = outside_root / "escape.pdf"
    outside_file.write_text("pdf", encoding="utf-8")
    symlink = allowed_root / "linked.pdf"
    symlink.symlink_to(outside_file)

    with pytest.raises(AgentMcpError) as exc:
        expand_local_paths([symlink], policy=PathPolicy(allowed_roots=[allowed_root]))

    assert exc.value.code is AgentMcpErrorCode.PATH_OUTSIDE_ALLOWED_ROOT


def test_expands_directory_collecting_supported_files_and_skipping_unsupported(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    nested = root / "nested"
    nested.mkdir(parents=True)
    pdf = root / "manual.pdf"
    png = nested / "diagram.png"
    unsupported = root / "archive.zip"
    pdf.write_text("pdf", encoding="utf-8")
    png.write_bytes(b"png")
    unsupported.write_bytes(b"zip")

    expanded = expand_local_paths([root], policy=PathPolicy(allowed_roots=[root]))

    assert expanded.files == sorted([pdf.resolve(), png.resolve()])
    assert expanded.total_bytes == pdf.stat().st_size + png.stat().st_size
    assert expanded.skipped == [
        {
            "path": str(unsupported.resolve()),
            "code": AgentMcpErrorCode.UNSUPPORTED_MEDIA_TYPE.value,
        }
    ]


def test_max_files_is_enforced_during_directory_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    first = root / "first.pdf"
    second = root / "second.pdf"
    third = root / "third.pdf"
    for path in (first, second, third):
        path.write_text("pdf", encoding="utf-8")
    yielded: list[Path] = []
    original_rglob = Path.rglob

    def tracking_rglob(path: Path, pattern: str):
        if path == root.resolve():
            for candidate in (first.resolve(), second.resolve(), third.resolve()):
                yielded.append(candidate)
                yield candidate
            return
        yield from original_rglob(path, pattern)

    monkeypatch.setattr(Path, "rglob", tracking_rglob)

    with pytest.raises(AgentMcpError) as exc:
        expand_local_paths([root], policy=PathPolicy(allowed_roots=[root], max_files=1))

    assert exc.value.code is AgentMcpErrorCode.BACKEND_ERROR
    assert yielded == [first.resolve(), second.resolve()]


def test_rejects_remote_uri_strings_before_local_path_checks(tmp_path: Path) -> None:
    with pytest.raises(AgentMcpError) as exc:
        expand_local_paths(["https://example.com/doc.pdf"], policy=PathPolicy(allowed_roots=[tmp_path]))

    assert exc.value.code is AgentMcpErrorCode.PATH_NOT_FOUND


def test_groups_paths_by_media_type(tmp_path: Path) -> None:
    paths = [
        tmp_path / "doc.pdf",
        tmp_path / "page.html",
        tmp_path / "note.txt",
        tmp_path / "audio.mp3",
        tmp_path / "video.mp4",
        tmp_path / "image.png",
    ]

    assert [media_type_for_path(path) for path in paths] == [
        "document",
        "html",
        "text",
        "audio",
        "video",
        "image",
    ]
    assert group_paths_by_media_type(paths) == {
        "document": [paths[0]],
        "html": [paths[1]],
        "text": [paths[2]],
        "audio": [paths[3]],
        "video": [paths[4]],
        "image": [paths[5]],
    }
