# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable operation markers for the bounded LanceDB sink."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal


class VdbOperationConflict(RuntimeError):
    """An operation identity was reused for a different request or input."""


class CommitOutcomeUnknown(RuntimeError):
    """A previous write may have committed without its durable data marker."""


def _token(value: str, length: int = 24) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


@dataclass(slots=True)
class SinkOperationMarkers:
    """Classify and advance one operation through Lance table tags.

    Tags are separate from the Lance data mutation.  They make acknowledged
    writes and finalization resumable and make the remaining add-to-tag gap
    fail closed for append instead of silently duplicating rows.
    """

    operation_id: str
    request_fingerprint: str
    mode: Literal["append", "overwrite"]
    state: Literal["write", "data", "success"]
    base_version: int | None
    recorded_version: int | None = None
    recorded_rows: int | None = None
    recorded_digest: str | None = None
    pending_tag: str | None = None
    data_tag: str | None = None
    _op_token: str = field(init=False, repr=False)
    _request_token: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._op_token = _token(self.operation_id)
        self._request_token = _token(self.request_fingerprint)

    @property
    def _operation_fragment(self) -> str:
        return f"_{self._op_token}_"

    def _state_prefix(self, state: str) -> str:
        return f"nemo_sink_{state}_{self._op_token}_{self._request_token}"

    @classmethod
    def prepare(
        cls,
        table: Any | None,
        *,
        operation_id: str,
        request_fingerprint: str,
        mode: Literal["append", "overwrite"],
    ) -> SinkOperationMarkers:
        markers = cls(
            operation_id=operation_id,
            request_fingerprint=request_fingerprint,
            mode=mode,
            state="write",
            base_version=None,
        )
        if table is None:
            return markers

        tags = table.tags.list()
        incomplete_tags = [name for name in tags if name.startswith(("nemo_sink_pending_", "nemo_sink_data_"))]
        foreign_incomplete = sorted(name for name in incomplete_tags if markers._operation_fragment not in name)
        if foreign_incomplete:
            raise VdbOperationConflict(
                "The LanceDB table has an unfinished bounded-sink operation; "
                "retry or reconcile that operation before starting another write."
            )
        operation_tags = [name for name in tags if markers._operation_fragment in name]
        current_request_tags = [name for name in operation_tags if markers._request_token in name]
        conflicting = sorted(set(operation_tags) - set(current_request_tags))
        if conflicting:
            raise VdbOperationConflict(
                f"VDB sink operation_id {operation_id!r} was already used for a different write request."
            )

        success = [name for name in current_request_tags if name.startswith(markers._state_prefix("success") + "_")]
        data = [name for name in current_request_tags if name.startswith(markers._state_prefix("data") + "_")]
        pending_name = markers._state_prefix("pending")

        if len(success) > 1 or len(data) > 1:
            raise VdbOperationConflict(f"VDB sink operation_id {operation_id!r} has conflicting durable markers.")
        if success:
            rows, digest = markers._parse_content_marker(success[0], state="success")
            return cls(
                operation_id=operation_id,
                request_fingerprint=request_fingerprint,
                mode=mode,
                state="success",
                base_version=None,
                recorded_version=int(tags[success[0]]["version"]),
                recorded_rows=rows,
                recorded_digest=digest,
                pending_tag=pending_name if pending_name in tags else None,
                data_tag=data[0] if data else None,
            )
        if data:
            rows, digest = markers._parse_content_marker(data[0], state="data")
            base_version = int(tags[pending_name]["version"]) if pending_name in tags else None
            return cls(
                operation_id=operation_id,
                request_fingerprint=request_fingerprint,
                mode=mode,
                state="data",
                base_version=base_version,
                recorded_version=int(tags[data[0]]["version"]),
                recorded_rows=rows,
                recorded_digest=digest,
                pending_tag=pending_name if pending_name in tags else None,
                data_tag=data[0],
            )

        table.checkout_latest()
        latest_version = int(table.version)
        if pending_name not in tags:
            table.tags.create(pending_name, latest_version)
            base_version = latest_version
        else:
            base_version = int(tags[pending_name]["version"])

        if latest_version != base_version:
            if mode == "append":
                raise CommitOutcomeUnknown(
                    f"VDB sink operation_id {operation_id!r} prepared at version {base_version}, "
                    f"but the latest version is {latest_version}; refusing to replay append because "
                    "the prior commit outcome is indeterminate."
                )
            # Replaying overwrite is content-idempotent. Move the base marker
            # forward so a new definite-failure check compares to this version.
            table.tags.update(pending_name, latest_version)
            base_version = latest_version

        markers.base_version = base_version
        markers.pending_tag = pending_name
        return markers

    def _parse_content_marker(self, name: str, *, state: str) -> tuple[int, str]:
        prefix = self._state_prefix(state) + "_"
        payload = name.removeprefix(prefix)
        rows_text, separator, digest = payload.partition("_")
        if not separator or not rows_text.isdigit() or len(digest) != 64:
            raise VdbOperationConflict(f"Malformed durable VDB sink marker {name!r}.")
        return int(rows_text), digest

    def verify_input(self, *, rows: int, digest: str) -> None:
        if self.recorded_rows != rows or self.recorded_digest != digest:
            raise VdbOperationConflict(
                f"VDB sink operation_id {self.operation_id!r} was retried with different canonical content."
            )

    def mark_data(self, table: Any, *, version: int, rows: int, digest: str) -> None:
        name = f"{self._state_prefix('data')}_{rows}_{digest}"
        table.tags.create(name, int(version))
        self.state = "data"
        self.recorded_version = int(version)
        self.recorded_rows = int(rows)
        self.recorded_digest = digest
        self.data_tag = name

    def abort_if_unchanged(self, table: Any | None) -> bool:
        """Remove a pending marker after a proven pre-commit failure."""

        if table is None or self.pending_tag is None or self.base_version is None:
            return False
        table.checkout_latest()
        if int(table.version) != int(self.base_version):
            return False
        if self.pending_tag in table.tags.list():
            table.tags.delete(self.pending_tag)
        self.pending_tag = None
        return True

    def mark_success(self, table: Any, *, version: int, rows: int, digest: str) -> None:
        name = f"{self._state_prefix('success')}_{rows}_{digest}"
        try:
            table.tags.create(name, int(version))
        except RuntimeError:
            if name not in table.tags.list() or table.tags.get_version(name) != int(version):
                raise
        self.state = "success"
        self.recorded_version = int(version)
        self.recorded_rows = int(rows)
        self.recorded_digest = digest
        self.cleanup_after_success(table)

    def cleanup_after_success(self, table: Any) -> None:
        """Finish idempotent marker cleanup after success became durable."""

        if self.data_tag is not None and self.data_tag in table.tags.list():
            table.tags.delete(self.data_tag)
        self.data_tag = None
        if self.pending_tag is not None and self.pending_tag in table.tags.list():
            table.tags.delete(self.pending_tag)
        self.pending_tag = None
