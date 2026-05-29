# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test helpers for the retriever-service test suite.

Provides :func:`create_test_job` so each per-router test can open a
short-lived :class:`JobAggregate` (J1+) before POSTing uploads. Tests
are free to ignore this helper and call ``/v1/ingest/job`` directly.
"""

from __future__ import annotations

import os

from fastapi.testclient import TestClient

# Suppress ANSI styling from rich/typer error panels so CLI tests can
# substring-match flag names like ``--no-vdb`` in ``result.output``.
# Rich enables color in CI (``CI=true``) which splits option names into
# separately-styled tokens (``\x1b[1;36m-\x1b[0m\x1b[1;36m-no\x1b[0m\x1b[1;36m-vdb\x1b[0m``).
os.environ.setdefault("NO_COLOR", "1")
os.environ.setdefault("TERM", "dumb")


def create_test_job(
    client: TestClient,
    *,
    expected_documents: int = 1,
    label: str | None = None,
) -> str:
    """Open a job aggregate via ``POST /v1/ingest/job`` and return its id.

    Test callers chain into the new nested upload routes:

    >>> job_id = create_test_job(client)
    >>> client.post(f"/v1/ingest/job/{job_id}/document", files=..., data=...)
    """
    payload: dict = {"expected_documents": expected_documents}
    if label is not None:
        payload["label"] = label
    resp = client.post("/v1/ingest/job", json=payload)
    assert resp.status_code == 201, f"job creation failed: {resp.status_code} {resp.text}"
    return resp.json()["job_id"]
