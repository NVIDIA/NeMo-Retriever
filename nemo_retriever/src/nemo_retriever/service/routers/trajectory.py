# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ATIF adapter layered over the standard managed text-ingestion interface."""

from __future__ import annotations

import asyncio
import json
from io import BytesIO

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import Field

from nemo_retriever.common.schemas.base import RichModel
from nemo_retriever.common.schemas.collections import CollectionName
from nemo_retriever.common.schemas.pipeline_spec import PipelineSpec
from nemo_retriever.common.schemas.requests import IngestRequest, JobCreateRequest
from nemo_retriever.common.schemas.responses import IngestAccepted, JobCreatedResponse
from nemo_retriever.service.routers import ingest
from nemo_retriever.service.trajectory_adapter import (
    decode_trajectory,
    project_agent_trajectory,
)

router = APIRouter(tags=["trajectory adapter"])


class TrajectoryIngestRequest(RichModel):
    """Options for adapting one ATIF trajectory to standard text ingestion."""

    collection_name: CollectionName
    label: str = "agent-trajectory"
    exclude_tool_names: list[str] = Field(default_factory=list)
    tool_output_char_limit: int | None = Field(default=None, ge=0)
    queue_admission_timeout_seconds: float = Field(default=30.0, gt=0, le=240)


class TrajectoryIngestResponse(RichModel):
    """One adapter submission, including an explicit empty-trajectory result."""

    job_id: str | None
    expected_documents: int
    status: str
    no_op: bool = False
    created_at: str | None = None
    label: str | None = None
    trace_id: str | None = None
    collection_name: str | None = None


class _QueueAdmissionTimeout(Exception):
    def __init__(self, headers: dict[str, str] | None) -> None:
        self.headers = headers


@router.post(
    "/adapters/trajectory/ingest",
    response_model=TrajectoryIngestResponse,
    status_code=202,
    summary="Project an ATIF trajectory and ingest it as text documents",
)
async def ingest_trajectory(
    request: Request,
    file: UploadFile = File(..., description="The ATIF trajectory to ingest"),
    metadata: str = Form(
        default="{}",
        description="JSON-encoded TrajectoryIngestRequest options",
    ),
) -> JobCreatedResponse | TrajectoryIngestResponse | Response:
    """Preprocess ATIF at the service boundary, then use standard text ingestion."""
    try:
        options = TrajectoryIngestRequest(**json.loads(metadata))
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {exc}")

    ingest._check_upload_size(file, request)
    content = await file.read()
    try:
        documents = project_agent_trajectory(
            decode_trajectory(content),
            exclude_tool_names=options.exclude_tool_names,
            tool_output_char_limit=options.tool_output_char_limit,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not documents:
        return TrajectoryIngestResponse(
            job_id=None,
            expected_documents=0,
            status="completed",
            no_op=True,
            collection_name=options.collection_name,
        )

    created = await ingest.create_job(
        request,
        Response(),
        JobCreateRequest(
            expected_documents=len(documents),
            label=options.label,
            collection_name=options.collection_name,
        ),
    )
    if isinstance(created, Response):
        return created

    pipeline = PipelineSpec(extraction_mode="text")
    admission_deadline = (
        asyncio.get_running_loop().time()
        + options.queue_admission_timeout_seconds
    )
    for accepted_events, document in enumerate(documents):
        ingest_metadata = IngestRequest(
            filename=document.filename,
            content_type="text/plain",
            metadata=document.metadata,
            pipeline=pipeline,
        )
        try:
            accepted = await _submit_event_with_retry(
                request=request,
                job_id=created.job_id,
                filename=document.filename,
                payload=document.text.encode("utf-8"),
                metadata=ingest_metadata.model_dump_json(exclude_none=True),
                deadline=admission_deadline,
            )
        except _QueueAdmissionTimeout as exc:
            raise HTTPException(
                status_code=429,
                detail={
                    "code": "trajectory_queue_admission_timeout",
                    "message": (
                        "Queue admission timed out; trajectory ingestion may be "
                        "partial."
                    ),
                    "job_id": created.job_id,
                    "accepted_events": accepted_events,
                    "total_events": len(documents),
                    "ingestion_may_be_partial": True,
                },
                headers=exc.headers,
            ) from exc
        if isinstance(accepted, Response):
            return accepted

    return created


async def _submit_event_with_retry(
    *,
    request: Request,
    job_id: str,
    filename: str,
    payload: bytes,
    metadata: str,
    deadline: float,
) -> IngestAccepted | Response:
    loop = asyncio.get_running_loop()
    while True:
        upload = UploadFile(file=BytesIO(payload), filename=filename)
        try:
            return await ingest.submit_document_to_job(
                request,
                job_id,
                upload,
                metadata=metadata,
                manifest_entry_id=None,
            )
        except HTTPException as exc:
            if exc.status_code != 429:
                raise
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise _QueueAdmissionTimeout(exc.headers) from exc
            delay = _retry_delay_seconds(exc.headers)
            await asyncio.sleep(min(delay, remaining))
            if loop.time() >= deadline:
                raise _QueueAdmissionTimeout(exc.headers) from exc


def _retry_delay_seconds(headers: dict[str, str] | None) -> float:
    value = (headers or {}).get("Retry-After")
    if value is not None:
        try:
            return max(0.0, float(value))
        except ValueError:
            pass
    return 0.25
