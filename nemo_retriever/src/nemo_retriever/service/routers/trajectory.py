# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ATIF adapter layered over the standard managed text-ingestion interface."""

from __future__ import annotations

import json
from io import BytesIO

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import Field

from nemo_retriever.common.schemas.base import RichModel
from nemo_retriever.common.schemas.collections import CollectionName
from nemo_retriever.common.schemas.pipeline_spec import PipelineSpec
from nemo_retriever.common.schemas.requests import IngestRequest, JobCreateRequest
from nemo_retriever.common.schemas.responses import JobCreatedResponse
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


@router.post(
    "/adapters/trajectory/ingest",
    response_model=JobCreatedResponse,
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
) -> JobCreatedResponse | Response:
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
        raise HTTPException(
            status_code=422,
            detail="trajectory contains no indexable text events",
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
    for document in documents:
        upload = UploadFile(
            file=BytesIO(document.text.encode("utf-8")),
            filename=document.filename,
        )
        ingest_metadata = IngestRequest(
            filename=document.filename,
            content_type="text/plain",
            metadata=document.metadata,
            pipeline=pipeline,
        )
        accepted = await ingest.submit_document_to_job(
            request,
            created.job_id,
            upload,
            metadata=ingest_metadata.model_dump_json(exclude_none=True),
            manifest_entry_id=None,
        )
        if isinstance(accepted, Response):
            return accepted

    return created
