# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_retriever.common.params import ExtractParams
from nemo_retriever.ingest.service import (
    ServiceIngestPlanRequest,
    ServiceIngestSourceOptions,
    _sanitize_service_stage_params,
    resolve_service_ingest_request,
)


def test_omitted_extract_images_stays_auto_through_the_service_wire(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    resolved = resolve_service_ingest_request(
        ServiceIngestPlanRequest(source=ServiceIngestSourceOptions(documents=[str(pdf)]))
    )
    wire = _sanitize_service_stage_params(resolved.extract_params, stage="extract")
    payload = wire.model_dump(mode="json", exclude_none=True, exclude_unset=True)

    assert resolved.extract_params.extract_images is None
    assert "extract_images" not in payload
    assert ExtractParams(**payload).extract_images is None
