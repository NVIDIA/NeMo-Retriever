# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence, cast

from nemo_retriever.ingestor import create_ingestor
from nemo_retriever.params import VdbUploadParams
from nemo_retriever.retriever import Retriever
from nemo_retriever.utils.input_files import expand_input_file_patterns, resolve_input_files


IngestRunModeValue = Literal["inprocess", "batch"]
_SUPPORTED_RUN_MODES: tuple[IngestRunModeValue, ...] = ("inprocess", "batch")


def _validate_run_mode(run_mode: str) -> IngestRunModeValue:
    if run_mode not in _SUPPORTED_RUN_MODES:
        raise ValueError(f"run_mode must be one of {', '.join(_SUPPORTED_RUN_MODES)}, got {run_mode!r}.")
    return cast(IngestRunModeValue, run_mode)


# The ingest command accepts bare dataset directories; expand those to PDFs
# before passing file/glob inputs through the shared input normalizer.
def _expand_pdf_ingest_documents(documents: Sequence[str]) -> list[str]:
    inputs: list[str] = []
    for document in documents:
        raw_document = str(document)
        path = Path(raw_document).expanduser()
        if path.is_dir():
            directory_files = resolve_input_files(path, "pdf")
            if not directory_files:
                raise FileNotFoundError(f"No PDF files found under directory: {path}")
            inputs.extend(str(file) for file in directory_files)
        else:
            inputs.append(raw_document)

    document_list = expand_input_file_patterns(inputs)
    non_pdf_documents = [document for document in document_list if Path(document).suffix.lower() != ".pdf"]
    if non_pdf_documents:
        examples = ", ".join(non_pdf_documents[:3])
        raise ValueError(
            "Only PDF inputs are supported by retriever ingest. "
            f"Use 'retriever pipeline run' for other input types. Non-PDF input(s): {examples}"
        )
    return document_list


def ingest_documents(
    documents: Sequence[str],
    *,
    run_mode: IngestRunModeValue = "inprocess",
    lancedb_uri: str = "lancedb",
    table_name: str = "nv-ingest",
) -> dict[str, Any]:
    """Run the minimal SDK ingestion chain used by the root CLI."""
    validated_run_mode = _validate_run_mode(run_mode)
    document_list = _expand_pdf_ingest_documents(documents)
    params = VdbUploadParams(vdb_kwargs={"uri": lancedb_uri, "table_name": table_name})

    result = (
        create_ingestor(run_mode=validated_run_mode).files(document_list).extract().embed().vdb_upload(params).ingest()
    )
    return {
        "documents": document_list,
        "lancedb_uri": lancedb_uri,
        "result": result,
        "table_name": table_name,
    }


def query_documents(
    query: str,
    *,
    top_k: int = 10,
    lancedb_uri: str = "lancedb",
    table_name: str = "nv-ingest",
) -> list[dict[str, Any]]:
    """Run the minimal SDK query path used by the root CLI."""
    retriever = Retriever(top_k=top_k, vdb_kwargs={"uri": lancedb_uri, "table_name": table_name})
    return retriever.query(query)
