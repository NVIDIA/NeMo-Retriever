# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated

import typer


QueryArgument = Annotated[str, typer.Argument(..., help="Query text.")]
TopKOption = Annotated[int, typer.Option("--top-k", min=1, help="Final number of hits to return.")]
CandidateKOption = Annotated[
    int | None,
    typer.Option(
        "--candidate-k",
        min=1,
        help=(
            "Candidate pool size before page deduplication or content-type filtering; "
            "must be greater than or equal to --top-k."
        ),
    ),
]
PageDedupOption = Annotated[
    bool,
    typer.Option(
        "--page-dedup/--no-page-dedup",
        help="Collapse hits to unique document pages.",
    ),
]
ContentTypesOption = Annotated[
    str | None,
    typer.Option(
        "--content-types",
        help="Comma-separated content types to keep, such as text,table; untyped hits are excluded.",
    ),
]
LanceDbUriOption = Annotated[
    str,
    typer.Option(
        "--lancedb-uri",
        help="LanceDB database URI to read; match the value used for retriever ingest local --lancedb-uri.",
    ),
]
TableNameOption = Annotated[
    str,
    typer.Option(
        "--table-name",
        help="LanceDB table name to read; match the value used for retriever ingest local --table-name.",
    ),
]
EmbedInvokeUrlOption = Annotated[
    str | None,
    typer.Option("--embed-invoke-url", help="Embedding NIM endpoint URL."),
]
EmbedModelNameOption = Annotated[
    str | None,
    typer.Option("--embed-model-name", help="Optional embedding model name override."),
]
RerankerInvokeUrlOption = Annotated[
    str | None,
    typer.Option("--reranker-invoke-url", help="Reranker endpoint URL."),
]
RerankerApiKeyEnvOption = Annotated[
    str | None,
    typer.Option(
        "--reranker-api-key-env",
        help=(
            "Environment variable containing the bearer token for --reranker-invoke-url. "
            "If omitted, NVIDIA_API_KEY / NGC_API_KEY is used when set."
        ),
    ),
]
RerankerModelNameOption = Annotated[
    str | None,
    typer.Option(
        "--reranker-model-name",
        help="Optional reranker model name override (used by the local GPU reranker).",
    ),
]
RerankerBackendOption = Annotated[
    str | None,
    typer.Option(
        "--reranker-backend",
        help=(
            "Backend for the local GPU reranker when no --reranker-invoke-url is given: "
            "'vllm' (default — high-throughput batch) or 'hf' (HuggingFace, faster cold "
            "start; preferred for ad-hoc / single-query CLI use)."
        ),
    ),
]
RerankOption = Annotated[
    bool,
    typer.Option(
        "--rerank/--no-rerank",
        help=(
            "Enable reranking after vector retrieval. Default off. Implicitly enabled when "
            "any of --reranker-invoke-url / --reranker-model-name / --reranker-backend is set."
        ),
    ),
]
HybridOption = Annotated[
    bool,
    typer.Option(
        "--hybrid/--no-hybrid",
        help=(
            "Fused vector + full-text (BM25) retrieval; falls back to vector-only if the table "
            "has no FTS index. Opt-in (default off) preserves the legacy vector-only default."
        ),
    ),
]
OutputFormatOption = Annotated[
    str,
    typer.Option(
        "--format",
        help=(
            "'hits' (default): raw ranked hit list (source/page/text/modality/score). "
            "'evidence': answer-ready, fidelity-tagged, cited evidence + coverage."
        ),
    ),
]
MaxTextCharsOption = Annotated[
    int | None,
    typer.Option(
        "--max-text-chars",
        help="('hits' format only) Truncate each hit's text to N chars (0 = metadata-only). Default: full text.",
    ),
]
ServiceUrlOption = Annotated[
    str,
    typer.Option("--service-url", help="Base URL of the retriever service."),
]
ServiceApiTokenOption = Annotated[
    str | None,
    typer.Option(
        "--service-api-token",
        envvar="NEMO_RETRIEVER_API_TOKEN",
        help="Bearer token for authenticating with the retriever service. Falls back to $NEMO_RETRIEVER_API_TOKEN.",
    ),
]
