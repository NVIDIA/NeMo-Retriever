# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated

import typer

from nemo_retriever._agentic.nemo_agent.llm import get_available_backends
from nemo_retriever.models import VL_EMBED_MODEL, VL_RERANK_MODEL

DEFAULT_EMBED_MODEL = VL_EMBED_MODEL
DEFAULT_RERANK_MODEL = VL_RERANK_MODEL

# Advertised in --agentic-llm-client help; sourced from the registry so a newly
# registered client shows up without editing this string.
_AGENTIC_LLM_CLIENT_CHOICES = ", ".join(get_available_backends())


QueryArgument = Annotated[str, typer.Argument(..., help="Query text.")]
TopKOption = Annotated[
    int,
    typer.Option("--top-k", min=1, help="Final number of results to return after filtering and deduplication."),
]
CandidateKOption = Annotated[
    int | None,
    typer.Option(
        "--candidate-k",
        min=1,
        help=(
            "Number of raw results to retrieve before filtering, page deduplication, "
            "and final truncation; must be greater than or equal to --top-k."
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
        help=(
            "Comma-separated content types to keep, such as text,table. Requires "
            "content-type metadata; untyped hits are excluded."
        ),
    ),
]
LanceDbUriOption = Annotated[
    str,
    typer.Option(
        "--lancedb-uri",
        help="LanceDB database URI to read; match the value used for retriever ingest --lancedb-uri.",
    ),
]
TableNameOption = Annotated[
    str,
    typer.Option(
        "--table-name",
        help="LanceDB table name to read; match the value used for retriever ingest --table-name.",
    ),
]
EmbedInvokeUrlOption = Annotated[
    str | None,
    typer.Option("--embed-invoke-url", envvar="EMBED_INVOKE_URL", help="Embedding NIM endpoint URL."),
]
EmbedModelNameOption = Annotated[
    str | None,
    typer.Option(
        "--embed-model-name",
        envvar="EMBED_MODEL_NAME",
        help=(
            "Embedding model override. When omitted, use the model recorded on the selected table, "
            f"then fall back to {DEFAULT_EMBED_MODEL} for a legacy table without metadata."
        ),
    ),
]
EmbedModelProviderPrefixOption = Annotated[
    str | None,
    typer.Option(
        "--embed-model-provider-prefix",
        envvar="EMBED_MODEL_PROVIDER_PREFIX",
        help="Optional LiteLLM provider prefix prepended to the remote embedding model name.",
    ),
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
        help=("Optional reranker model name override. When reranking locally, " f"defaults to {DEFAULT_RERANK_MODEL}."),
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
    bool | None,
    typer.Option(
        "--rerank/--no-rerank",
        help=(
            "Enable reranking after vector retrieval. Default off. When neither flag is passed, implicitly enabled "
            "when any of --reranker-invoke-url / --reranker-model-name / --reranker-backend is set."
        ),
    ),
]
RetrievalModeOption = Annotated[
    str,
    typer.Option(
        "--retrieval-mode",
        help=(
            "Advanced override: auto, dense, hybrid, or sparse. Leave at auto to inspect the table and use "
            "the supported default mode."
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
AgenticOption = Annotated[
    bool,
    typer.Option(
        "--agentic",
        help="Run an LLM-driven agentic (ReAct) retrieval loop instead of the default retrieval pass.",
    ),
]
AgenticLlmModelOption = Annotated[
    str | None,
    typer.Option(
        "--agentic-llm-model",
        help=(
            "Chat model the agent drives. Defaults to nemotron-8b for local in-process runs; "
            "required when --agentic-invoke-url is provided."
        ),
    ),
]
AgenticInvokeUrlOption = Annotated[
    str | None,
    typer.Option(
        "--agentic-invoke-url",
        help="OpenAI-compatible chat-completions endpoint for the agent LLM (agentic mode).",
    ),
]
AgenticReasoningEffortOption = Annotated[
    str | None,
    typer.Option(
        "--agentic-reasoning-effort",
        help="reasoning_effort forwarded on agentic LLM calls.",
    ),
]
AgenticReactMaxStepsOption = Annotated[
    int,
    typer.Option(
        "--agentic-react-max-steps",
        min=1,
        help="Maximum ReAct loop iterations for the agentic query.",
    ),
]
AgenticTextTruncationOption = Annotated[
    int,
    typer.Option(
        "--agentic-text-truncation",
        min=0,
        help="Max characters of each candidate shown to the agent; 0 disables truncation.",
    ),
]
AgenticTemperatureOption = Annotated[
    float | None,
    typer.Option(
        "--agentic-temperature",
        min=0.0,
        help=(
            "Sampling temperature for agentic LLM calls. "
            "Omit to leave it unset (endpoint/model default; 0.0 = greedy)."
        ),
    ),
]
AgenticLocalTensorParallelSizeOption = Annotated[
    int,
    typer.Option(
        "--agentic-local-tensor-parallel-size",
        min=1,
        help=(
            "vLLM tensor_parallel_size for the in-process agent LLM. "
            "Use 2+ with matching CUDA_VISIBLE_DEVICES for multi-GPU local "
            "profiles (e.g. super-49b); ignored when --agentic-invoke-url is set."
        ),
    ),
]
AgenticLlmClientOption = Annotated[
    str | None,
    typer.Option(
        "--agentic-llm-client",
        help=(
            "LLM client that builds the agent LLM in agentic mode. Optional: defaults to "
            "'callable' for both in-process local runs and remote (--agentic-invoke-url) runs. "
            f"Registered clients: {_AGENTIC_LLM_CLIENT_CHOICES}. Any client other than 'callable' "
            "is remote-only and requires --agentic-invoke-url."
        ),
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
