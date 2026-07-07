# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated

import typer

from nemo_retriever.models import VL_EMBED_MODEL, VL_RERANK_MODEL

DEFAULT_EMBED_MODEL = VL_EMBED_MODEL
DEFAULT_RERANK_MODEL = VL_RERANK_MODEL


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
    typer.Option(
        "--embed-model-name",
        help=f"Optional embedding model name override. Defaults to {DEFAULT_EMBED_MODEL} when omitted.",
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
    bool,
    typer.Option(
        "--rerank/--no-rerank",
        help=(
            "Enable reranking after vector retrieval. Default off. Implicitly enabled when "
            "any of --reranker-invoke-url / --reranker-model-name / --reranker-backend is set."
        ),
    ),
]
RetrievalModeOption = Annotated[
    str,
    typer.Option(
        "--retrieval-mode",
        help=(
            "Expert LanceDB retrieval mode: auto, dense, hybrid, or sparse. Default auto inspects the table "
            "and chooses the supported mode."
        ),
    ),
]
HybridOption = Annotated[
    bool,
    typer.Option(
        "--hybrid",
        help="Deprecated alias for --retrieval-mode hybrid.",
        hidden=True,
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
        help="Run an LLM-driven agentic (ReAct) retrieval loop instead of the default dense pass.",
    ),
]
AgenticLlmModelOption = Annotated[
    str | None,
    typer.Option(
        "--agentic-llm-model",
        help="Chat model the agent drives. Required when --agentic is set.",
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
AgenticBackendTopKOption = Annotated[
    int,
    typer.Option(
        "--agentic-backend-top-k",
        min=1,
        help="Backend retrieve-pool depth per agentic retrieval call.",
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
    float,
    typer.Option(
        "--agentic-temperature",
        min=0.0,
        help="Sampling temperature for agentic LLM calls (0.0 = greedy).",
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

# ── Answer / generation options ────────────────────────────────────────────────

LlmModelOption = Annotated[
    str | None,
    typer.Option(
        "--llm-model",
        help=(
            "LLM model string for RAG generation. Uses the litellm provider prefix convention "
            "(e.g. 'nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5', 'openai/gpt-4o'). "
            "Defaults to the client's built-in default when omitted."
        ),
    ),
]
LlmInvokeUrlOption = Annotated[
    str | None,
    typer.Option(
        "--llm-invoke-url",
        help="OpenAI-compatible chat-completions endpoint for generation (e.g. https://integrate.api.nvidia.com/v1).",
    ),
]
LlmApiKeyEnvOption = Annotated[
    str | None,
    typer.Option(
        "--llm-api-key-env",
        help=(
            "Environment variable containing the API key for --llm-invoke-url. "
            "Falls back to NVIDIA_API_KEY / NGC_API_KEY when omitted."
        ),
    ),
]
LlmMaxTokensOption = Annotated[
    int,
    typer.Option("--llm-max-tokens", min=1, help="Maximum tokens for the generated answer."),
]
LlmTemperatureOption = Annotated[
    float,
    typer.Option("--llm-temperature", min=0.0, help="Sampling temperature for generation (0.0 = greedy)."),
]
ReasoningOption = Annotated[
    bool | None,
    typer.Option(
        "--reasoning/--no-reasoning",
        help=("Enable or disable chain-of-thought reasoning on Nemotron models. " "Omit to use the model's default."),
    ),
]
ReferenceOption = Annotated[
    str | None,
    typer.Option(
        "--reference",
        help=(
            "Gold answer for automatic scoring (token F1, exact match, context coverage). "
            "Scoring is skipped when omitted."
        ),
    ),
]
MultimodalOption = Annotated[
    bool,
    typer.Option(
        "--multimodal",
        is_flag=True,
        help=(
            "Enable Vision-Language Model (VLM) generation. Visual chunks (image, chart, "
            "infographic, table) that have a stored image URI are loaded and sent to the "
            "VLM inline alongside their text captions. Requires --llm-model to point to a "
            "multimodal-capable endpoint."
        ),
    ),
]
