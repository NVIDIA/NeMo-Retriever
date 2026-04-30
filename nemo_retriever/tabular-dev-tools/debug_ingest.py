# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Debug entry-point: builds a Graph from individual tabular operators so you
can set breakpoints and step through each stage of the tabular ingest pipeline.

should be stored in .vscode/debug_ingest.py

Runs entirely on CPU — embedding is delegated to the hosted NVIDIA NIM endpoint
(build.nvidia.com).  Before running, export your API key:

    export NVIDIA_API_KEY="nvapi-..."

Tweak the params below to match your local setup before running.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# tabular-dev-tools is not an installed package; extend sys.path so DuckDB
# can be imported directly when running this debug script.
sys.path.insert(0, str(Path(__file__).parents[1] / "nemo_retriever" / "tabular-dev-tools"))
from duckdb_connector import DuckDB  # noqa: E402

from nemo_retriever.graph import Graph
from nemo_retriever.graph.tabular_schema_extract_operator import TabularSchemaExtractOp
from nemo_retriever.graph.tabular_fetch_embeddings_operator import TabularFetchEmbeddingsOp
from nemo_retriever.text_embed.operators import _BatchEmbedActor
from nemo_retriever.retriever import Retriever
from nemo_retriever.tabular_data.retrieval.text_to_sql.main import get_agent_response
from nemo_retriever.tabular_data.retrieval.text_to_sql.state import AgentPayload
from nemo_retriever.vector_store.lancedb_store import handle_lancedb
from nemo_retriever.params import (
    EmbedParams,
    TabularExtractParams,
    VdbUploadParams,
)

# ── Validate required environment variables ───────────────────────────────────

_NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
if not _NVIDIA_API_KEY:
    raise EnvironmentError(
        "NVIDIA_API_KEY is not set. "
        "Export it before running:\n\n"
        "    export NVIDIA_API_KEY='nvapi-...'\n\n"
        "Get your key at https://build.nvidia.com"
    )

# ── Configure your run here ───────────────────────────────────────────────────

# DuckDB connector — path to the .duckdb file (relative to workspace root)
# or ":memory:" for an ephemeral in-memory database.
connector = DuckDB("./spider2.duckdb")
TABULAR_PARAMS = TabularExtractParams(
    connector=connector,
)

# Remote NIM embedding endpoint — no local GPU required.
# Model hosted on build.nvidia.com; billed against your NVIDIA API key.
EMBED_PARAMS = EmbedParams(
    embed_invoke_url="https://integrate.api.nvidia.com/v1",
    model_name="nvidia/llama-nemotron-embed-1b-v2",
    api_key=_NVIDIA_API_KEY,
    embed_modality="text",
)

# Post-merge schema: VdbUploadParams now uses (vdb_op, vdb_kwargs).
# The kwargs are the LanceDbParams fields when vdb_op == "lancedb".
VDB_PARAMS = VdbUploadParams(
    vdb_op="lancedb",
    vdb_kwargs={
        "lancedb_uri": "lancedb",
        "table_name": "nv-ingest-tabular",
        "overwrite": True,
        "create_index": True,
    },
)

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    graph = (
        Graph()
        >> TabularSchemaExtractOp(tabular_params=TABULAR_PARAMS)
        >> TabularFetchEmbeddingsOp(database_name=connector.database_name)
        >> _BatchEmbedActor(params=EMBED_PARAMS)
    )

    results = graph.execute(None)
    result_df = results[0] if results else None

    if result_df is not None and not result_df.empty:
        lancedb_kwargs = VDB_PARAMS.vdb_kwargs
        handle_lancedb(
            result_df.to_dict(orient="records"),
            uri=lancedb_kwargs["lancedb_uri"],
            table_name=lancedb_kwargs["table_name"],
        )
        print("Tabular ingest result:", len(result_df), "rows written to LanceDB")
    else:
        print("Tabular ingest result: no rows produced")

    lancedb_kwargs = VDB_PARAMS.vdb_kwargs
    retriever = Retriever(
        vdb="lancedb",
        vdb_kwargs={
            "uri": lancedb_kwargs["lancedb_uri"],
            "table_name": lancedb_kwargs["table_name"],
        },
        top_k=15,
        embedding_api_key=_NVIDIA_API_KEY,
        embedding_http_endpoint=EMBED_PARAMS.embed_invoke_url,
    )

    payload: AgentPayload = {
        "question": "sum unqieue seat ids",
        "retriever": retriever,
        "connector": connector,
        "path_state": {},
        "custom_prompts": "",
        "acronyms": "",
    }

    agent_result = get_agent_response(payload)
    print("get_agent_response result:", agent_result)
