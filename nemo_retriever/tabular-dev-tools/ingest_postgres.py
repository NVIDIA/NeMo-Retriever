"""Ingest the local docker-compose Postgres into Neo4j via NeMo Retriever.

Run after ``docker compose up -d`` and ``scripts.seed_local_postgres``.

Usage::

    PYTHONPATH=nemo_retriever/src uv run --no-sync python .vscode/ingest_postgres.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nemo_retriever" / "tabular-dev-tools"))
from apply_metadata import apply_metadata  # noqa: E402
from postgres_connector import PostgresDatabase  # noqa: E402
from nemo_retriever.graph import Graph
from nemo_retriever.graph.tabular_schema_extract_operator import TabularSchemaExtractOp
from nemo_retriever.graph.tabular_fetch_embeddings_operator import (
    TabularFetchEmbeddingsOp,
)
from nemo_retriever.text_embed.operators import _BatchEmbedActor
from nemo_retriever.retriever import Retriever
from nemo_retriever.tabular_data.retrieval.text_to_sql.main import get_agent_response
from nemo_retriever.tabular_data.retrieval.text_to_sql.state import AgentPayload
from nemo_retriever.tabular_data.retrieval.deep_agent.main import (
    get_agent_response as get_deep_agent_response,
)
from nemo_retriever.tabular_data.retrieval.deep_agent.state import (
    AgentPayload as DeepAgentPayload,
)
from nemo_retriever.vdb import IngestVdbOperator
from nemo_retriever.params import (
    EmbedParams,
    TabularExtractParams,
    VdbUploadParams,
)

logger = logging.getLogger("ingest_postgres")

_NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
if not _NVIDIA_API_KEY:
    raise EnvironmentError(
        "NVIDIA_API_KEY is not set. "
        "Export it before running:\n\n"
        "    export NVIDIA_API_KEY='nvapi-...'\n\n"
        "Get your key at https://build.nvidia.com"
    )

EMBED_PARAMS = EmbedParams(
    embed_invoke_url="https://integrate.api.nvidia.com/v1",
    model_name="nvidia/llama-nemotron-embed-1b-v2",
    api_key=_NVIDIA_API_KEY,
    embed_modality="text",
)

# vdb_kwargs are forwarded straight to ``nemo_retriever.vdb.lancedb.LanceDB``.
VDB_PARAMS = VdbUploadParams(
    vdb_op="lancedb",
    vdb_kwargs={
        "uri": "lancedb",
        "table_name": "nv-ingest-tabular",
        "overwrite": True,
    },
)

DATABASE: str = os.environ.get("POSTGRES_DB", "testdb")


def _conn_string(db: str) -> str:
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


_CONNECTOR: PostgresDatabase | None = None


def _get_connector() -> PostgresDatabase:
    """Open the Postgres connection lazily and reuse across phases."""
    global _CONNECTOR
    if _CONNECTOR is None:
        _CONNECTOR = PostgresDatabase(_conn_string(DATABASE))
    return _CONNECTOR


def run_ingest() -> None:
    """Ingest the Postgres schema into Neo4j and write embeddings to LanceDB."""
    connector = _get_connector()

    TABULAR_PARAMS = TabularExtractParams(
        connector=connector,
    )

    extract_graph = Graph() >> TabularSchemaExtractOp(tabular_params=TABULAR_PARAMS)
    extract_graph.execute(None)

    apply_metadata(connector.database_name)

    embed_graph = (
        Graph()
        >> TabularFetchEmbeddingsOp(database_name=connector.database_name)
        >> _BatchEmbedActor(params=EMBED_PARAMS)
    )
    results = embed_graph.execute(None)
    result_df = results[0] if results else None

    if result_df is not None and not result_df.empty:
        ingest_op = IngestVdbOperator(
            vdb_op=VDB_PARAMS.vdb_op,
            vdb_kwargs=VDB_PARAMS.vdb_kwargs,
        )
        ingest_op(result_df.to_dict(orient="records"))
        logger.info("Tabular ingest result: %d rows written to LanceDB", len(result_df))
    else:
        logger.info("Tabular ingest result: no rows produced")


def run_retrieve() -> None:
    """Run the text-to-SQL agent against the previously ingested LanceDB."""
    connector = _get_connector()
    lancedb_kwargs = VDB_PARAMS.vdb_kwargs
    retriever = Retriever(
        vdb="lancedb",
        vdb_kwargs={
            "uri": lancedb_kwargs["uri"],
            "table_name": lancedb_kwargs["table_name"],
        },
        top_k=15,
        embedding_api_key=_NVIDIA_API_KEY,
        embedding_http_endpoint=EMBED_PARAMS.embed_invoke_url,
    )

    payload: AgentPayload = {
        "question": "How many DORs were created",
        "retriever": retriever,
        "connector": connector,
        "path_state": {},
        "custom_prompts": "",
        "acronyms": "",
    }

    agent_result = get_agent_response(payload)
    logger.info("get_agent_response result: %s", agent_result)


def run_retrieve_deep() -> None:
    """Run the deep-agent text-to-SQL pipeline against the previously ingested LanceDB."""
    connector = _get_connector()

    payload: DeepAgentPayload = {
        "question": "How many DORs were created",
        "history": [],
        "path_state": {},
        "db_connector": connector,
    }

    agent_result = get_deep_agent_response(payload)
    logger.info("get_deep_agent_response result: %s", agent_result)


_ALL_MODES = ("ingest", "retrieve", "retrieve-deep")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=_ALL_MODES,
        nargs="*",
        default=None,
        help="Phases to run. Pass one or more (e.g. --mode ingest retrieve). " "Default: run all phases.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    modes = args.mode if args.mode else _ALL_MODES
    if "ingest" in modes:
        run_ingest()
    if "retrieve" in modes:
        run_retrieve()
    if "retrieve-deep" in modes:
        run_retrieve_deep()
