#!/usr/bin/env python3
"""Local debug script to run run_mode_ingest (structured path). Part of relational_db.

Run from repo root:
  PYTHONPATH=nemo_retriever/src python -m nemo_retriever.relational_db.debug_run_mode_ingest

List embedding models for your NVIDIA API key:
  PYTHONPATH=nemo_retriever/src python -m nemo_retriever.relational_db.debug_run_mode_ingest --list-models

Structured path: embed_params and vdb_params are required; otherwise ingest_structured() only runs
fetch_structured() and skips embed + vdb_upload. Structured data is written to table nv-ingest-structured.
Breakpoints: inprocess.run_pipeline_tasks_on_df (embed_text_main_text_embed); inprocess._embed_group (model.embed).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from nemo_retriever.application.modes.run_batch_structured import run_batch_structured
from nemo_retriever.params import (
    EmbedParams,
    IngestorCreateParams,
    StructuredExtractParams,
    VdbUploadParams,
)


def _get_embed_api_key() -> str:
    """Return embedding API key from env (NVIDIA_API_KEY)."""
    return (os.environ.get("NVIDIA_API_KEY") or "").strip()


def _check_embed_key() -> None:
    """Print whether embedding API key is set (masked). Run before ingest to debug 401."""
    key = _get_embed_api_key()
    if not key:
        print("DEBUG: No embedding API key found. Set NVIDIA_API_KEY (or load .env).")
        return
    k = key
    if len(k) <= 12:
        masked = "*" * len(k)
    else:
        masked = f"{k[:4]}...{k[-4:]}"
    print(
        f"DEBUG: Embedding API key is set ({masked}, len={len(k)}). "
        "If you still get 401, the key may be invalid or lack access to the embedding model."
    )


def list_embedding_models() -> None:
    """Call NVIDIA Inference API GET /v1/models and print embedding models for your key."""
    base = os.environ.get("NVIDIA_INFERENCE_BASE", "https://inference-api.nvidia.com")
    key = _get_embed_api_key()
    if not key:
        print("Set LLM_API_KEY or NVIDIA_API_KEY (e.g. in .env) then run again.")
        return
    url = f"{base.rstrip('/')}/v1/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"GET {url} failed: {e.code} {e.reason}")
        if e.fp:
            print(e.fp.read().decode())
        return
    models = data.get("data") if isinstance(data, dict) else []
    if not models:
        print("No models in response. Raw:", data)
        return
    ids = [m.get("id") for m in models if m.get("id")]
    embed_ids = [i for i in ids if "embed" in i.lower()]
    show = embed_ids if embed_ids else ids
    print("Available models for your key (embedding-related or all):")
    for i in sorted(show):
        print(f"  {i}")
    if embed_ids:
        print("\nSet embed_params.model_name to one of the ids above (e.g. the first).")
    else:
        print("\nNo model id contains 'embed'; check NVIDIA docs for which of the above support embeddings.")


def main() -> None:
    if "--list-models" in sys.argv:
        list_embedding_models()
        return
    _check_embed_key()

    create_params = IngestorCreateParams(
        documents=[],  # no unstructured documents
        allow_no_gpu=True,
    )

    structured_params = StructuredExtractParams(db_connection_string="./spider2.duckdb")
    # Set structured_params = None to skip structured path when Neo4j is not running.

    embed_params = EmbedParams(
        model_name="nvidia/nvidia/llama-3.2-nv-embedqa-1b-v2",
        embed_modality="text",
        text_column="text",
        inference_batch_size=16,
        embedding_endpoint="https://inference-api.nvidia.com",
        embedding_api_key=_get_embed_api_key() or None,
    )
    vdb_params = VdbUploadParams(
        lancedb=dict(
            lancedb_uri="lancedb",
            table_name="nv-ingest",
            overwrite=True,
            create_index=True,
        )
    )

    result = run_batch_structured(
        create_params=create_params,
        structured_params=structured_params,
        embed_params=embed_params,
        vdb_params=vdb_params,
    )

    print("ingest() result:", result)


if __name__ == "__main__":
    main()
