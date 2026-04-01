# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph operator for the full tabular ingestion pipeline."""

from __future__ import annotations

from typing import Any

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.params import EmbedParams
from nemo_retriever.params import TabularExtractParams
from nemo_retriever.params import VdbUploadParams


_TABULAR_TABLE = "nv-ingest-tabular"


class TabularIngestOperator(AbstractOperator, CPUOperator):
    """Graph operator that orchestrates the full tabular ingestion pipeline.

    Mirrors the steps of :meth:`BatchIngestor.ingest_tabular` as a first-class
    :class:`AbstractOperator` so it can be composed in a :class:`Graph` without
    touching the ingestor API.

    Steps executed in :meth:`process`:

    1. ``extract_tabular_db_data``      — pull schema entities from the DB
    2. ``store_relational_db_in_neo4j`` — write entities as graph nodes
    3. ``populate_tabular_semantic_layer`` (future)
    4. ``populate_tabular_usage_weights``  (future)
    5. ``generate_tabular_descriptions``   (future)
    6. ``fetch_tabular_embedding_dataframe`` → Ray Dataset
    7. :class:`_BatchEmbedActor`         — GPU/remote batch embedding
    8. :class:`_LanceDBWriteActor`       — streaming write to LanceDB
    """

    def __init__(
        self,
        *,
        tabular_params: TabularExtractParams | None = None,
        embed_params: EmbedParams | None = None,
        vdb_params: VdbUploadParams | None = None,
        allow_no_gpu: bool = False,
    ) -> None:
        super().__init__()
        self._tabular_params = tabular_params
        self._embed_params = embed_params
        self._vdb_params = vdb_params
        self._allow_no_gpu = allow_no_gpu

    def preprocess(self, data: Any, **kwargs: Any) -> TabularExtractParams | None:
        """Accept a :class:`TabularExtractParams` passed as graph data, or fall back to stored params."""
        if isinstance(data, TabularExtractParams):
            return data
        return self._tabular_params

    def process(self, data: TabularExtractParams | None, **kwargs: Any) -> Any:
        """Run the full tabular ingestion pipeline."""
        import ray
        import ray.data as rd

        from nemo_retriever.ingest_modes.batch import _BatchEmbedActor, _LanceDBWriteActor
        from nemo_retriever.params.utils import build_embed_kwargs
        from nemo_retriever.tabular_data.ingestion.extract_data import (
            extract_tabular_db_data,
            store_relational_db_in_neo4j,
        )
        from nemo_retriever.tabular_data.ingestion.embeddings import fetch_tabular_embedding_dataframe
        from nemo_retriever.tabular_data.neo4j import get_neo4j_conn
        from nemo_retriever.utils.ray_resource_hueristics import (
            gather_cluster_resources,
            resolve_requested_plan,
        )

        neo4j_conn = get_neo4j_conn()

        # Steps 1–2: extract from DB and populate Neo4j graph
        schema_data = extract_tabular_db_data(params=data)
        store_relational_db_in_neo4j(data=schema_data, neo4j_conn=neo4j_conn)

        # Steps 3–5: semantic layer, usage weights, descriptions (not yet implemented)

        # Step 6: build Ray Dataset from Neo4j embedding metadata
        df = fetch_tabular_embedding_dataframe()
        if df.empty:
            return None
        rd_dataset = rd.from_pandas(df)

        # Step 7: embed
        if self._embed_params is not None:
            cluster_resources = gather_cluster_resources(ray)
            requested_plan = resolve_requested_plan(
                cluster_resources=cluster_resources,
                allow_no_gpu=self._allow_no_gpu,
            )

            embed_kwargs = build_embed_kwargs(self._embed_params, include_batch_tuning=True)
            endpoint = (embed_kwargs.get("embedding_endpoint") or embed_kwargs.get("embed_invoke_url") or "").strip()
            embed_actor_num_gpus = 0.0 if endpoint else requested_plan.get_embed_gpus_per_actor()

            rd_dataset = rd_dataset.repartition(target_num_rows_per_block=requested_plan.get_embed_batch_size())
            rd_dataset = rd_dataset.map_batches(
                _BatchEmbedActor,
                batch_size=requested_plan.get_embed_batch_size(),
                batch_format="pandas",
                num_gpus=embed_actor_num_gpus,
                compute=rd.ActorPoolStrategy(
                    initial_size=requested_plan.get_embed_initial_actors(),
                    min_size=requested_plan.get_embed_min_actors(),
                    max_size=requested_plan.get_embed_max_actors(),
                ),
                fn_constructor_kwargs={"params": self._embed_params},
            )

        # Step 8: write to LanceDB
        if self._vdb_params is not None:
            tabular_vdb_params = self._vdb_params.model_copy(
                update={"lancedb": self._vdb_params.lancedb.model_copy(update={"table_name": _TABULAR_TABLE})}
            )
            rd_dataset = rd_dataset.map_batches(
                _LanceDBWriteActor,
                batch_format="pandas",
                num_cpus=1,
                compute=rd.ActorPoolStrategy(size=1),
                fn_constructor_kwargs={"params": tabular_vdb_params},
            )

        result = rd_dataset.count()

        if self._vdb_params is not None:
            import lancedb as _lancedb

            from nemo_retriever.vector_store.lancedb_store import LanceDBConfig, create_lancedb_index

            lancedb_params = self._vdb_params.lancedb
            cfg = LanceDBConfig(
                uri=lancedb_params.lancedb_uri,
                table_name=_TABULAR_TABLE,
                overwrite=lancedb_params.overwrite,
                create_index=lancedb_params.create_index,
            )
            if cfg.create_index:
                db = _lancedb.connect(uri=cfg.uri)
                table = db.open_table(cfg.table_name)
                create_lancedb_index(table, cfg=cfg)

        return result

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
