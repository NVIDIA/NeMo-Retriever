# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recall Evaluator — Designer component for running recall/BEIR evaluation against LanceDB.

Reuses the existing evaluation logic from ``nemo_retriever.recall.core`` and
``nemo_retriever.recall.beir``, and prints the standard run summary via
``nemo_retriever.utils.detection_summary.print_run_summary``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Annotated, Any, Optional

from nemo_retriever.graph.designer import Param, designer_component

logger = logging.getLogger(__name__)


@designer_component(
    name="Recall Evaluator",
    category="Evaluation",
    compute="cpu",
    description="Runs recall or BEIR evaluation against a LanceDB table and prints the standard run summary",
    category_color="#42d6a4",
    component_type="pipeline_evaluator",
)
class RecallEvaluatorActor:
    """Evaluation node placed after a LanceDB Writer in the Designer pipeline.

    Supports both ``recall`` (ground-truth query CSV) and ``beir``
    (HuggingFace BEIR dataset) evaluation modes.  After evaluation, calls
    ``print_run_summary`` to emit the same structured output that the
    batch pipeline produces.
    """

    def __init__(
        self,
        evaluation_mode: Annotated[str, Param(label="Evaluation Mode", choices=["recall", "beir"])] = "recall",
        lancedb_uri: Annotated[str, Param(label="LanceDB URI", placeholder="/path/to/lancedb")] = "lancedb",
        lancedb_table: Annotated[str, Param(label="Table Name")] = "nv-ingest",
        query_csv: Annotated[str, Param(label="Query CSV", placeholder="/path/to/query_gt.csv")] = "",
        embedding_model: Annotated[str, Param(label="Embedding Model")] = "nvidia/llama-nemotron-embed-1b-v2",
        recall_required: Annotated[bool, Param(label="Recall Required")] = True,
        match_mode: Annotated[str, Param(label="Match Mode", choices=["audio_segment"])] = "audio_segment",
        recall_adapter: Annotated[str, Param(label="Recall Adapter", choices=["none"])] = "none",
        ks: Annotated[str, Param(label="K Values", placeholder="1,3,5,10")] = "1,3,5,10",
        hybrid: Annotated[bool, Param(label="Hybrid Search")] = False,
        beir_loader: Annotated[str, Param(label="BEIR Loader", choices=["vidore_hf"])] = "vidore_hf",
        beir_dataset_name: Annotated[
            str, Param(label="BEIR Dataset Name", placeholder="e.g. vidore_v3_computer_science")
        ] = "",
        beir_split: Annotated[str, Param(label="BEIR Split")] = "test",
        beir_query_language: Annotated[str, Param(label="Query Language", placeholder="Optional (e.g. en, fr)")] = "",
        beir_doc_id_field: Annotated[
            str,
            Param(label="Doc ID Field", choices=["pdf_basename", "pdf_page", "source_id", "path"]),
        ] = "pdf_basename",
    ) -> None:
        self.evaluation_mode = evaluation_mode
        self.lancedb_uri = str(Path(lancedb_uri).expanduser().resolve())
        self.lancedb_table = lancedb_table
        self.query_csv = query_csv
        self.embedding_model = embedding_model
        self.recall_required = recall_required
        self.match_mode = match_mode
        self.recall_adapter = recall_adapter
        self.hybrid = hybrid
        self.beir_loader = beir_loader
        self.beir_dataset_name = beir_dataset_name
        self.beir_split = beir_split
        self.beir_query_language = beir_query_language or None
        self.beir_doc_id_field = beir_doc_id_field

        self._ks: tuple[int, ...] = (
            tuple(int(k) for k in ks.split(",") if k.strip()) if isinstance(ks, str) else tuple(ks)
        )
        if not self._ks:
            self._ks = (1, 3, 5, 10)

    def evaluate(self) -> dict[str, Any]:
        """Run the configured evaluation and print the standard run summary.

        Returns the ``summary_dict`` produced by ``print_run_summary``.
        """
        from nemo_retriever.model import resolve_embed_model
        from nemo_retriever.utils.detection_summary import print_run_summary

        resolved_model = resolve_embed_model(self.embedding_model)

        evaluation_label = "Recall"
        evaluation_total_time = 0.0
        evaluation_metrics: dict[str, float] = {}
        evaluation_query_count: Optional[int] = None

        recall_total_time = 0.0
        recall_metrics: dict[str, float] = {}

        if self.evaluation_mode == "beir":
            evaluation_label = "BEIR"
            from nemo_retriever.recall.beir import BeirConfig, evaluate_lancedb_beir

            beir_cfg = BeirConfig(
                lancedb_uri=self.lancedb_uri,
                lancedb_table=self.lancedb_table,
                embedding_model=resolved_model,
                loader=self.beir_loader,
                dataset_name=self.beir_dataset_name,
                split=self.beir_split,
                query_language=self.beir_query_language,
                doc_id_field=self.beir_doc_id_field,
                ks=self._ks,
                hybrid=self.hybrid,
            )
            eval_start = time.perf_counter()
            beir_dataset, _raw_hits, _run, evaluation_metrics = evaluate_lancedb_beir(beir_cfg)
            evaluation_total_time = time.perf_counter() - eval_start
            evaluation_query_count = len(beir_dataset.query_ids)
        else:
            if self.match_mode != "audio_segment" or self.recall_adapter != "none":
                raise ValueError("Legacy recall evaluation is only supported for audio_segment runs")

            from nemo_retriever.recall.core import RecallConfig, retrieve_and_score

            query_csv_path = Path(self.query_csv)
            if not query_csv_path.exists():
                logger.warning("Query CSV not found at %s; skipping recall evaluation.", query_csv_path)
                return {}

            recall_cfg = RecallConfig(
                lancedb_uri=self.lancedb_uri,
                lancedb_table=self.lancedb_table,
                embedding_model=resolved_model,
                ks=self._ks,
                match_mode=self.match_mode,
                hybrid=self.hybrid,
            )
            eval_start = time.perf_counter()
            _df_query, _gold, _raw_hits, _retrieved_keys, evaluation_metrics = retrieve_and_score(
                query_csv=query_csv_path,
                cfg=recall_cfg,
            )
            evaluation_total_time = time.perf_counter() - eval_start
            evaluation_query_count = len(_df_query.index)

            recall_metrics = dict(evaluation_metrics)
            recall_total_time = evaluation_total_time

        summary_dict = print_run_summary(
            processed_pages=-1,
            input_path=Path(self.lancedb_uri),
            hybrid=self.hybrid,
            lancedb_uri=self.lancedb_uri,
            lancedb_table_name=self.lancedb_table,
            total_time=-1,
            ingest_only_total_time=-1,
            ray_dataset_download_total_time=-1,
            lancedb_write_total_time=-1,
            evaluation_total_time=evaluation_total_time,
            evaluation_metrics=evaluation_metrics,
            recall_total_time=recall_total_time,
            recall_metrics=recall_metrics,
            evaluation_label=evaluation_label,
            evaluation_count=evaluation_query_count,
        )
        return summary_dict
