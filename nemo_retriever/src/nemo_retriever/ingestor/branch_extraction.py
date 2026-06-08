# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Branch execution for manifest-planned retriever ingest extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable

from nemo_retriever.graph import InprocessExecutor, RayDataExecutor
from nemo_retriever.graph.ingestor_runtime import batch_tuning_to_node_overrides, build_graph, build_post_extract_graph
from nemo_retriever.ingestor.manifest import (
    ExtractionBranchPlan,
    ResolvedExtractionInputs,
    format_branch_summary,
    resolve_branch_extraction_inputs,
)


logger = logging.getLogger(__name__)


def ensure_pandas_columns(batch_df: Any, *, columns: tuple[str, ...]) -> Any:
    """Pad a pandas batch to a stable schema before unioning branch outputs."""

    for column in columns:
        if column not in batch_df.columns:
            batch_df[column] = None
    return batch_df.loc[:, list(columns)]


@dataclass
class ExtractionBranchExecutor:
    """Run manifest extraction branches and common post-extraction stages."""

    run_mode: str
    branches: tuple[ExtractionBranchPlan, ...]
    documents: list[str]
    buffers: list[tuple[str, BytesIO]]
    split_config: dict[str, Any]
    extract_params: Any | None
    text_params: Any | None
    html_params: Any | None
    audio_chunk_params: Any | None
    asr_params: Any | None
    video_frame_params: Any | None
    video_text_dedup_params: Any | None
    av_fuse_params: Any | None
    embed_params: Any | None
    caption_params: Any | None
    dedup_params: Any | None
    store_params: Any | None
    vdb_upload_params: Any | None
    webhook_params: Any | None
    post_extract_order: tuple[str, ...]
    ray_address: str | None
    batch_size: int
    num_cpus: float
    num_gpus: float
    node_overrides: dict[str, dict[str, Any]]
    show_progress: bool
    allow_no_gpu: bool
    ensure_batch_runtime: Callable[[], tuple[Any, Any]]

    def execute(self) -> Any:
        logger.info(
            "Retriever ingest manifest planned %d extraction branches: %s",
            len(self.branches),
            format_branch_summary(self.branches),
        )
        if self.run_mode == "batch":
            return self._execute_batch()
        return self._execute_inprocess()

    def _execute_batch(self) -> Any:
        _ray, cluster_resources = self.ensure_batch_runtime()
        effective_allow_no_gpu = self.allow_no_gpu or cluster_resources.available_gpu_count() == 0
        branch_datasets: list[Any] = []
        for branch in self.branches:
            effective_extraction = self._resolve_branch(branch)
            logger.info(
                "Retriever ingest extraction branch family=%s files=%d graph_mode=%s",
                branch.family,
                len(branch.input_paths),
                effective_extraction.extraction_mode,
            )
            graph = self._build_extraction_only_graph(effective_extraction)
            derived_overrides = batch_tuning_to_node_overrides(
                effective_extraction.extract_params,
                None,
                store_params=None,
                cluster_resources=cluster_resources,
                allow_no_gpu=effective_allow_no_gpu,
                caption_params=None,
                video_frame_params=effective_extraction.video_frame_params,
            )
            executor = self._ray_executor(graph, derived_overrides)
            branch_datasets.append(executor.build_dataset(list(branch.input_paths)))

        normalized = normalize_ray_branch_datasets(branch_datasets)
        combined = normalized[0]
        for branch_ds in normalized[1:]:
            combined = combined.union(branch_ds)

        logger.info("Retriever ingest post-extraction stages: %s", format_post_stage_summary(self.post_extract_order))
        post_graph = build_post_extract_graph(
            dedup_params=self.dedup_params,
            embed_params=self.embed_params,
            caption_params=self.caption_params,
            store_params=self.store_params,
            vdb_upload_params=self.vdb_upload_params,
            webhook_params=self.webhook_params,
            stage_order=self.post_extract_order,
            reshape_content_before_embed=self._should_reshape_content_before_embed(),
        )
        post_overrides = batch_tuning_to_node_overrides(
            None,
            self.embed_params,
            store_params=self.store_params,
            cluster_resources=cluster_resources,
            allow_no_gpu=effective_allow_no_gpu,
            caption_params=self.caption_params,
            video_frame_params=None,
        )
        return self._ray_executor(post_graph, post_overrides).ingest(combined)

    def _execute_inprocess(self) -> Any:
        frames = []
        for branch in self.branches:
            effective_extraction = self._resolve_branch(branch)
            logger.info(
                "Retriever ingest extraction branch family=%s files=%d graph_mode=%s",
                branch.family,
                len(branch.input_paths),
                effective_extraction.extraction_mode,
            )
            graph = self._build_extraction_only_graph(effective_extraction)
            executor = InprocessExecutor(graph, show_progress=self.show_progress)
            frames.append(executor.ingest(self._inprocess_branch_input(branch)))

        combined = concat_dataframes(frames)
        logger.info("Retriever ingest post-extraction stages: %s", format_post_stage_summary(self.post_extract_order))
        post_graph = build_post_extract_graph(
            dedup_params=self.dedup_params,
            embed_params=self.embed_params,
            caption_params=self.caption_params,
            store_params=self.store_params,
            vdb_upload_params=self.vdb_upload_params,
            webhook_params=self.webhook_params,
            stage_order=self.post_extract_order,
            reshape_content_before_embed=self._should_reshape_content_before_embed(),
        )
        return InprocessExecutor(post_graph, show_progress=self.show_progress).ingest(combined)

    def _should_reshape_content_before_embed(self) -> bool:
        return any(branch.family in {"pdf", "image"} for branch in self.branches)

    def _resolve_branch(self, branch: ExtractionBranchPlan) -> ResolvedExtractionInputs:
        return resolve_branch_extraction_inputs(
            branch,
            extract_params=self.extract_params,
            text_params=self.text_params,
            html_params=self.html_params,
            audio_chunk_params=self.audio_chunk_params,
            asr_params=self.asr_params,
            video_frame_params=self.video_frame_params,
            video_text_dedup_params=self.video_text_dedup_params,
            av_fuse_params=self.av_fuse_params,
        )

    def _build_extraction_only_graph(self, effective_extraction: ResolvedExtractionInputs) -> Any:
        return build_graph(
            extraction_mode=effective_extraction.extraction_mode,
            extract_params=effective_extraction.extract_params,
            text_params=effective_extraction.text_params,
            html_params=effective_extraction.html_params,
            audio_chunk_params=effective_extraction.audio_chunk_params,
            asr_params=effective_extraction.asr_params,
            video_frame_params=effective_extraction.video_frame_params,
            video_text_dedup_params=effective_extraction.video_text_dedup_params,
            av_fuse_params=effective_extraction.av_fuse_params,
            split_config=self.split_config,
            stage_order=(),
        )

    def _ray_executor(self, graph: Any, derived_overrides: dict[str, dict[str, Any]]) -> RayDataExecutor:
        return RayDataExecutor(
            graph,
            ray_address=self.ray_address,
            batch_size=self.batch_size,
            num_cpus=self.num_cpus,
            num_gpus=self.num_gpus,
            node_overrides=merge_node_overrides(derived_overrides, self.node_overrides),
        )

    def _inprocess_branch_input(self, branch: ExtractionBranchPlan) -> Any:
        if not self.buffers:
            return list(branch.input_paths)

        import pandas as pd

        buffer_by_name = {name: buf for name, buf in self.buffers}
        file_paths: list[str] = []
        buffer_rows: list[dict[str, Any]] = []
        for path in branch.input_paths:
            if path in buffer_by_name:
                buffer_rows.append({"bytes": buffer_by_name[path].getvalue(), "path": path})
            else:
                file_paths.append(path)

        frames = []
        if file_paths:
            frames.append(InprocessExecutor._load_files(file_paths))
        if buffer_rows:
            frames.append(pd.DataFrame(buffer_rows))
        return concat_dataframes(frames)


def merge_node_overrides(
    derived_overrides: dict[str, dict[str, Any]],
    explicit_overrides: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Merge generated and caller-supplied Ray node override dictionaries.

    Parameters
    ----------
    derived_overrides
        Overrides calculated from runtime resource heuristics and tuning
        parameters.
    explicit_overrides
        Overrides provided directly by the caller. Values here take precedence
        over matching keys from ``derived_overrides``.

    Returns
    -------
    dict[str, dict[str, Any]]
        A merged override mapping keyed by graph node name.
    """

    merged_overrides: dict[str, dict[str, Any]] = {}
    for node_name in set(derived_overrides) | set(explicit_overrides):
        merged_overrides[node_name] = {
            **derived_overrides.get(node_name, {}),
            **explicit_overrides.get(node_name, {}),
        }
    return merged_overrides


def concat_dataframes(frames: list[Any]) -> Any:
    """Concatenate branch DataFrames while preserving the union of columns.

    Parameters
    ----------
    frames
        Pandas DataFrames produced by extraction branches.

    Returns
    -------
    Any
        A pandas DataFrame with every column seen across the inputs. When
        ``frames`` is empty, returns an empty frame with ``bytes`` and ``path``
        columns.
    """

    import pandas as pd

    if not frames:
        return pd.DataFrame(columns=["bytes", "path"])
    columns: list[str] = []
    seen: set[str] = set()
    for frame in frames:
        for column in frame.columns:
            if column not in seen:
                columns.append(column)
                seen.add(column)
    normalized = [frame.reindex(columns=columns) for frame in frames]
    return pd.concat(normalized, ignore_index=True, sort=False)


def normalize_ray_branch_datasets(branch_datasets: list[Any]) -> list[Any]:
    """Pad Ray branch datasets to a common schema before unioning them.

    Parameters
    ----------
    branch_datasets
        Ray ``Dataset`` objects produced by manifest extraction branches.

    Returns
    -------
    list[Any]
        Datasets whose pandas batches are projected to a stable column order.
        If any dataset lacks an already-known schema, the original datasets are
        returned unchanged so Ray does not eagerly execute extraction just to
        discover schema information.
    """

    columns: list[str] = []
    seen: set[str] = set()
    for dataset in branch_datasets:
        dataset_columns = ray_dataset_columns(dataset)
        if not dataset_columns:
            # Avoid eager schema discovery: Ray computes missing schemas by
            # executing a limit=1 plan, which pre-runs extraction branches.
            return branch_datasets
        for column in dataset_columns:
            if column not in seen:
                columns.append(column)
                seen.add(column)
    if not columns:
        return branch_datasets
    stable_columns = tuple(columns)
    return [
        dataset.map_batches(
            ensure_pandas_columns,
            batch_format="pandas",
            fn_kwargs={"columns": stable_columns},
        )
        for dataset in branch_datasets
    ]


def ray_dataset_columns(dataset: Any) -> tuple[str, ...]:
    """Return known column names from a Ray dataset without forcing execution.

    Parameters
    ----------
    dataset
        Ray ``Dataset`` or test double exposing ``schema``.

    Returns
    -------
    tuple[str, ...]
        Column names if Ray already has schema metadata, otherwise an empty
        tuple.

    Raises
    ------
    Exception
        Propagates non-compatibility errors raised by ``dataset.schema``.
    """

    try:
        schema = dataset.schema(fetch_if_missing=False)
    except TypeError:
        schema = dataset.schema()
    if schema is None:
        return ()
    names = getattr(schema, "names", None)
    if callable(names):
        names = names()
    if names is None:
        base_schema = getattr(schema, "base_schema", None)
        names = getattr(base_schema, "names", None) if base_schema is not None else None
        if callable(names):
            names = names()
    if names is None:
        return ()
    return tuple(str(name) for name in names)


def format_post_stage_summary(post_extract_order: tuple[str, ...]) -> str:
    """Format post-extraction stage names for log messages.

    Parameters
    ----------
    post_extract_order
        Ordered stage names that will run after branch union.

    Returns
    -------
    str
        Comma-separated stage names, or ``"none"`` when no post stages are
        configured.
    """

    return ", ".join(post_extract_order) if post_extract_order else "none"
