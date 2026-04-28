# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GraphIngestor: builds operator graphs directly and runs them via an executor.

Unlike the high-level :func:`create_ingestor` factory this class constructs
the :class:`~nemo_retriever.graph.Graph` itself—using
:func:`~nemo_retriever.graph.ingestor_runtime.build_graph`—and
passes it to a :class:`~nemo_retriever.graph.RayDataExecutor` or
:class:`~nemo_retriever.graph.InprocessExecutor` for execution.

Usage::

    from nemo_retriever.graph_ingestor import GraphIngestor
    from nemo_retriever.params import ExtractParams, EmbedParams

    result_ds = (
        GraphIngestor(run_mode="batch")
        .files(["/data/*.pdf"])
        .extract(ExtractParams(method="pdfium"))
        .embed(EmbedParams(model_name="nvidia/llama-nemotron-embed-1b-v2"))
        .ingest()
    )
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Union

from nemo_retriever.graph import InprocessExecutor, RayDataExecutor
from nemo_retriever.graph.ingestor_runtime import batch_tuning_to_node_overrides, build_graph
from nemo_retriever.utils.ray_resource_hueristics import gather_cluster_resources
from nemo_retriever.ingestor import ingestor
from nemo_retriever.params import (
    ASRParams,
    AudioChunkParams,
    CaptionParams,
    DedupParams,
    EmbedParams,
    ExtractParams,
    HtmlChunkParams,
    StoreParams,
    TextChunkParams,
    WebhookParams,
)
from nemo_retriever.utils.remote_auth import resolve_remote_api_key


def _resolve_api_key(params: Any) -> Any:
    """Auto-resolve api_key from NVIDIA_API_KEY / NGC_API_KEY if not explicitly set."""
    if params is None:
        return params
    if not getattr(params, "api_key", None) and hasattr(params, "model_copy"):
        key = resolve_remote_api_key()
        if key:
            return params.model_copy(update={"api_key": key})
    return params


def _coerce(params: Any, kwargs: dict[str, Any], *, default_factory: Callable[[], Any] | None = None) -> Any:
    """Merge keyword overrides into a params object and materialize defaults when requested."""
    if params is None:
        if default_factory is None:
            return kwargs or None
        params = default_factory()
        if not kwargs:
            return params
    if not kwargs:
        return params
    if hasattr(params, "model_copy"):
        return params.model_copy(update=kwargs)
    return params


class GraphIngestor(ingestor):
    """Ingestor that constructs and executes operator graphs directly.

    The fluent builder methods record pipeline stages. When :meth:`ingest` is
    called it builds a :class:`~nemo_retriever.graph.Graph` and feeds it to
    the appropriate executor.

    Parameters
    ----------
    run_mode
        ``"batch"`` (Ray Data, default) or ``"inprocess"`` (single-process
        pandas).
    ray_address
        Ray cluster address. ``None`` starts a local cluster.
    batch_size
        Default ``map_batches`` batch size for ``RayDataExecutor``.
    num_cpus
        Default CPU resources per operator node (batch mode).
    num_gpus
        Default GPU resources per operator node (batch mode).
    node_overrides
        Per-node resource/batching overrides forwarded to
        :class:`~nemo_retriever.graph.RayDataExecutor`.  Keys are node names
        (e.g. ``"OCRActor"``); values are dicts accepted by
        ``RayDataExecutor.__init__`` (``num_gpus``, ``batch_size``, etc.).
    show_progress
        Show a tqdm progress bar when running in inprocess mode.
    """

    RUN_MODE = "graph"

    def __init__(
        self,
        *,
        run_mode: str = "batch",
        documents: Optional[List[str]] = None,
        ray_address: Optional[str] = None,
        ray_log_to_driver: bool = True,
        debug: bool = False,
        allow_no_gpu: bool = False,
        batch_size: int = 1,
        num_cpus: float = 1,
        num_gpus: float = 0,
        node_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        show_progress: bool = True,
    ) -> None:
        super().__init__(documents=documents)
        if run_mode not in {"batch", "inprocess"}:
            raise ValueError(f"run_mode must be 'batch' or 'inprocess', got {run_mode!r}")
        self._run_mode = run_mode
        self._ray_address = ray_address
        self._ray_log_to_driver = ray_log_to_driver
        self._debug = debug
        self._allow_no_gpu = allow_no_gpu
        self._batch_size = batch_size
        self._num_cpus = num_cpus
        self._num_gpus = num_gpus
        self._node_overrides: Dict[str, Dict[str, Any]] = node_overrides or {}
        self._show_progress = show_progress
        self._rd_dataset: Any = None

        # Pipeline configuration accumulated by fluent methods
        self._extraction_mode: str = "pdf"
        self._extract_params: Any = None
        self._text_params: Any = None
        self._html_params: Any = None
        self._audio_chunk_params: Any = None
        self._asr_params: Any = None
        self._embed_params: Any = None
        self._split_params: Any = None
        self._caption_params: Any = None
        self._dedup_params: Any = None
        self._store_params: Any = None
        self._webhook_params: Any = None
        # Ordered list of stage names; "extract" is tracked but excluded from
        # the post-extraction stage_order passed to graph builders.
        self._stage_order: List[str] = []

    # ------------------------------------------------------------------
    # Input configuration
    # ------------------------------------------------------------------

    def files(self, documents: Union[str, List[str]]) -> "GraphIngestor":
        """Set the input file paths or glob patterns."""
        self._documents = [documents] if isinstance(documents, str) else list(documents)
        return self

    # ------------------------------------------------------------------
    # Extraction stage (sets extraction_mode and primary params)
    # ------------------------------------------------------------------

    def extract(self, params: Optional[ExtractParams] = None, **kwargs: Any) -> "GraphIngestor":
        """Configure PDF/document extraction (extraction_mode='pdf')."""
        self._extraction_mode = "pdf"
        self._extract_params = _resolve_api_key(_coerce(params, kwargs, default_factory=ExtractParams))
        self._record_stage("extract")
        return self

    def extract_image_files(self, params: Optional[ExtractParams] = None, **kwargs: Any) -> "GraphIngestor":
        """Configure image extraction (extraction_mode='image')."""
        self._extraction_mode = "image"
        self._extract_params = _resolve_api_key(_coerce(params, kwargs, default_factory=ExtractParams))
        self._record_stage("extract")
        return self

    def extract_txt(self, params: Optional[TextChunkParams] = None, **kwargs: Any) -> "GraphIngestor":
        """Configure plain-text extraction (extraction_mode='text')."""
        self._extraction_mode = "text"
        self._text_params = _coerce(params, kwargs, default_factory=TextChunkParams)
        self._record_stage("extract")
        return self

    def extract_html(self, params: Optional[HtmlChunkParams] = None, **kwargs: Any) -> "GraphIngestor":
        """Configure HTML extraction (extraction_mode='html')."""
        self._extraction_mode = "html"
        self._html_params = _coerce(params, kwargs, default_factory=HtmlChunkParams)
        self._record_stage("extract")
        return self

    def extract_audio(
        self,
        params: Optional[AudioChunkParams] = None,
        *,
        asr_params: Optional[ASRParams] = None,
        **kwargs: Any,
    ) -> "GraphIngestor":
        """Configure audio extraction (extraction_mode='audio')."""
        self._extraction_mode = "audio"
        self._audio_chunk_params = _coerce(params, kwargs, default_factory=AudioChunkParams)
        self._asr_params = asr_params or ASRParams()
        self._record_stage("extract")
        return self

    # ------------------------------------------------------------------
    # Post-extraction transform stages
    # ------------------------------------------------------------------

    def dedup(self, params: Optional[DedupParams] = None, **kwargs: Any) -> "GraphIngestor":
        """Record a dedup stage."""
        self._dedup_params = _coerce(params, kwargs, default_factory=DedupParams)
        self._record_stage("dedup")
        return self

    def caption(self, params: Optional[CaptionParams] = None, **kwargs: Any) -> "GraphIngestor":
        """Record a caption stage."""
        self._caption_params = _resolve_api_key(_coerce(params, kwargs, default_factory=CaptionParams))
        self._record_stage("caption")
        return self

    def split(self, params: Optional[TextChunkParams] = None, **kwargs: Any) -> "GraphIngestor":
        """Record a text-split stage."""
        self._split_params = _coerce(params, kwargs, default_factory=TextChunkParams)
        self._record_stage("split")
        return self

    def store(self, params: Optional[StoreParams] = None, **kwargs: Any) -> "GraphIngestor":
        """Record a store stage for persisting extracted images/text to storage."""
        self._store_params = _coerce(params, kwargs, default_factory=StoreParams)
        self._record_stage("store")
        return self

    def embed(self, params: Optional[EmbedParams] = None, **kwargs: Any) -> "GraphIngestor":
        """Record an embedding stage."""
        self._embed_params = _resolve_api_key(_coerce(params, kwargs, default_factory=EmbedParams))
        self._record_stage("embed")
        return self

    def webhook(self, params: Optional[WebhookParams] = None, **kwargs: Any) -> "GraphIngestor":
        """Record a webhook notification stage (always runs last).

        When ``endpoint_url`` is set, processed results are HTTP-POSTed to
        that URL.  If ``endpoint_url`` is ``None`` the stage is a no-op.
        """
        self._webhook_params = _coerce(params, kwargs, default_factory=WebhookParams)
        self._record_stage("webhook")
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def ingest(self, params: Any = None, **kwargs: Any) -> Any:
        """Build the operator graph and run it through the configured executor.

        Returns
        -------
        ``run_mode='batch'``
            A materialized ``ray.data.Dataset``.
        ``run_mode='inprocess'``
            A ``pandas.DataFrame``.
        """
        # Auto-enable dedup before captioning so that images overlapping
        # with table/chart/infographic detections are removed first.
        # Skip for image-only extraction — the image IS the content.
        if self._caption_params is not None and self._dedup_params is None and self._extraction_mode != "image":
            self._dedup_params = DedupParams()
            if "dedup" not in self._stage_order:
                # Insert dedup right before caption in the stage order.
                try:
                    idx = self._stage_order.index("caption")
                except ValueError:
                    idx = len(self._stage_order)
                self._stage_order.insert(idx, "dedup")

        post_extract_order = tuple(s for s in self._stage_order if s != "extract")

        if self._run_mode == "batch":
            import ray

            if self._ray_address or not ray.is_initialized():
                venv = os.path.dirname(os.path.dirname(sys.executable))
                venv_bin = os.path.join(venv, "bin")
                pypath = os.pathsep.join(p for p in sys.path if p)
                ray_env_vars: dict[str, str] = {
                    "VIRTUAL_ENV": venv,
                    "PATH": venv_bin + os.pathsep + os.environ.get("PATH", ""),
                    "PYTHONPATH": pypath,
                }
                for _fwd_key in ("HF_TOKEN", "HF_HOME", "HUGGING_FACE_HUB_TOKEN", "NVIDIA_API_KEY"):
                    if os.environ.get(_fwd_key):
                        ray_env_vars[_fwd_key] = os.environ[_fwd_key]
                runtime_env = {"env_vars": ray_env_vars}
                ray.init(
                    address=self._ray_address,
                    ignore_reinit_error=True,
                    runtime_env=runtime_env,
                )
            cluster_resources = gather_cluster_resources(ray)

            graph = build_graph(
                extraction_mode=self._extraction_mode,
                extract_params=self._extract_params,
                text_params=self._text_params,
                html_params=self._html_params,
                audio_chunk_params=self._audio_chunk_params,
                asr_params=self._asr_params,
                embed_params=self._embed_params,
                split_params=self._split_params,
                caption_params=self._caption_params,
                dedup_params=self._dedup_params,
                store_params=self._store_params,
                webhook_params=self._webhook_params,
                stage_order=post_extract_order,
            )
            # Derive per-node Ray scheduling config from BatchTuningParams plus
            # cluster-scaled heuristic defaults, then let any explicit
            # node_overrides passed to __init__ take precedence.
            effective_allow_no_gpu = self._allow_no_gpu or cluster_resources.available_gpu_count() == 0
            derived_overrides = batch_tuning_to_node_overrides(
                self._extract_params,
                self._embed_params,
                cluster_resources=cluster_resources,
                allow_no_gpu=effective_allow_no_gpu,
                caption_params=self._caption_params,
            )
            merged_overrides: Dict[str, Dict[str, Any]] = {}
            for node_name in set(derived_overrides) | set(self._node_overrides):
                merged_overrides[node_name] = {
                    **derived_overrides.get(node_name, {}),
                    **self._node_overrides.get(node_name, {}),
                }
            executor = RayDataExecutor(
                graph,
                ray_address=self._ray_address,
                batch_size=self._batch_size,
                num_cpus=self._num_cpus,
                num_gpus=self._num_gpus,
                node_overrides=merged_overrides,
            )
            result = executor.ingest(self._documents)
            self._rd_dataset = result
        else:
            graph = build_graph(
                extraction_mode=self._extraction_mode,
                extract_params=self._extract_params,
                text_params=self._text_params,
                html_params=self._html_params,
                audio_chunk_params=self._audio_chunk_params,
                asr_params=self._asr_params,
                embed_params=self._embed_params,
                split_params=self._split_params,
                caption_params=self._caption_params,
                dedup_params=self._dedup_params,
                store_params=self._store_params,
                webhook_params=self._webhook_params,
                stage_order=post_extract_order,
            )
            executor = InprocessExecutor(graph, show_progress=self._show_progress)
            self._rd_dataset = None
            result = executor.ingest(self._documents)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_error(v: Any) -> bool:
        def _is_populated_error_field(key: str, value: Any) -> bool:
            if value is None:
                return False
            if key == "failed" and isinstance(value, bool):
                return value
            if isinstance(value, str):
                return bool(value.strip())
            if isinstance(value, (list, tuple, set, dict)):
                return len(value) > 0
            return bool(value)

        if v is None:
            return False
        if isinstance(v, dict):
            for k in ("error", "errors", "exception", "traceback", "failed"):
                if k in v and _is_populated_error_field(k, v.get(k)):
                    return True
            return any(GraphIngestor._has_error(x) for x in v.values())
        if isinstance(v, list):
            return any(GraphIngestor._has_error(x) for x in v)
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return False
            if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                try:
                    return GraphIngestor._has_error(json.loads(s))
                except Exception:
                    pass
            low = s.lower()
            return any(tok in low for tok in ("error", "exception", "traceback", "failed"))
        return False

    @staticmethod
    def extract_error_rows(batch: Any) -> Any:
        if batch is None:
            return batch
        columns = getattr(batch, "columns", None)
        if columns is None:
            return batch
        error_candidate_columns = (
            "error",
            "errors",
            "exception",
            "traceback",
            "metadata",
            "source",
            "embedding",
        )
        cols = [c for c in error_candidate_columns if c in columns]
        if not cols:
            return batch.iloc[0:0]

        mask = batch[cols[0]].apply(GraphIngestor._has_error).astype(bool)
        for c in cols[1:]:
            mask = mask | batch[c].apply(GraphIngestor._has_error).astype(bool)
        return batch[mask]

    def get_error_rows(self, dataset: Any = None) -> Any:
        target = dataset if dataset is not None else self._rd_dataset
        if target is None:
            raise RuntimeError("No Ray Dataset available to inspect for errors.")
        return target.map_batches(self.extract_error_rows, batch_format="pandas")

    def get_dataset(self) -> Any:
        return self._rd_dataset

    def _record_stage(self, name: str) -> None:
        """Append *name* to the stage order list (deduplicated in place)."""
        if name not in self._stage_order:
            self._stage_order.append(name)
