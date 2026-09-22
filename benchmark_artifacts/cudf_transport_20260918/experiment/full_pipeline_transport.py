"""Experiment-scoped full NRL tails for narrow and cuDF-RDT transport.

The stock Ray Data graph is retained through extraction.  Its embedding and
global VDB stages are replaced with a streamed Ray Core tail so the benchmark
measures extraction, embedding, transport, LanceDB writes, and final index
construction without the stock one-block heterogeneous-dataframe repartition.
"""

from __future__ import annotations

import json
import math
import os
import queue
import threading
import time
from collections import deque
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from typing import Any

import ray

from cudf_cuda_ipc import register_cudf_cuda_ipc
from nemo_retriever.operators.abstract_operator import AbstractOperator


_MODE_ENV = "NRL_FULL_TRANSPORT_MODE"
_FLUSH_ROWS_ENV = "NRL_FULL_TRANSPORT_FLUSH_ROWS"
_STREAMING_ENV = "NRL_FULL_TRANSPORT_STREAMING"
_STREAMING_GPU_ENV = "NRL_FULL_TRANSPORT_STREAMING_GPU"
_EMBED_START_FRACTION_ENV = "NRL_EMBED_START_FRACTION"
_EXPECTED_PAGES_ENV = "NRL_EXPECTED_PAGES"
_DUTY_BATCHES_ENV = "NRL_EMBED_DUTY_BATCHES"
_DUTY_COOLDOWN_ENV = "NRL_EMBED_DUTY_COOLDOWN_SECONDS"
_INCREMENTAL_PDF_BATCH_ENV = "NRL_INCREMENTAL_PDF_PAGE_BATCH"
_TAIL_BATCH_SIZE_ENV = "NRL_EMBED_TAIL_BATCH_SIZE"
_INFERENCE_BATCH_SIZE_ENV = "NRL_EMBED_INFERENCE_BATCH_SIZE"
_BACKLOG_HIGH_ROWS_ENV = "NRL_EMBED_BACKLOG_HIGH_ROWS"
_BACKLOG_LOW_ROWS_ENV = "NRL_EMBED_BACKLOG_LOW_ROWS"
_BACKLOG_MAX_ROWS_ENV = "NRL_EMBED_BACKLOG_MAX_ROWS"
_MPS_ACTIVE_THREAD_PERCENTAGE_ENV = "NRL_EMBED_MPS_ACTIVE_THREAD_PERCENTAGE"
_CUSTOM_ARROW_DIRECT_ENV = "NRL_CUSTOM_ARROW_DIRECT"
_CUSTOM_RESULT_MODE_ENV = "NRL_CUSTOM_RESULT_MODE"
_PAGE_BLOCK_ROWS_ENV = "NRL_PAGE_ELEMENTS_BLOCK_ROWS"
_CUSTOM_MAX_INFLIGHT_ENV = "NRL_CUSTOM_MAX_INFLIGHT"
_CUSTOM_SOURCE_GPU_ENV = "NRL_CUSTOM_SOURCE_GPU"
_CUSTOM_SOURCE_REPLICAS_ENV = "NRL_CUSTOM_SOURCE_REPLICAS"
_CUSTOM_WRITER_GPU_ENV = "NRL_CUSTOM_WRITER_GPU"
_PAGE_MPS_ENV = "NRL_PAGE_ELEMENTS_MPS_PERCENTAGE"
_OCR_MPS_ENV = "NRL_OCR_MPS_PERCENTAGE"
_CUSTOM_SOURCE_MPS_ENV = "NRL_CUSTOM_SOURCE_MPS_PERCENTAGE"
_CUSTOM_WRITER_MPS_ENV = "NRL_CUSTOM_WRITER_MPS_PERCENTAGE"
_OCR_COST_BUDGET_ENV = "NRL_OCR_COST_BUDGET"
_INSTALLED = False

# Ray validates transport names when it evaluates @ray.method, so registration
# must precede the custom source class definition even for narrow-mode imports.
register_cudf_cuda_ipc()


class _IncrementalPDFSplitCPUActor:
    """Experiment-only PDF splitter that yields bounded page batches.

    The production splitter constructs every single-page PDF for an input batch
    before returning.  This callable preserves its row contract but yields a
    generator of small pandas frames, allowing Ray Data to apply downstream
    backpressure between page chunks.
    """

    def __init__(self, split_params: Any = None) -> None:
        from nemo_retriever.common.params import PdfSplitParams

        self.split_params = split_params or PdfSplitParams()
        self.page_batch_size = max(1, int(os.environ.get(_INCREMENTAL_PDF_BATCH_ENV, "16")))

    def __call__(self, pdf_batch: Any) -> Iterator[Any]:
        import pandas as pd

        from nemo_retriever.models.nim.error_reporter import report_error
        from nemo_retriever.operators.extract.pdf.split import _error_record, pdfium

        if not isinstance(pdf_batch, pd.DataFrame):
            raise NotImplementedError("incremental PDF split currently only supports pandas.DataFrame input")
        if pdfium is None:
            raise ImportError("pypdfium2 is required for incremental PDF splitting")

        explicit_cols = frozenset(("bytes", "path", "page_number", "metadata", "source_id"))
        page_rows: list[dict[str, Any]] = []
        for _, row in pdf_batch.iterrows():
            pdf_path = row["path"] if "path" in pdf_batch.columns else None
            pdf_bytes = row["bytes"] if "bytes" in pdf_batch.columns else None
            extra = {key: value for key, value in row.to_dict().items() if key not in explicit_cols}
            document = None
            try:
                if not isinstance(pdf_bytes, (bytes, bytearray, memoryview)):
                    raise ValueError(f"Unsupported bytes payload type: {type(pdf_bytes)!r}")
                try:
                    document = pdfium.PdfDocument(pdf_bytes)
                except Exception:
                    document = pdfium.PdfDocument(BytesIO(bytes(pdf_bytes)))

                start_idx = 0 if self.split_params.start_page is None else max(int(self.split_params.start_page) - 1, 0)
                end_idx = (
                    len(document) - 1
                    if self.split_params.end_page is None
                    else min(int(self.split_params.end_page) - 1, len(document) - 1)
                )
                for page_idx in range(start_idx, end_idx + 1):
                    single = pdfium.PdfDocument.new()
                    try:
                        single.import_pages(document, pages=[page_idx])
                        buffer = BytesIO()
                        single.save(buffer)
                        output = {
                            "bytes": buffer.getvalue(),
                            "path": pdf_path,
                            "page_number": page_idx + 1,
                            "metadata": {"source_path": pdf_path},
                            "source_id": f"{pdf_path}_{page_idx + 1}",
                        }
                        output.update(extra)
                        page_rows.append(output)
                    finally:
                        single.close()
                    if len(page_rows) >= self.page_batch_size:
                        yield pd.DataFrame(page_rows)
                        page_rows = []
            except BaseException as exc:
                report_error("pdf_split", exc)
                error = _error_record(
                    source_path=str(pdf_path) if pdf_path is not None else None,
                    stage="split_pdf",
                    exc=exc,
                    page_number=0,
                )
                error.update(extra)
                page_rows.append(error)
            finally:
                if document is not None:
                    try:
                        document.close()
                    except Exception:
                        pass

        if page_rows:
            yield pd.DataFrame(page_rows)


class _CostAwareOCRBatcher(AbstractOperator):
    """Yield order-preserving page groups bounded by detected-element cost."""

    PRESERVE_PANDAS_OUTPUT = True

    def __init__(self, cost_budget: int) -> None:
        super().__init__(cost_budget=cost_budget)
        if int(cost_budget) <= 0:
            raise ValueError("OCR cost budget must be positive")
        self.cost_budget = int(cost_budget)

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    @staticmethod
    def _row_cost(row: Any) -> int:
        value = row.get("page_elements_v3_num_detections")
        try:
            if value is not None and not math.isnan(float(value)):
                return max(1, int(value))
        except (TypeError, ValueError):
            pass
        detections = row.get("page_elements_v3")
        try:
            return max(1, len(detections))
        except TypeError:
            return 1

    def process(self, data: Any, **kwargs: Any) -> Iterator[Any]:
        import pandas as pd

        if not isinstance(data, pd.DataFrame):
            raise NotImplementedError("cost-aware OCR batching requires pandas.DataFrame input")
        start = 0
        running_cost = 0
        for position, (_, row) in enumerate(data.iterrows()):
            cost = self._row_cost(row)
            if position > start and running_cost + cost > self.cost_budget:
                yield data.iloc[start:position].copy()
                start = position
                running_cost = 0
            running_cost += cost
        if start < len(data.index):
            yield data.iloc[start:].copy()

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def __call__(self, data: Any, **kwargs: Any) -> Iterator[Any]:
        return self.run(data, **kwargs)


def _clean_vdb_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
    from nemo_retriever.common.vdb.sidecar_metadata import split_sidecar_from_vdb_kwargs

    kwargs, sidecar = split_sidecar_from_vdb_kwargs(dict(raw))
    if sidecar is not None:
        raise NotImplementedError("The full transport experiment does not support VDB sidecar metadata")
    if "uri" not in kwargs and kwargs.get("lancedb_uri"):
        kwargs["uri"] = kwargs.pop("lancedb_uri")
    kwargs["build_index"] = False
    return kwargs


class _StreamingLanceWriter:
    def __init__(self, vdb_kwargs: dict[str, Any]) -> None:
        from nemo_retriever.common.vdb.lancedb import LanceDB

        self.vdb = LanceDB(**_clean_vdb_kwargs(vdb_kwargs))
        self.pending: list[dict[str, Any]] = []
        self.flush_rows = int(os.environ.get(_FLUSH_ROWS_ENV, "2048"))
        self.rows = 0
        self.flushes = 0
        self.flush_seconds = 0.0
        self.flush_details: list[dict[str, Any]] = []
        self.started = time.perf_counter()

    def add(self, records: list[list[dict[str, Any]]]) -> int:
        incoming = [record for batch in records for record in batch]
        self.pending.extend(incoming)
        self.rows += len(incoming)
        if len(self.pending) >= self.flush_rows:
            self.flush()
        return len(incoming)

    def flush(self) -> None:
        if not self.pending:
            return
        rows = len(self.pending)
        started = time.perf_counter()
        self.vdb.run([self.pending])
        elapsed = time.perf_counter() - started
        self.flush_seconds += elapsed
        self.flush_details.append({"rows": rows, "seconds": elapsed})
        self.pending = []
        self.flushes += 1
        # The first flush creates the table. Later flushes append to it.
        self.vdb.overwrite = False

    def finalize(self, *, mode: str) -> dict[str, Any]:
        import lancedb

        self.flush()
        table = lancedb.connect(self.vdb.uri).open_table(self.vdb.table_name)
        index_started = time.perf_counter()
        self.vdb.write_to_index(
            records=None,
            table=table,
            index_type=self.vdb.index_type,
            metric=self.vdb.metric,
            num_partitions=self.vdb.num_partitions,
            num_sub_vectors=self.vdb.num_sub_vectors,
            hybrid=self.vdb.hybrid,
            sparse=self.vdb.sparse,
            fts_language=self.vdb.fts_language,
        )
        index_seconds = time.perf_counter() - index_started
        result = {
            "mode": mode,
            "rows": self.rows,
            "flushes": self.flushes,
            "elapsed_seconds": time.perf_counter() - self.started,
            "flush_seconds": self.flush_seconds,
            "flush_details": self.flush_details,
            "index_seconds": index_seconds,
            "lancedb_uri": self.vdb.uri,
            "table_name": self.vdb.table_name,
            "indexes": [index.name for index in table.list_indices()],
        }
        stats_path = Path(self.vdb.uri).parent / "full_transport_stats.json"
        stats_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result


class _ArrowStreamingLanceWriter:
    """Append canonical Arrow tables without materializing vector Python lists."""

    def __init__(self, vdb_kwargs: dict[str, Any]) -> None:
        from nemo_retriever.common.vdb.lancedb import LanceDB

        self.vdb = LanceDB(**_clean_vdb_kwargs(vdb_kwargs))
        self.pending: list[Any] = []
        self.pending_rows = 0
        self.flush_rows = int(os.environ.get(_FLUSH_ROWS_ENV, "2048"))
        self.rows = 0
        self.flushes = 0
        self.flush_seconds = 0.0
        self.flush_details: list[dict[str, Any]] = []
        self.started = time.perf_counter()

    def add_arrow(self, table: Any) -> int:
        rows = int(table.num_rows)
        if rows:
            self.pending.append(table)
            self.pending_rows += rows
            self.rows += rows
        if self.pending_rows >= self.flush_rows:
            self.flush()
        return rows

    def flush(self) -> None:
        if not self.pending:
            return
        import lancedb
        import pyarrow as pa

        from nemo_retriever.common.vdb.lancedb import _lancedb_arrow_schema

        rows = self.pending_rows
        started = time.perf_counter()
        data = pa.concat_tables(self.pending, promote_options="default")
        schema = _lancedb_arrow_schema(
            int(self.vdb.vector_dim or 2048),
            retrieval_mode="hybrid" if self.vdb.hybrid else "dense",
            embedding_model_name=self.vdb.embedding_model_name,
            embedding_model_revision=self.vdb.embedding_model_revision,
        )
        data = data.cast(schema)
        db = lancedb.connect(self.vdb.uri)
        write_kwargs: dict[str, Any] = {"on_bad_vectors": self.vdb.on_bad_vectors}
        if self.vdb.on_bad_vectors == "fill":
            write_kwargs["fill_value"] = self.vdb.fill_value
        if self.vdb.overwrite:
            db.create_table(
                self.vdb.table_name,
                data=data,
                schema=schema,
                mode="overwrite",
                **write_kwargs,
            )
            self.vdb.overwrite = False
        else:
            db.open_table(self.vdb.table_name).add(data, mode="append", **write_kwargs)
        elapsed = time.perf_counter() - started
        self.flush_seconds += elapsed
        self.flush_details.append({"rows": rows, "seconds": elapsed})
        self.pending = []
        self.pending_rows = 0
        self.flushes += 1

    def finalize(self, *, mode: str) -> dict[str, Any]:
        import lancedb

        self.flush()
        table = lancedb.connect(self.vdb.uri).open_table(self.vdb.table_name)
        index_started = time.perf_counter()
        self.vdb.write_to_index(
            records=None,
            table=table,
            index_type=self.vdb.index_type,
            metric=self.vdb.metric,
            num_partitions=self.vdb.num_partitions,
            num_sub_vectors=self.vdb.num_sub_vectors,
            hybrid=self.vdb.hybrid,
            sparse=self.vdb.sparse,
            fts_language=self.vdb.fts_language,
        )
        index_seconds = time.perf_counter() - index_started
        result = {
            "mode": mode,
            "rows": self.rows,
            "flushes": self.flushes,
            "elapsed_seconds": time.perf_counter() - self.started,
            "flush_seconds": self.flush_seconds,
            "flush_details": self.flush_details,
            "index_seconds": index_seconds,
            "lancedb_uri": self.vdb.uri,
            "table_name": self.vdb.table_name,
            "indexes": [index.name for index in table.list_indices()],
        }
        stats_path = Path(self.vdb.uri).parent / "full_transport_stats.json"
        stats_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result


class _NarrowEmbedWriter:
    def __init__(self, embed_params: Any, vdb_kwargs: dict[str, Any]) -> None:
        from nemo_retriever.operators.embed.gpu_operator import _BatchEmbedActor

        started = time.perf_counter()
        self.embedder = _BatchEmbedActor(embed_params)
        self.model_init_seconds = time.perf_counter() - started
        self.writer = _StreamingLanceWriter(vdb_kwargs)
        self.embed_seconds = 0.0
        self.record_build_seconds = 0.0
        self.process_calls = 0

    def process(self, batch: Any) -> int:
        from nemo_retriever.common.vdb.records import to_client_vdb_records

        started = time.perf_counter()
        embedded = self.embedder.run(batch)
        self.embed_seconds += time.perf_counter() - started
        started = time.perf_counter()
        records = to_client_vdb_records(embedded)
        self.record_build_seconds += time.perf_counter() - started
        self.process_calls += 1
        return self.writer.add(records)

    def finalize(self) -> dict[str, Any]:
        result = self.writer.finalize(mode="narrow_gpu_island")
        result["timings"] = {
            "model_init_seconds": self.model_init_seconds,
            "embed_seconds": self.embed_seconds,
            "record_build_seconds": self.record_build_seconds,
            "process_calls": self.process_calls,
        }
        return result


class _CustomEmbedSource:
    def __init__(self, embed_params: Any) -> None:
        from nemo_retriever.operators.embed.gpu_operator import _BatchEmbedActor

        started = time.perf_counter()
        self.embedder = _BatchEmbedActor(embed_params)
        self.model_init_seconds = time.perf_counter() - started
        self.expected_dimensions = int(getattr(embed_params, "dimensions", None) or 2048)
        self.sidecars: dict[int, tuple[list[list[dict[str, Any]]], float]] = {}
        self.results: dict[int, Any] = {}
        self.embed_seconds = 0.0
        self.record_build_seconds = 0.0
        self.cudf_pack_seconds = 0.0
        self.process_calls = 0

    @ray.method(tensor_transport="CUDF_CUDA_IPC")
    def embed_vectors(self, batch: Any, batch_id: int):
        import cudf
        import numpy as np
        import pyarrow as pa

        from nemo_retriever.common.vdb.records import to_client_vdb_records

        started = time.perf_counter()
        embedded = self.embedder.run(batch)
        self.embed_seconds += time.perf_counter() - started
        if os.environ.get(_CUSTOM_RESULT_MODE_ENV, "sink_only").strip().lower() == "historical":
            self.results[batch_id] = embedded
        started = time.perf_counter()
        records = to_client_vdb_records(embedded)
        flat = [record for record_batch in records for record in record_batch]
        vectors = []
        kept_records = []
        for record in flat:
            metadata = record.get("metadata")
            if not isinstance(metadata, dict) or metadata.get("embedding") is None:
                raise ValueError("Canonical VDB record is missing metadata.embedding")
            vector = np.asarray(metadata.get("embedding"), dtype=np.float32).reshape(-1)
            # Match LanceDB's benchmark configuration: on_bad_vectors="drop".
            # Embedding failures can leave an empty vector on an otherwise
            # canonical record, and those rows must not enter the cuDF frame.
            if vector.size != self.expected_dimensions:
                continue
            metadata.pop("embedding")
            vectors.append(vector)
            kept_records.append(record)
        filtered_records = [kept_records] if kept_records else []
        self.record_build_seconds += time.perf_counter() - started

        if not vectors:
            frame = cudf.DataFrame({"row_id": np.empty((0,), dtype=np.int64)})
        else:
            started = time.perf_counter()
            values = np.asarray(vectors, dtype=np.float32)
            rows, dimensions = values.shape
            offsets = pa.array(range(0, (rows + 1) * dimensions, dimensions), type=pa.int64())
            vector_lists = pa.ListArray.from_arrays(offsets, pa.array(values.reshape(-1)))
            table = pa.table(
                {
                    "row_id": pa.array(np.arange(rows, dtype=np.int64)),
                    "vector": vector_lists,
                }
            )
            frame = cudf.DataFrame.from_arrow(table)
            self.cudf_pack_seconds += time.perf_counter() - started
        self.process_calls += 1
        self.sidecars[batch_id] = (filtered_records, time.perf_counter())
        return frame

    def take_sidecar(self, batch_id: int) -> tuple[list[list[dict[str, Any]]], float]:
        return self.sidecars.pop(batch_id)

    def take_result(self, batch_id: int) -> Any:
        return self.results.pop(batch_id)

    def stats(self) -> dict[str, Any]:
        return {
            "model_init_seconds": self.model_init_seconds,
            "embed_seconds": self.embed_seconds,
            "record_build_seconds": self.record_build_seconds,
            "cudf_pack_seconds": self.cudf_pack_seconds,
            "process_calls": self.process_calls,
        }


class _CustomCudfWriter:
    def __init__(self, vdb_kwargs: dict[str, Any]) -> None:
        self.arrow_direct = os.environ.get(_CUSTOM_ARROW_DIRECT_ENV, "").strip().lower() in {"1", "true", "yes"}
        self.writer = _ArrowStreamingLanceWriter(vdb_kwargs) if self.arrow_direct else _StreamingLanceWriter(vdb_kwargs)
        self.cudf_to_host_seconds = 0.0
        self.record_merge_seconds = 0.0
        self.arrow_build_seconds = 0.0
        self.handoff_queue_seconds = 0.0
        self.process_calls = 0

    def write(
        self,
        vectors: Any,
        sidecar: tuple[list[list[dict[str, Any]]], float],
    ) -> int:
        records, ready_at = sidecar
        self.handoff_queue_seconds += max(0.0, time.perf_counter() - ready_at)
        self.process_calls += 1
        if "vector" not in vectors.columns:
            return 0
        flat = [record for record_batch in records for record in record_batch]
        if self.arrow_direct:
            import pyarrow as pa

            from nemo_retriever.common.vdb.lancedb import _get_text_for_element, _json_str

            started = time.perf_counter()
            vector_values = vectors["vector"].to_arrow()
            if isinstance(vector_values, pa.ChunkedArray):
                vector_values = vector_values.combine_chunks()
            if pa.types.is_list(vector_values.type) or pa.types.is_large_list(vector_values.type):
                dimensions = int(len(vector_values.values) / max(1, len(vector_values)))
                vector_values = pa.FixedSizeListArray.from_arrays(vector_values.values, dimensions)
            self.cudf_to_host_seconds += time.perf_counter() - started
            if len(vector_values) != len(flat):
                raise ValueError(f"vector/sidecar mismatch: {len(vector_values)} != {len(flat)}")
            started = time.perf_counter()
            texts: list[str] = []
            metadata_values: list[str] = []
            source_values: list[str] = []
            ids: list[str] = []
            for record in flat:
                metadata = record.get("metadata", {})
                content_meta = metadata.get("content_metadata", {}) if isinstance(metadata, dict) else {}
                if not isinstance(content_meta, dict):
                    content_meta = {}
                source_meta = metadata.get("source_metadata", {}) if isinstance(metadata, dict) else {}
                row_id = content_meta.get("id")
                if row_id is None and isinstance(metadata, dict):
                    row_id = metadata.get("id")
                texts.append(_get_text_for_element(record))
                metadata_values.append(_json_str(content_meta))
                source_values.append(_json_str(source_meta))
                ids.append(str(row_id) if row_id is not None else "")
            table = pa.Table.from_arrays(
                [
                    vector_values,
                    pa.array(texts, type=pa.string()),
                    pa.array(metadata_values, type=pa.string()),
                    pa.array(source_values, type=pa.string()),
                    pa.array(ids, type=pa.string()),
                ],
                names=["vector", "text", "metadata", "source", "id"],
            )
            self.arrow_build_seconds += time.perf_counter() - started
            return self.writer.add_arrow(table)
        started = time.perf_counter()
        vector_values = vectors["vector"].to_arrow().to_pylist()
        self.cudf_to_host_seconds += time.perf_counter() - started
        started = time.perf_counter()
        if len(vector_values) != len(flat):
            raise ValueError(f"vector/sidecar mismatch: {len(vector_values)} != {len(flat)}")
        for record, vector in zip(flat, vector_values, strict=True):
            record["metadata"]["embedding"] = vector
        self.record_merge_seconds += time.perf_counter() - started
        return self.writer.add(records)

    def finalize(self) -> dict[str, Any]:
        result = self.writer.finalize(
            mode="custom_cudf_cuda_ipc_arrow" if self.arrow_direct else "custom_cudf_cuda_ipc"
        )
        result["writer_timings"] = {
            "cudf_to_host_seconds": self.cudf_to_host_seconds,
            "record_merge_seconds": self.record_merge_seconds,
            "arrow_build_seconds": self.arrow_build_seconds,
            "handoff_queue_seconds": self.handoff_queue_seconds,
            "process_calls": self.process_calls,
        }
        return result


def _prefix_graph(nodes: list[Any], stop: int):
    from nemo_retriever.graph.pipeline_graph import Graph, Node

    prefix = Graph()
    incremental_pdf = int(os.environ.get(_INCREMENTAL_PDF_BATCH_ENV, "0")) > 0
    ocr_cost_budget = int(os.environ.get(_OCR_COST_BUDGET_ENV, "0"))
    if ocr_cost_budget < 0:
        raise ValueError(f"{_OCR_COST_BUDGET_ENV} must be non-negative")
    copies = []
    for node in nodes[:stop]:
        if ocr_cost_budget and node.name == "OCRActor":
            batcher = _CostAwareOCRBatcher(ocr_cost_budget)
            copies.append(
                Node(
                    batcher,
                    name="CostAwareOCRBatcher",
                    operator_class=_CostAwareOCRBatcher,
                    operator_kwargs={"cost_budget": ocr_cost_budget},
                )
            )
        operator_class = node.operator_class
        if incremental_pdf and node.name == "PDFSplitCPUActor":
            operator_class = _IncrementalPDFSplitCPUActor
        copies.append(
            Node(
                node.operator,
                name=node.name,
                operator_class=operator_class,
                operator_kwargs=dict(node.operator_kwargs),
            )
        )
    prefix.add_chain(*copies)
    return prefix


def _batch_page_keys(batch: Any) -> set[str]:
    """Return stable page identities from an extracted pandas batch."""
    if "source_id" in batch.columns:
        return {str(value) for value in batch["source_id"].dropna().tolist()}
    if "path" in batch.columns and "page_number" in batch.columns:
        return {
            f"{path}_{page_number}"
            for path, page_number in zip(batch["path"].tolist(), batch["page_number"].tolist(), strict=True)
        }
    if "metadata" in batch.columns:
        keys = set()
        for metadata in batch["metadata"].tolist():
            if isinstance(metadata, dict):
                source = metadata.get("source_path") or metadata.get("source_id")
                page = metadata.get("page_number")
                if source is not None:
                    keys.add(f"{source}_{page}" if page is not None else str(source))
        return keys
    return set()


def _batch_memory_bytes(batch: Any) -> int:
    try:
        return int(batch.memory_usage(index=True, deep=True).sum())
    except Exception:
        return 0


def _find_tail(nodes: list[Any]) -> tuple[int, int]:
    embed_index = next((i for i, node in enumerate(nodes) if node.name == "_BatchEmbedActor"), -1)
    vdb_index = next((i for i, node in enumerate(nodes) if node.name == "IngestVdbOperator"), -1)
    if embed_index < 0 or vdb_index < 0 or vdb_index <= embed_index:
        raise RuntimeError("Full transport mode requires a graph ending in embed followed by VDB upload")
    if vdb_index != len(nodes) - 1:
        raise RuntimeError("Full transport experiment does not support stages after VDB upload")
    return embed_index, vdb_index


def _install_executor_patch() -> None:
    from nemo_retriever.graph.executor import RayDataExecutor

    original_ingest = RayDataExecutor.ingest

    def transport_ingest(self, data: Any, **kwargs: Any):
        executor_started = time.perf_counter()
        mode = os.environ.get(_MODE_ENV, "").strip().lower()
        if mode not in {"narrow", "custom"}:
            return original_ingest(self, data, **kwargs)

        import pandas as pd

        nodes = self._linearize(self.graph)
        embed_index, vdb_index = _find_tail(nodes)
        embed_node = nodes[embed_index]
        vdb_node = nodes[vdb_index]
        embed_params = embed_node.operator_kwargs["params"]
        inference_batch_override = int(os.environ.get(_INFERENCE_BATCH_SIZE_ENV, "0"))
        if inference_batch_override < 0:
            raise ValueError(f"{_INFERENCE_BATCH_SIZE_ENV} must be non-negative")
        if inference_batch_override:
            embed_params = embed_params.model_copy(update={"inference_batch_size": inference_batch_override})
        vdb_kwargs = dict(vdb_node.operator_kwargs.get("vdb_kwargs") or {})

        prefix = _prefix_graph(nodes, embed_index)
        prefix_overrides = {key: value for key, value in self._node_overrides.items() if key != embed_node.name}
        ocr_cost_budget = int(os.environ.get(_OCR_COST_BUDGET_ENV, "0"))
        if ocr_cost_budget:
            prefix_overrides["CostAwareOCRBatcher"] = {
                "batch_size": None,
                "concurrency": 1,
                "num_cpus": 0.1,
            }
            prefix_overrides.setdefault("OCRActor", {})["batch_size"] = None
        page_block_rows = int(os.environ.get(_PAGE_BLOCK_ROWS_ENV, "0"))
        if page_block_rows < 0:
            raise ValueError(f"{_PAGE_BLOCK_ROWS_ENV} must be non-negative")
        if page_block_rows:
            prefix_overrides.setdefault("PageElementDetectionActor", {})["target_num_rows_per_block"] = page_block_rows
        for node_name, env_name in (
            ("PageElementDetectionActor", _PAGE_MPS_ENV),
            ("OCRActor", _OCR_MPS_ENV),
        ):
            percentage = int(os.environ.get(env_name, "0"))
            if not 0 <= percentage <= 100:
                raise ValueError(f"{env_name} must be between 0 and 100")
            if percentage:
                prefix_overrides.setdefault(node_name, {})["runtime_env"] = {
                    "env_vars": {"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(percentage)}
                }
        prefix_executor = RayDataExecutor(
            prefix,
            ray_address=self._ray_address,
            batch_size=self._default_batch_size,
            batch_format=self._default_batch_format,
            num_cpus=self._default_num_cpus,
            num_gpus=self._default_num_gpus,
            node_overrides=prefix_overrides,
            auto_concurrency_nodes=self._auto_concurrency_nodes - {embed_node.name},
            source_cpu_reservation=self._source_cpu_reservation,
        )
        dataset_build_started = time.perf_counter()
        if ocr_cost_budget:
            import ray.data as rd

            original_map_batches = rd.Dataset.map_batches

            def cost_block_map_batches(dataset: Any, fn: Any, *args: Any, **map_kwargs: Any):
                whole_gpu_block = map_kwargs.get("batch_size") is None and float(map_kwargs.get("num_gpus") or 0) > 0
                if whole_gpu_block:
                    private_defaults = {
                        "compute": None,
                        "batch_format": "default",
                        "zero_copy_batch": True,
                        "fn_args": None,
                        "fn_kwargs": None,
                        "fn_constructor_args": None,
                        "fn_constructor_kwargs": None,
                        "num_cpus": None,
                        "num_gpus": None,
                        "memory": None,
                        "concurrency": None,
                        "udf_modifying_row_count": True,
                        "ray_remote_args_fn": None,
                    }
                    for key, value in private_defaults.items():
                        map_kwargs.setdefault(key, value)
                    return dataset._map_batches_without_batch_size_validation(
                        fn,
                        *args,
                        **map_kwargs,
                    )
                return original_map_batches(dataset, fn, *args, **map_kwargs)

            rd.Dataset.map_batches = cost_block_map_batches
            try:
                extracted = prefix_executor.build_dataset(data, **kwargs)
            finally:
                rd.Dataset.map_batches = original_map_batches
        else:
            extracted = prefix_executor.build_dataset(data, **kwargs)
        dataset_build_seconds = time.perf_counter() - dataset_build_started
        streaming = os.environ.get(_STREAMING_ENV, "").strip().lower() in {"1", "true", "yes"}
        materialize_seconds = 0.0
        if not streaming:
            materialize_started = time.perf_counter()
            extracted = extracted.materialize()
            materialize_seconds = time.perf_counter() - materialize_started

        configured_batch_size = int(self._node_overrides.get(embed_node.name, {}).get("batch_size") or 32)
        batch_size = int(os.environ.get(_TAIL_BATCH_SIZE_ENV, "0")) or configured_batch_size
        if batch_size <= 0:
            raise ValueError(f"{_TAIL_BATCH_SIZE_ENV} must resolve to a positive integer")
        inference_batch_size = int(getattr(embed_params, "inference_batch_size", 0) or 0)
        streaming_gpu = float(os.environ.get(_STREAMING_GPU_ENV, "0.4"))
        pending: list[Any] = []
        stored_rows = 0
        batch_count = 0
        first_batch_at: float | None = None
        last_batch_at: float | None = None
        iteration_started = time.perf_counter()
        source_stats: dict[str, Any] | None = None
        admission_stats: dict[str, Any] = {}
        retained_results: list[Any] = []

        if mode == "narrow":
            mps_percentage = int(os.environ.get(_MPS_ACTIVE_THREAD_PERCENTAGE_ENV, "0"))
            if not 0 <= mps_percentage <= 100:
                raise ValueError(f"{_MPS_ACTIVE_THREAD_PERCENTAGE_ENV} must be between 0 and 100")
            actor_options: dict[str, Any] = {
                "num_gpus": streaming_gpu if streaming else 1,
                "num_cpus": 0 if streaming else 1,
            }
            if mps_percentage:
                actor_options["runtime_env"] = {"env_vars": {"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(mps_percentage)}}
            actor_type = ray.remote(
                **actor_options,
            )(_NarrowEmbedWriter)
            start_fraction = float(os.environ.get(_EMBED_START_FRACTION_ENV, "0"))
            if not 0.0 <= start_fraction <= 1.0:
                raise ValueError(f"{_EMBED_START_FRACTION_ENV} must be between 0 and 1")
            expected_pages = int(os.environ.get(_EXPECTED_PAGES_ENV, "0"))
            if start_fraction > 0 and expected_pages <= 0:
                raise ValueError(f"{_EXPECTED_PAGES_ENV} must be positive when delayed admission is enabled")
            duty_batches = int(os.environ.get(_DUTY_BATCHES_ENV, "0"))
            duty_cooldown = float(os.environ.get(_DUTY_COOLDOWN_ENV, "0"))
            if duty_batches < 0 or duty_cooldown < 0:
                raise ValueError("duty-cycle batch count and cooldown must be non-negative")
            backlog_high_rows = int(os.environ.get(_BACKLOG_HIGH_ROWS_ENV, "0"))
            backlog_low_rows = int(os.environ.get(_BACKLOG_LOW_ROWS_ENV, "0"))
            backlog_max_rows = int(os.environ.get(_BACKLOG_MAX_ROWS_ENV, "0"))
            if backlog_high_rows < 0 or backlog_low_rows < 0 or backlog_max_rows < 0:
                raise ValueError("backlog watermarks must be non-negative")
            if backlog_high_rows:
                if start_fraction:
                    raise ValueError("adaptive backlog and fractional delayed admission are mutually exclusive")
                if not 0 <= backlog_low_rows < backlog_high_rows:
                    raise ValueError("adaptive backlog requires 0 <= low rows < high rows")
                backlog_max_rows = backlog_max_rows or backlog_high_rows * 2
                if backlog_max_rows < backlog_high_rows:
                    raise ValueError("adaptive backlog max rows must be at least the high watermark")

            tail = actor_type.remote(embed_params, vdb_kwargs) if start_fraction == 0 else None
            buffered: list[tuple[Any, int, int]] = []
            pages_seen: set[str] = set()
            buffered_bytes = 0
            buffered_rows = 0
            max_buffered_bytes = 0
            max_buffered_rows = 0
            max_buffered_batches = 0
            admission_started_at = iteration_started if tail is not None else None
            dispatches_since_pause = 0
            duty_pause_seconds = 0.0
            duty_pauses = 0

            def collect_pending(*, all_pending: bool = False) -> None:
                nonlocal pending, stored_rows
                if not pending:
                    return
                if all_pending:
                    stored_rows += sum(int(value) for value in ray.get(pending))
                    pending = []
                    return
                ready, pending = ray.wait(pending, num_returns=1)
                stored_rows += int(ray.get(ready[0]))

            def submit(batch_or_ref: Any) -> None:
                nonlocal dispatches_since_pause, duty_pause_seconds, duty_pauses
                if tail is None:
                    raise RuntimeError("embedding tail has not been admitted")
                pending.append(tail.process.remote(batch_or_ref))
                dispatches_since_pause += 1
                if len(pending) >= 4:
                    collect_pending()
                if duty_batches and duty_cooldown and dispatches_since_pause >= duty_batches:
                    collect_pending(all_pending=True)
                    pause_started = time.perf_counter()
                    time.sleep(duty_cooldown)
                    duty_pause_seconds += time.perf_counter() - pause_started
                    duty_pauses += 1
                    dispatches_since_pause = 0

            def admit_tail() -> None:
                nonlocal tail, admission_started_at, buffered, buffered_bytes, buffered_rows
                if tail is not None:
                    return
                admission_started_at = time.perf_counter()
                tail = actor_type.remote(embed_params, vdb_kwargs)
                for batch_ref, _rows, _bytes in buffered:
                    submit(batch_ref)
                buffered = []
                buffered_bytes = 0
                buffered_rows = 0

            source_batches = extracted.iter_batches(batch_size=batch_size, batch_format="pandas", prefetch_batches=2)
            adaptive_refills = 0
            adaptive_wait_seconds = 0.0
            if backlog_high_rows:
                sentinel = object()
                max_batches = max(1, math.ceil(backlog_max_rows / batch_size))
                source_queue: queue.Queue[Any] = queue.Queue(maxsize=max_batches)
                producer_error: list[BaseException] = []

                def produce_batches() -> None:
                    nonlocal first_batch_at, last_batch_at, batch_count
                    try:
                        for source_batch in source_batches:
                            now = time.perf_counter()
                            first_batch_at = first_batch_at or now
                            last_batch_at = now
                            batch_count += 1
                            rows = len(source_batch)
                            memory_bytes = _batch_memory_bytes(source_batch)
                            source_queue.put((ray.put(source_batch), rows, memory_bytes))
                    except BaseException as exc:
                        producer_error.append(exc)
                    finally:
                        source_queue.put(sentinel)

                producer = threading.Thread(target=produce_batches, name="nrl-extraction-producer", daemon=True)
                producer.start()
                adaptive_buffer: deque[tuple[Any, int, int]] = deque()
                adaptive_rows = 0
                adaptive_bytes = 0
                producer_done = False

                def take_source(*, block: bool) -> bool:
                    nonlocal adaptive_rows, adaptive_bytes, producer_done
                    try:
                        item = source_queue.get() if block else source_queue.get_nowait()
                    except queue.Empty:
                        return False
                    if item is sentinel:
                        producer_done = True
                        return False
                    batch_ref, rows, memory_bytes = item
                    adaptive_buffer.append((batch_ref, rows, memory_bytes))
                    adaptive_rows += rows
                    adaptive_bytes += memory_bytes
                    return True

                def refill_to(target_rows: int) -> None:
                    nonlocal adaptive_refills, adaptive_wait_seconds
                    if producer_done or adaptive_rows >= target_rows:
                        return
                    adaptive_refills += 1
                    wait_started = time.perf_counter()
                    while adaptive_rows < target_rows and not producer_done:
                        take_source(block=True)
                    adaptive_wait_seconds += time.perf_counter() - wait_started

                refill_to(backlog_high_rows)
                admission_started_at = time.perf_counter()
                while adaptive_buffer or not producer_done:
                    if adaptive_rows < backlog_low_rows and not producer_done:
                        refill_to(backlog_high_rows)
                    while adaptive_rows < backlog_max_rows and not producer_done and take_source(block=False):
                        pass
                    max_buffered_rows = max(max_buffered_rows, adaptive_rows)
                    max_buffered_bytes = max(max_buffered_bytes, adaptive_bytes)
                    max_buffered_batches = max(max_buffered_batches, len(adaptive_buffer))
                    if not adaptive_buffer:
                        refill_to(backlog_high_rows)
                        continue
                    batch_ref, rows, memory_bytes = adaptive_buffer.popleft()
                    adaptive_rows -= rows
                    adaptive_bytes -= memory_bytes
                    submit(batch_ref)
                producer.join()
                if producer_error:
                    raise producer_error[0]
            else:
                for batch in source_batches:
                    now = time.perf_counter()
                    first_batch_at = first_batch_at or now
                    last_batch_at = now
                    batch_count += 1
                    if tail is None:
                        pages_seen.update(_batch_page_keys(batch))
                        rows = len(batch)
                        memory_bytes = _batch_memory_bytes(batch)
                        buffered.append((ray.put(batch), rows, memory_bytes))
                        buffered_rows += rows
                        buffered_bytes += memory_bytes
                        max_buffered_rows = max(max_buffered_rows, buffered_rows)
                        max_buffered_bytes = max(max_buffered_bytes, buffered_bytes)
                        max_buffered_batches = max(max_buffered_batches, len(buffered))
                        if len(pages_seen) >= expected_pages * start_fraction:
                            admit_tail()
                    else:
                        submit(batch)
                admit_tail()
            collect_pending(all_pending=True)
            drain_finished = time.perf_counter()
            finalize_started = time.perf_counter()
            assert tail is not None
            final = ray.get(tail.finalize.remote())
            admission_stats = {
                "start_fraction": start_fraction,
                "expected_pages": expected_pages or None,
                "unique_pages_seen_at_start": len(pages_seen),
                "admission_delay_seconds": (admission_started_at or iteration_started) - iteration_started,
                "max_buffered_batches": max_buffered_batches,
                "max_buffered_rows": max_buffered_rows,
                "max_buffered_bytes_estimate": max_buffered_bytes,
                "duty_batches": duty_batches,
                "duty_cooldown_seconds": duty_cooldown,
                "duty_pauses": duty_pauses,
                "duty_pause_seconds": duty_pause_seconds,
                "incremental_pdf_page_batch": int(os.environ.get(_INCREMENTAL_PDF_BATCH_ENV, "0")),
                "backlog_low_rows": backlog_low_rows,
                "backlog_high_rows": backlog_high_rows,
                "backlog_max_rows": backlog_max_rows,
                "backlog_refills": adaptive_refills,
                "backlog_wait_seconds": adaptive_wait_seconds,
                "mps_active_thread_percentage": mps_percentage,
            }
        else:
            register_cudf_cuda_ipc()
            result_mode = os.environ.get(_CUSTOM_RESULT_MODE_ENV, "sink_only").strip().lower()
            if result_mode not in {"historical", "sink_only"}:
                raise ValueError(f"{_CUSTOM_RESULT_MODE_ENV} must be 'historical' or 'sink_only'")
            backlog_high_rows = int(os.environ.get(_BACKLOG_HIGH_ROWS_ENV, "0"))
            backlog_low_rows = int(os.environ.get(_BACKLOG_LOW_ROWS_ENV, "0"))
            backlog_max_rows = int(os.environ.get(_BACKLOG_MAX_ROWS_ENV, "0"))
            if backlog_high_rows < 0 or backlog_low_rows < 0 or backlog_max_rows < 0:
                raise ValueError("custom backlog watermarks must be non-negative")
            if backlog_high_rows:
                if not 0 <= backlog_low_rows < backlog_high_rows:
                    raise ValueError("custom backlog requires 0 <= low rows < high rows")
                backlog_max_rows = backlog_max_rows or backlog_high_rows * 2
                if backlog_max_rows < backlog_high_rows:
                    raise ValueError("custom backlog max rows must be at least the high watermark")
            source_gpu = float(
                os.environ.get(
                    _CUSTOM_SOURCE_GPU_ENV,
                    str(streaming_gpu * 0.875 if streaming else 0.9),
                )
            )
            writer_gpu = float(
                os.environ.get(
                    _CUSTOM_WRITER_GPU_ENV,
                    str(streaming_gpu - source_gpu if streaming else 0.1),
                )
            )
            if source_gpu <= 0 or writer_gpu <= 0:
                raise ValueError("custom source and writer GPU reservations must be positive")
            source_replicas = int(os.environ.get(_CUSTOM_SOURCE_REPLICAS_ENV, "1"))
            if source_replicas <= 0:
                raise ValueError(f"{_CUSTOM_SOURCE_REPLICAS_ENV} must be positive")
            if streaming and source_gpu + writer_gpu > streaming_gpu + 1e-9:
                raise ValueError("custom source and writer GPU reservations exceed the streaming GPU budget")
            source_mps = int(os.environ.get(_CUSTOM_SOURCE_MPS_ENV, "0"))
            writer_mps = int(os.environ.get(_CUSTOM_WRITER_MPS_ENV, "0"))
            if not 0 <= source_mps <= 100 or not 0 <= writer_mps <= 100:
                raise ValueError("custom source and writer MPS percentages must be between 0 and 100")
            source_options: dict[str, Any] = {
                "num_gpus": source_gpu / source_replicas,
                "num_cpus": 0 if streaming else 1,
                "enable_tensor_transport": True,
            }
            writer_options: dict[str, Any] = {
                "num_gpus": writer_gpu,
                "num_cpus": 0 if streaming else 1,
                "enable_tensor_transport": True,
            }
            if source_mps:
                source_options["runtime_env"] = {"env_vars": {"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(source_mps)}}
            if writer_mps:
                writer_options["runtime_env"] = {"env_vars": {"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(writer_mps)}}
            source_type = ray.remote(
                **source_options,
            )(_CustomEmbedSource)
            writer_type = ray.remote(
                **writer_options,
            )(_CustomCudfWriter)
            sources = [source_type.remote(embed_params) for _ in range(source_replicas)]
            writer = writer_type.remote(vdb_kwargs)
            source_batches = extracted.iter_batches(batch_size=batch_size, batch_format="pandas", prefetch_batches=2)
            source_queue: queue.Queue[Any] | None = None
            producer: threading.Thread | None = None
            producer_error: list[BaseException] = []
            sentinel = object()
            if backlog_high_rows:
                max_batches = max(1, math.ceil(backlog_max_rows / batch_size))
                source_queue = queue.Queue(maxsize=max_batches)

                def produce_custom_batches() -> None:
                    try:
                        for source_batch in source_batches:
                            source_queue.put(source_batch)
                    except BaseException as exc:
                        producer_error.append(exc)
                    finally:
                        source_queue.put(sentinel)

                producer = threading.Thread(
                    target=produce_custom_batches,
                    name="nrl-custom-extraction-producer",
                    daemon=True,
                )
                producer.start()

                def admitted_batches() -> Iterator[Any]:
                    buffered: deque[Any] = deque()
                    buffered_rows = 0
                    producer_done = False
                    max_rows = 0
                    max_batches_seen = 0
                    refills = 0
                    wait_seconds = 0.0
                    while buffered or not producer_done:
                        if buffered_rows <= backlog_low_rows and not producer_done:
                            refills += 1
                            wait_started = time.perf_counter()
                            while buffered_rows < backlog_high_rows and not producer_done:
                                item = source_queue.get()
                                if item is sentinel:
                                    producer_done = True
                                    break
                                buffered.append(item)
                                buffered_rows += len(item)
                                max_rows = max(max_rows, buffered_rows)
                                max_batches_seen = max(max_batches_seen, len(buffered))
                            wait_seconds += time.perf_counter() - wait_started
                        if buffered:
                            item = buffered.popleft()
                            buffered_rows -= len(item)
                            yield item
                    admission_stats.update(
                        {
                            "backlog_low_rows": backlog_low_rows,
                            "backlog_high_rows": backlog_high_rows,
                            "backlog_max_rows": backlog_max_rows,
                            "backlog_refills": refills,
                            "backlog_wait_seconds": wait_seconds,
                            "max_buffered_rows": max_rows,
                            "max_buffered_batches": max_batches_seen,
                        }
                    )

                batches = admitted_batches()
            else:
                batches = source_batches

            max_inflight = int(os.environ.get(_CUSTOM_MAX_INFLIGHT_ENV, "4"))
            if max_inflight <= 0:
                raise ValueError(f"{_CUSTOM_MAX_INFLIGHT_ENV} must be positive")
            result_refs: list[Any] = []
            pending_wait_seconds = 0.0
            max_pending = 0
            for batch_id, batch in enumerate(batches):
                now = time.perf_counter()
                first_batch_at = first_batch_at or now
                last_batch_at = now
                batch_count += 1
                source = sources[batch_id % source_replicas]
                vector_ref = source.embed_vectors.remote(batch, batch_id)
                sidecar_ref = source.take_sidecar.remote(batch_id)
                if result_mode == "historical":
                    result_refs.append(source.take_result.remote(batch_id))
                pending.append(writer.write.remote(vector_ref, sidecar_ref))
                max_pending = max(max_pending, len(pending))
                if len(pending) >= max_inflight:
                    wait_started = time.perf_counter()
                    ready, pending = ray.wait(pending, num_returns=1)
                    stored_rows += int(ray.get(ready[0]))
                    pending_wait_seconds += time.perf_counter() - wait_started
            if pending:
                stored_rows += sum(int(value) for value in ray.get(pending))
            if producer is not None:
                producer.join()
            if producer_error:
                raise producer_error[0]
            if result_refs:
                retained_results = list(ray.get(result_refs))
            admission_stats.update(
                {
                    "max_inflight_batches": max_inflight,
                    "max_pending_batches": max_pending,
                    "pending_wait_seconds": pending_wait_seconds,
                    "source_gpu": source_gpu,
                    "source_gpu_per_replica": source_gpu / source_replicas,
                    "source_replicas": source_replicas,
                    "writer_gpu": writer_gpu,
                    "source_mps_percentage": source_mps,
                    "writer_mps_percentage": writer_mps,
                    "page_elements_block_rows": page_block_rows,
                }
            )
            drain_finished = time.perf_counter()
            per_replica_stats = ray.get([source.stats.remote() for source in sources])
            source_stats = {
                "replicas": source_replicas,
                "per_replica": per_replica_stats,
                "model_init_seconds": max(
                    (float(stats["model_init_seconds"]) for stats in per_replica_stats),
                    default=0.0,
                ),
                "embed_seconds": sum(float(stats["embed_seconds"]) for stats in per_replica_stats),
                "record_build_seconds": sum(float(stats["record_build_seconds"]) for stats in per_replica_stats),
                "cudf_pack_seconds": sum(float(stats["cudf_pack_seconds"]) for stats in per_replica_stats),
                "process_calls": sum(int(stats["process_calls"]) for stats in per_replica_stats),
            }
            finalize_started = time.perf_counter()
            final = ray.get(writer.finalize.remote())

        if int(final["rows"]) != stored_rows:
            raise RuntimeError(f"streamed row accounting mismatch: {final['rows']} != {stored_rows}")
        finalized_at = time.perf_counter()
        final["streaming"] = streaming
        final["driver_timings"] = {
            "executor_total_seconds": finalized_at - executor_started,
            "dataset_build_seconds": dataset_build_seconds,
            "materialize_seconds": materialize_seconds,
            "time_to_first_batch_seconds": (first_batch_at - iteration_started if first_batch_at is not None else None),
            "input_stream_seconds": (last_batch_at - iteration_started if last_batch_at is not None else None),
            "tail_drain_seconds": drain_finished - (last_batch_at or iteration_started),
            "finalize_seconds": finalized_at - finalize_started,
            "batches": batch_count,
            "batch_size": batch_size,
            "inference_batch_size": inference_batch_size,
            "streaming_gpu_fraction": streaming_gpu if streaming else 1.0,
        }
        if source_stats is not None:
            final["source_timings"] = source_stats
        if admission_stats:
            final["admission"] = admission_stats
        final["result_mode"] = os.environ.get(_CUSTOM_RESULT_MODE_ENV, "sink_only") if mode == "custom" else None
        stats_path = Path(final["lancedb_uri"]).parent / "full_transport_stats.json"
        stats_path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if retained_results:
            return pd.concat(retained_results, ignore_index=True)
        return pd.DataFrame({"stored": [True] * stored_rows})

    RayDataExecutor.ingest = transport_ingest


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    mode = os.environ.get(_MODE_ENV, "").strip().lower()
    if mode not in {"narrow", "custom"}:
        raise ValueError(f"{_MODE_ENV} must be 'narrow' or 'custom'; got {mode!r}")
    _install_executor_patch()
    _INSTALLED = True
