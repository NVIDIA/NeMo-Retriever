# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from importlib import import_module
import logging
from pathlib import Path
import sys
import time
from typing import Any, Optional, TextIO

from nemo_retriever.ingest_modes.lancedb_utils import lancedb_schema
from nemo_retriever.model import resolve_embed_model
from nemo_retriever.recall.beir import BeirConfig, evaluate_lancedb_beir
from nemo_retriever.recall.core import RecallConfig, retrieve_and_score

DEFAULT_LANCEDB_URI = "lancedb"
DEFAULT_LANCEDB_TABLE = "nv-ingest"

logger = logging.getLogger(__name__)


class _TeeStream:
    """Write stream output to the terminal and an optional file."""

    def __init__(self, primary: TextIO, mirror: TextIO) -> None:
        self._primary = primary
        self._mirror = mirror

    def write(self, data: str) -> int:
        self._primary.write(data)
        self._mirror.write(data)
        return len(data)

    def flush(self) -> None:
        self._primary.flush()
        self._mirror.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._primary, "isatty", lambda: False)())

    def fileno(self) -> int:
        return int(getattr(self._primary, "fileno")())

    def writable(self) -> bool:
        return bool(getattr(self._primary, "writable", lambda: True)())

    @property
    def encoding(self) -> str:
        return str(getattr(self._primary, "encoding", "utf-8"))


def configure_cli_logging(log_file: Optional[Path], *, debug: bool = False) -> tuple[Optional[TextIO], TextIO, TextIO]:
    """Configure root logging and optionally tee stdout/stderr to a file."""

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_level = logging.DEBUG if debug else logging.INFO
    if log_file is None:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            force=True,
        )
        return None, original_stdout, original_stderr

    target = Path(log_file).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = open(target, "a", encoding="utf-8", buffering=1)

    sys.stdout = _TeeStream(sys.__stdout__, handle)
    sys.stderr = _TeeStream(sys.__stderr__, handle)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logging.getLogger(__name__).info("Writing combined pipeline logs to %s", str(target))
    return handle, original_stdout, original_stderr


def restore_cli_logging(
    handle: Optional[TextIO],
    original_stdout: TextIO,
    original_stderr: TextIO,
) -> None:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    if handle is not None:
        try:
            handle.flush()
        finally:
            handle.close()


def lancedb_module() -> Any:
    return import_module("lancedb")


def ensure_lancedb_table(uri: str, table_name: str) -> None:
    """Ensure the LanceDB URI exists and the target table can be opened."""

    Path(uri).mkdir(parents=True, exist_ok=True)
    db = lancedb_module().connect(uri)
    try:
        db.open_table(table_name)
        return
    except Exception:
        pass

    import pyarrow as pa  # type: ignore

    schema = lancedb_schema()
    empty = pa.table({field.name: [] for field in schema}, schema=schema)
    db.create_table(table_name, data=empty, schema=schema, mode="create")


def open_lancedb_table_with_retry(
    uri: str,
    table_name: str,
    *,
    retries: int = 3,
    sleep_seconds: float = 2.0,
) -> Any:
    db = lancedb_module().connect(uri)
    open_err: Exception | None = None
    for _ in range(max(1, retries)):
        try:
            return db.open_table(table_name)
        except Exception as exc:
            open_err = exc
            ensure_lancedb_table(uri, table_name)
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Could not open LanceDB table {table_name!r} at {uri!r}") from open_err


def count_lancedb_rows(uri: str, table_name: str) -> int | None:
    try:
        table = open_lancedb_table_with_retry(uri, table_name, retries=1, sleep_seconds=0.0)
        return int(table.count_rows())
    except Exception:
        return None


def estimate_processed_pages(uri: str, table_name: str) -> int | None:
    """Estimate processed pages from unique `(source_id, page_number)` pairs."""

    try:
        table = open_lancedb_table_with_retry(uri, table_name, retries=1, sleep_seconds=0.0)
        df = table.to_pandas()[["source_id", "page_number"]]
        return int(df.dropna(subset=["source_id", "page_number"]).drop_duplicates().shape[0])
    except Exception:
        return count_lancedb_rows(uri, table_name)


def safe_pdf_page_count(path: Path) -> int | None:
    try:
        import pypdfium2 as pdfium  # type: ignore

        doc = pdfium.PdfDocument(str(path))
        try:
            try:
                count = int(len(doc))
            except Exception:
                count = int(doc.get_page_count())  # type: ignore[attr-defined]
        finally:
            try:
                doc.close()
            except Exception:
                pass
        return max(count, 0)
    except Exception:
        return None


def resolve_input_pages(input_type: str, input_files: list[Path]) -> int | None:
    if input_type == "image":
        return len(input_files)
    if input_type != "pdf":
        return None

    total_pages = 0
    counted_any = False
    for path in input_files:
        page_count = safe_pdf_page_count(path)
        if page_count is None:
            continue
        counted_any = True
        total_pages += page_count
    if counted_any:
        return total_pages
    return None


def evaluate_lancedb_metrics(
    evaluation_cfg,
    *,
    lancedb_uri: str,
    lancedb_table: str,
    embed_model_name: str,
    embed_invoke_url: str | None,
    embed_api_key: str | None,
    hybrid: bool,
):
    from .reports import EvaluationSummary

    if evaluation_cfg.evaluation_mode not in {"recall", "beir"}:
        raise ValueError(f"Unsupported evaluation mode: {evaluation_cfg.evaluation_mode}")

    table = open_lancedb_table_with_retry(lancedb_uri, lancedb_table)
    try:
        if int(table.count_rows()) == 0:
            logger.warning(
                "LanceDB table %r exists but is empty; skipping %s evaluation.",
                lancedb_table,
                evaluation_cfg.evaluation_mode,
            )
            return EvaluationSummary(label="BEIR" if evaluation_cfg.evaluation_mode == "beir" else "Recall"), 0.0
    except Exception:
        pass

    resolved_model = resolve_embed_model(embed_model_name)
    evaluation_label = "BEIR" if evaluation_cfg.evaluation_mode == "beir" else "Recall"

    if evaluation_cfg.evaluation_mode == "beir":
        if not evaluation_cfg.beir_loader:
            raise ValueError("--beir-loader is required when --evaluation-mode=beir")
        if not evaluation_cfg.beir_dataset_name:
            raise ValueError("--beir-dataset-name is required when --evaluation-mode=beir")

        beir_cfg = BeirConfig(
            lancedb_uri=lancedb_uri,
            lancedb_table=lancedb_table,
            embedding_model=resolved_model,
            loader=str(evaluation_cfg.beir_loader),
            dataset_name=str(evaluation_cfg.beir_dataset_name),
            split=str(evaluation_cfg.beir_split),
            query_language=evaluation_cfg.beir_query_language,
            doc_id_field=str(evaluation_cfg.beir_doc_id_field),
            ks=tuple(evaluation_cfg.beir_ks) if evaluation_cfg.beir_ks else (1, 3, 5, 10),
            embedding_http_endpoint=embed_invoke_url,
            embedding_api_key=(embed_api_key or "").strip(),
            hybrid=hybrid,
            reranker=bool(evaluation_cfg.reranker),
            reranker_model_name=str(evaluation_cfg.reranker_model_name),
        )
        evaluation_start = time.perf_counter()
        beir_dataset, _raw_hits, _run, metrics = evaluate_lancedb_beir(beir_cfg)
        return (
            EvaluationSummary(label=evaluation_label, query_count=len(beir_dataset.query_ids), metrics=metrics),
            time.perf_counter() - evaluation_start,
        )

    query_csv = Path(str(evaluation_cfg.query_csv)).expanduser() if evaluation_cfg.query_csv else None
    if query_csv is None or not query_csv.exists():
        logger.warning("Query CSV not found at %s; skipping recall evaluation.", query_csv)
        return EvaluationSummary(label=evaluation_label), 0.0

    recall_cfg = RecallConfig(
        lancedb_uri=lancedb_uri,
        lancedb_table=lancedb_table,
        embedding_model=resolved_model,
        embedding_http_endpoint=embed_invoke_url,
        embedding_api_key=(embed_api_key or "").strip(),
        top_k=10,
        ks=(1, 5, 10),
        hybrid=hybrid,
        match_mode=str(evaluation_cfg.recall_match_mode),
        reranker=str(evaluation_cfg.reranker_model_name) if evaluation_cfg.reranker else None,
    )
    evaluation_start = time.perf_counter()
    df_query, _gold, _raw_hits, _retrieved_keys, metrics = retrieve_and_score(query_csv=query_csv, cfg=recall_cfg)
    return (
        EvaluationSummary(label=evaluation_label, query_count=len(df_query.index), metrics=metrics),
        time.perf_counter() - evaluation_start,
    )
