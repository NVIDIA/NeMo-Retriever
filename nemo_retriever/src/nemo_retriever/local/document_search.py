# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent-friendly local document search workflow.

This module intentionally keeps the public CLI narrow.  It wraps the existing
graph ingestion and Retriever query surfaces with a small manifest, safety
checks, and JSON output that agents can inspect without scraping prose logs.
"""

from __future__ import annotations

import fnmatch
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import typer

from nemo_retriever.model import resolve_embed_model
from nemo_retriever.utils.remote_auth import resolve_remote_api_key

app = typer.Typer(
    help=(
        "Small, project-local document search: init a local index, search it, "
        "inspect status, and diagnose setup issues."
    )
)

DEFAULT_INDEX = Path(".nemo-retriever/local-index")
DEFAULT_LANCEDB_SUBDIR = "lancedb"
DEFAULT_LANCEDB_TABLE = "local-documents"
DEFAULT_MAX_DOCS = 200
DEFAULT_TOP_K = 5
DEFAULT_EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-1b-v2"
DEFAULT_REMOTE_EMBED_ENDPOINT = "https://integrate.api.nvidia.com/v1/embeddings"
MANIFEST_FILENAME = "manifest.json"

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "pdf",
    ".txt": "txt",
    ".md": "txt",
    ".markdown": "txt",
    ".docx": "doc",
    ".pptx": "doc",
}

SUPPORTED_FORMATS = sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS)


@dataclass(frozen=True)
class LocalDocument:
    path: Path
    relative_path: str
    input_type: str
    extension: str
    size_bytes: int
    mtime_ns: int
    page_count: int | None = None

    def to_manifest(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "relative_path": self.relative_path,
            "input_type": self.input_type,
            "extension": self.extension,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "page_count": self.page_count,
        }


@dataclass(frozen=True)
class Discovery:
    input_path: Path
    corpus_root: Path
    documents: list[LocalDocument]
    skipped: list[dict[str, Any]]
    warnings: list[str]
    include: list[str]
    exclude: list[str]
    max_docs: int
    max_pages: int | None


@dataclass(frozen=True)
class InferenceConfig:
    requested: str
    endpoint_mode: str
    embed_invoke_url: str | None
    api_key_configured: bool


class LocalSearchError(RuntimeError):
    """Expected local-search failure with an agent-readable payload."""

    def __init__(self, message: str, *, code: str = "local_search_error", warnings: list[str] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.warnings = warnings or []


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_index(index: Path) -> Path:
    return Path(index).expanduser().resolve()


def _lancedb_uri(index: Path) -> Path:
    return _resolve_index(index) / DEFAULT_LANCEDB_SUBDIR


def _manifest_path(index: Path) -> Path:
    return _resolve_index(index) / MANIFEST_FILENAME


def _normalize_output(output: str) -> str:
    value = (output or "text").strip().lower()
    if value not in {"json", "text"}:
        raise typer.BadParameter("--output must be one of: json, text")
    return value


def _normalize_inference(inference: str) -> str:
    value = (inference or "auto").strip().lower()
    if value not in {"auto", "remote", "local"}:
        raise typer.BadParameter("--inference must be one of: auto, remote, local")
    return value


def _echo_payload(
    payload: dict[str, Any],
    *,
    output: str,
    text_formatter: Callable[[dict[str, Any]], str],
) -> None:
    if _normalize_output(output) == "json":
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(text_formatter(payload))


def _error_payload(command: str, exc: Exception) -> dict[str, Any]:
    if isinstance(exc, LocalSearchError):
        return {
            "command": command,
            "ok": False,
            "error": {
                "code": exc.code,
                "message": str(exc),
            },
            "warnings": exc.warnings,
        }
    return {
        "command": command,
        "ok": False,
        "error": {
            "code": type(exc).__name__,
            "message": str(exc),
        },
        "warnings": [],
    }


def _fail(command: str, exc: Exception, *, output: str) -> None:
    payload = _error_payload(command, exc)
    _echo_payload(payload, output=output, text_formatter=lambda p: f"ERROR: {p['error']['message']}")
    raise typer.Exit(code=1)


def _relative_name(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _matches(patterns: Iterable[str], path: Path, root: Path) -> bool:
    pattern_list = [p for p in patterns if p]
    if not pattern_list:
        return True

    relative = _relative_name(path, root)
    candidates = {relative, path.name, relative.lower(), path.name.lower()}
    return any(fnmatch.fnmatch(candidate, pattern) for pattern in pattern_list for candidate in candidates)


def _count_pdf_pages(path: Path) -> int | None:
    try:
        import pypdfium2  # type: ignore
    except Exception:
        return None

    try:
        pdf = pypdfium2.PdfDocument(str(path))
    except Exception:
        return None
    try:
        return int(len(pdf))
    finally:
        close = getattr(pdf, "close", None)
        if callable(close):
            close()


def _discover_documents(
    input_path: Path,
    *,
    include: list[str] | None,
    exclude: list[str] | None,
    max_docs: int,
    max_pages: int | None,
) -> Discovery:
    source = Path(input_path).expanduser().resolve()
    if not source.exists():
        raise LocalSearchError(f"Input path does not exist: {source}", code="input_path_not_found")
    if max_docs < 1:
        raise LocalSearchError("--max-docs must be at least 1", code="invalid_max_docs")
    if max_pages is not None and max_pages < 1:
        raise LocalSearchError("--max-pages must be at least 1 when provided", code="invalid_max_pages")

    corpus_root = source.parent if source.is_file() else source
    candidates = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file())

    include_patterns = include or []
    exclude_patterns = exclude or []
    documents: list[LocalDocument] = []
    skipped: list[dict[str, Any]] = []
    warnings: list[str] = []
    estimated_pages = 0
    page_count_unavailable = False

    for path in candidates:
        rel = _relative_name(path, corpus_root)
        if include_patterns and not _matches(include_patterns, path, corpus_root):
            skipped.append({"path": str(path), "relative_path": rel, "reason": "include_filter"})
            continue
        if exclude_patterns and _matches(exclude_patterns, path, corpus_root):
            skipped.append({"path": str(path), "relative_path": rel, "reason": "exclude_filter"})
            continue

        extension = path.suffix.lower()
        input_type = SUPPORTED_EXTENSIONS.get(extension)
        if input_type is None:
            skipped.append(
                {
                    "path": str(path),
                    "relative_path": rel,
                    "reason": "unsupported_format",
                    "extension": extension,
                }
            )
            continue

        page_count = _count_pdf_pages(path) if extension == ".pdf" and max_pages is not None else None
        if max_pages is not None:
            if extension == ".pdf" and page_count is None:
                page_count_unavailable = True
            increment = page_count if page_count is not None else 1
            if estimated_pages + increment > max_pages:
                skipped.append(
                    {
                        "path": str(path),
                        "relative_path": rel,
                        "reason": "max_pages_exceeded",
                        "page_count": page_count,
                    }
                )
                continue
            estimated_pages += increment

        stat = path.stat()
        documents.append(
            LocalDocument(
                path=path,
                relative_path=rel,
                input_type=input_type,
                extension=extension,
                size_bytes=int(stat.st_size),
                mtime_ns=int(stat.st_mtime_ns),
                page_count=page_count,
            )
        )

    if len(documents) > max_docs:
        overflow = documents[max_docs:]
        documents = documents[:max_docs]
        skipped.extend(
            {
                "path": str(doc.path),
                "relative_path": doc.relative_path,
                "reason": "max_docs_exceeded",
            }
            for doc in overflow
        )

    if page_count_unavailable:
        warnings.append(
            "Some PDF page counts could not be read; --max-pages was enforced with a conservative one-page estimate "
            "for those files."
        )
    if not documents:
        warnings.append(
            "No supported documents were selected. Supported extensions: " + ", ".join(SUPPORTED_FORMATS) + "."
        )

    return Discovery(
        input_path=source,
        corpus_root=corpus_root,
        documents=documents,
        skipped=skipped,
        warnings=warnings,
        include=include_patterns,
        exclude=exclude_patterns,
        max_docs=max_docs,
        max_pages=max_pages,
    )


def _group_documents(documents: list[LocalDocument]) -> dict[str, list[LocalDocument]]:
    grouped: dict[str, list[LocalDocument]] = {}
    for doc in documents:
        grouped.setdefault(doc.input_type, []).append(doc)
    return grouped


def _resolve_inference_config(
    *,
    inference: str,
    embed_invoke_url: str | None,
    api_key: str | None,
) -> InferenceConfig:
    requested = _normalize_inference(inference)
    endpoint = (embed_invoke_url or "").strip() or None
    resolved_key = resolve_remote_api_key(api_key)
    if requested == "remote" and endpoint is None:
        endpoint = DEFAULT_REMOTE_EMBED_ENDPOINT
    endpoint_mode = "remote" if endpoint else "local"
    if requested == "local":
        endpoint = None
        endpoint_mode = "local"

    if endpoint and "integrate.api.nvidia.com" in endpoint and resolved_key is None:
        raise LocalSearchError(
            "Remote hosted inference requires --api-key or NVIDIA_API_KEY/NGC_API_KEY.",
            code="missing_remote_api_key",
        )
    return InferenceConfig(
        requested=requested,
        endpoint_mode=endpoint_mode,
        embed_invoke_url=endpoint,
        api_key_configured=resolved_key is not None,
    )


def _validate_local_inference_available() -> None:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:
        raise LocalSearchError(
            "Local inference requires the local model stack. Install `nemo-retriever[local]` or use "
            "`--inference remote` with NVIDIA_API_KEY.",
            code="local_inference_unavailable",
        ) from exc

    if not bool(torch.cuda.is_available()):
        raise LocalSearchError(
            "Local inference requires a CUDA GPU in this workflow. Use `--inference remote` for a CPU-only machine.",
            code="local_cuda_unavailable",
        )

    try:
        import transformers  # noqa: F401
    except ModuleNotFoundError as exc:
        raise LocalSearchError(
            "Local inference requires `transformers`. Install `nemo-retriever[local]` or use `--inference remote`.",
            code="local_inference_unavailable",
        ) from exc


def _manifest_base(
    *,
    discovery: Discovery,
    index_path: Path,
    inference_config: InferenceConfig,
    embedding_model: str,
    warnings: list[str],
) -> dict[str, Any]:
    docs = [doc.to_manifest() for doc in discovery.documents]
    return {
        "schema_version": 1,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "last_query_at": None,
        "index_path": str(_resolve_index(index_path)),
        "lancedb_uri": str(_lancedb_uri(index_path)),
        "lancedb_table": DEFAULT_LANCEDB_TABLE,
        "input_path": str(discovery.input_path),
        "corpus_root": str(discovery.corpus_root),
        "include": discovery.include,
        "exclude": discovery.exclude,
        "max_docs": discovery.max_docs,
        "max_pages": discovery.max_pages,
        "documents": docs,
        "documents_discovered": len(docs) + len(discovery.skipped),
        "documents_processed": 0,
        "documents_skipped": discovery.skipped,
        "chunk_count": 0,
        "embedding_model": resolve_embed_model(embedding_model),
        "inference": {
            "requested": inference_config.requested,
            "endpoint_mode": inference_config.endpoint_mode,
            "embed_invoke_url": inference_config.embed_invoke_url,
            "api_key_configured": inference_config.api_key_configured,
        },
        "warnings": warnings,
    }


def _load_manifest(index: Path) -> dict[str, Any]:
    path = _manifest_path(index)
    if not path.exists():
        raise LocalSearchError(
            f"No local-search manifest found at {path}. Run `retriever local init` first.",
            code="index_not_initialized",
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LocalSearchError(f"Manifest is not valid JSON: {path}", code="manifest_invalid") from exc
    if not isinstance(raw, dict):
        raise LocalSearchError(f"Manifest must contain a JSON object: {path}", code="manifest_invalid")
    return raw


def _write_manifest(index: Path, manifest: dict[str, Any]) -> None:
    index_path = _resolve_index(index)
    index_path.mkdir(parents=True, exist_ok=True)
    _lancedb_uri(index).mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = _utc_now()
    _manifest_path(index).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reset_lancedb_dir(index: Path) -> None:
    uri = _lancedb_uri(index)
    if uri.exists():
        shutil.rmtree(uri)
    uri.mkdir(parents=True, exist_ok=True)


def _table_info(index: Path, table_name: str) -> dict[str, Any]:
    uri = _lancedb_uri(index)
    info: dict[str, Any] = {
        "readable": False,
        "uri": str(uri),
        "table": table_name,
        "table_exists": False,
        "row_count": 0,
        "error": None,
    }
    try:
        import lancedb  # type: ignore

        db = lancedb.connect(str(uri))
        names = set(db.table_names())
        info["readable"] = True
        info["table_exists"] = table_name in names
        if info["table_exists"]:
            table = db.open_table(table_name)
            info["row_count"] = int(table.count_rows())
            schema = getattr(table, "schema", None)
            if schema is not None:
                info["schema"] = [field.name for field in schema]
                try:
                    vector_field = schema.field("vector")
                    info["vector_dimensions"] = getattr(vector_field.type, "list_size", None)
                except Exception:
                    info["vector_dimensions"] = None
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def _detect_staleness(manifest: dict[str, Any]) -> dict[str, Any]:
    changed: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for doc in manifest.get("documents", []):
        path = Path(str(doc.get("path", "")))
        if not path.exists():
            missing.append({"path": str(path), "reason": "missing"})
            continue
        stat = path.stat()
        if int(doc.get("size_bytes", -1)) != int(stat.st_size) or int(doc.get("mtime_ns", -1)) != int(stat.st_mtime_ns):
            changed.append({"path": str(path), "reason": "modified"})

    new_supported = 0
    corpus_root = Path(str(manifest.get("corpus_root") or ""))
    if corpus_root.exists() and corpus_root.is_dir():
        indexed = {str(Path(str(doc.get("path", ""))).resolve()) for doc in manifest.get("documents", [])}
        for path in corpus_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS and str(path.resolve()) not in indexed:
                new_supported += 1

    return {
        "stale": bool(changed or missing or new_supported),
        "changed_documents": changed,
        "missing_documents": missing,
        "new_supported_documents": new_supported,
    }


def _health_checks(manifest: dict[str, Any], table_info: dict[str, Any], stale: dict[str, Any]) -> list[dict[str, Any]]:
    checks = [
        {"name": "manifest_present", "status": "ok", "message": "Manifest is readable."},
        {
            "name": "lancedb_readable",
            "status": "ok" if table_info["readable"] else "error",
            "message": "LanceDB is readable." if table_info["readable"] else str(table_info.get("error")),
        },
        {
            "name": "table_present",
            "status": "ok" if table_info["table_exists"] else "error",
            "message": (
                f"Table {table_info['table']!r} is present."
                if table_info["table_exists"]
                else f"Table {table_info['table']!r} was not found."
            ),
        },
        {
            "name": "index_not_empty",
            "status": "ok" if int(table_info.get("row_count") or 0) > 0 else "warning",
            "message": f"{int(table_info.get('row_count') or 0)} chunks are indexed.",
        },
        {
            "name": "corpus_fresh",
            "status": "ok" if not stale["stale"] else "warning",
            "message": "Indexed files match the manifest." if not stale["stale"] else "Indexed files have changed.",
        },
    ]
    if manifest.get("schema_version") != 1:
        checks.append(
            {
                "name": "schema_version",
                "status": "warning",
                "message": f"Unexpected manifest schema_version={manifest.get('schema_version')!r}.",
            }
        )
    return checks


def _warning_messages(manifest: dict[str, Any], stale: dict[str, Any], table_info: dict[str, Any]) -> list[str]:
    warnings = list(manifest.get("warnings") or [])
    if int(table_info.get("row_count") or 0) == 0:
        warnings.append("The local index is empty; run `retriever local init` and check skipped documents.")
    if stale["stale"]:
        warnings.append("The corpus appears stale; re-run `retriever local init` before relying on results.")
    return warnings


def _run_ingestion(
    discovery: Discovery,
    *,
    index_path: Path,
    inference_config: InferenceConfig,
    embedding_model: str,
    api_key: str | None,
    text_chunk_max_tokens: int,
    text_chunk_overlap_tokens: int,
) -> dict[str, Any]:
    from nemo_retriever.params import TextChunkParams
    from nemo_retriever.pipeline.__main__ import (
        _build_embed_params,
        _build_extract_params,
        _build_ingestor,
        _collect_results,
        _count_uploadable_vdb_records,
        _upload_vdb_records,
    )

    grouped = _group_documents(discovery.documents)
    group_summaries: list[dict[str, Any]] = []
    processed_paths: set[str] = set()
    total_rows = 0
    total_uploadable = 0
    first_upload = True
    resolved_api_key = resolve_remote_api_key(api_key)
    if inference_config.endpoint_mode == "local":
        _validate_local_inference_available()
    _lancedb_uri(index_path).mkdir(parents=True, exist_ok=True)

    for input_type, docs in sorted(grouped.items()):
        extract_params = _build_extract_params(
            method="pdfium",
            dpi=300,
            extract_text=True,
            extract_tables=False,
            extract_charts=False,
            extract_infographics=False,
            extract_page_as_image=False,
            use_graphic_elements=False,
            use_table_structure=False,
            table_output_format=None,
            extract_remote_api_key=resolved_api_key,
            page_elements_invoke_url=None,
            ocr_invoke_url=None,
            ocr_version="v2",
            graphic_elements_invoke_url=None,
            table_structure_invoke_url=None,
            pdf_split_batch_size=1,
            pdf_extract_batch_size=0,
            pdf_extract_tasks=0,
            pdf_extract_cpus_per_task=0.0,
            page_elements_actors=0,
            page_elements_batch_size=0,
            page_elements_cpus_per_actor=0.0,
            page_elements_gpus_per_actor=0.0,
            ocr_actors=0,
            ocr_batch_size=0,
            ocr_cpus_per_actor=0.0,
            ocr_gpus_per_actor=0.0,
            nemotron_parse_actors=0,
            nemotron_parse_batch_size=0,
            nemotron_parse_gpus_per_actor=0.0,
        )
        embed_params = _build_embed_params(
            embed_model_name=resolve_embed_model(embedding_model),
            embed_invoke_url=inference_config.embed_invoke_url,
            embed_remote_api_key=resolved_api_key,
            embed_modality="text",
            text_elements_modality=None,
            structured_elements_modality=None,
            embed_granularity="element",
            embed_actors=0,
            embed_batch_size=0,
            embed_cpus_per_actor=0.0,
            embed_gpus_per_actor=0.0 if inference_config.endpoint_mode == "remote" else None,
            local_ingest_embed_backend="hf",
        )
        text_chunk_params = TextChunkParams(
            max_tokens=int(text_chunk_max_tokens),
            overlap_tokens=int(text_chunk_overlap_tokens),
        )

        ingestor = _build_ingestor(
            run_mode="inprocess",
            ray_address=None,
            file_patterns=[str(doc.path) for doc in docs],
            input_type=input_type,
            extract_params=extract_params,
            embed_params=embed_params,
            text_chunk_params=text_chunk_params,
            enable_text_chunk=True,
            enable_dedup=False,
            enable_caption=False,
            dedup_iou_threshold=0.45,
            caption_invoke_url=None,
            caption_remote_api_key=resolved_api_key,
            caption_model_name="nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
            caption_device=None,
            caption_context_text_max_chars=0,
            caption_gpu_memory_utilization=0.5,
            caption_gpus_per_actor=0.0,
            caption_temperature=1.0,
            caption_top_p=None,
            caption_max_tokens=1024,
            store_images_uri=None,
            segment_audio=False,
            audio_split_type="size",
            audio_split_interval=500000,
            video_extract_audio=False,
            video_extract_frames=False,
            video_frame_fps=0.5,
            video_frame_dedup=False,
            video_frame_text_dedup=False,
            video_frame_text_dedup_max_dropped_frames=2,
            video_av_fuse=False,
        )
        raw_result = ingestor.ingest()
        records, result_df, _download_secs, _input_units = _collect_results("inprocess", raw_result)
        uploadable = _count_uploadable_vdb_records(records)
        if uploadable:
            _upload_vdb_records(
                records,
                vdb_op="lancedb",
                vdb_kwargs={
                    "uri": str(_lancedb_uri(index_path)),
                    "table_name": DEFAULT_LANCEDB_TABLE,
                    "overwrite": first_upload,
                    "hybrid": False,
                },
            )
            first_upload = False
            processed_paths.update(str(doc.path) for doc in docs)

        row_count = int(len(getattr(result_df, "index", [])))
        total_rows += row_count
        total_uploadable += uploadable
        group_summaries.append(
            {
                "input_type": input_type,
                "documents": len(docs),
                "rows": row_count,
                "uploadable_chunks": uploadable,
            }
        )

    if discovery.documents and total_uploadable == 0:
        raise LocalSearchError(
            "Ingestion completed but produced no uploadable chunks. Check document contents and inference settings.",
            code="no_uploadable_chunks",
        )

    table_info = _table_info(index_path, DEFAULT_LANCEDB_TABLE)
    indexed_chunks = (
        int(table_info.get("row_count") or 0) if table_info["readable"] and table_info["table_exists"] else 0
    )
    if discovery.documents and indexed_chunks == 0:
        raise LocalSearchError(
            "Ingestion produced candidate chunks, but no rows were indexed in LanceDB. "
            "Check embedding endpoint/authentication and vector-dimension warnings in the logs.",
            code="no_indexed_chunks",
        )
    return {
        "documents_processed": len(processed_paths),
        "pipeline_rows": total_rows,
        "uploadable_chunks": total_uploadable,
        "chunk_count": indexed_chunks,
        "groups": group_summaries,
    }


def _summarize_pipeline_log(raw: str) -> list[str]:
    warnings: list[str] = []
    interesting = (
        "Embedding failed",
        "Unauthorized",
        "Forbidden",
        "dropped_",
        "Skipping LanceDB",
    )
    for line in raw.replace("\r", "\n").splitlines():
        clean = line.strip()
        if clean and any(token in clean for token in interesting):
            warnings.append(clean[:500])
    return warnings[-6:]


def _run_ingestion_for_output(*, output: str, **kwargs: Any) -> dict[str, Any]:
    if _normalize_output(output) != "json":
        return _run_ingestion(**kwargs)

    sys.stdout.flush()
    sys.stderr.flush()
    with tempfile.TemporaryFile() as captured:
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        error: LocalSearchError | None = None
        result: dict[str, Any] | None = None
        try:
            os.dup2(captured.fileno(), 1)
            os.dup2(captured.fileno(), 2)
            result = _run_ingestion(**kwargs)
        except LocalSearchError as exc:
            error = exc
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)

        captured.seek(0)
        log_text = captured.read().decode("utf-8", errors="replace")

    if error is not None:
        warnings = error.warnings + _summarize_pipeline_log(log_text)
        if "401 Unauthorized" in log_text and "integrate.api.nvidia.com" in log_text:
            raise LocalSearchError(
                "Remote embedding endpoint rejected the configured API key. "
                "Set a valid NVIDIA_API_KEY/NGC_API_KEY, pass --api-key, or run with --inference local "
                "from an environment that has nemo-retriever[local] installed.",
                code="remote_embedding_unauthorized",
                warnings=warnings,
            ) from error
        raise LocalSearchError(str(error), code=error.code, warnings=warnings) from error
    assert result is not None
    return result


def _search_index_for_output(*, output: str, **kwargs: Any) -> dict[str, Any]:
    if _normalize_output(output) != "json":
        return _search_index(**kwargs)

    sys.stdout.flush()
    sys.stderr.flush()
    with tempfile.TemporaryFile() as captured:
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        error: Exception | None = None
        result: dict[str, Any] | None = None
        try:
            os.dup2(captured.fileno(), 1)
            os.dup2(captured.fileno(), 2)
            result = _search_index(**kwargs)
        except Exception as exc:
            error = exc
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)

        captured.seek(0)
        log_text = captured.read().decode("utf-8", errors="replace")

    captured_warnings = _summarize_pipeline_log(log_text)
    if error is not None:
        if isinstance(error, LocalSearchError):
            raise LocalSearchError(
                str(error),
                code=error.code,
                warnings=error.warnings + captured_warnings,
            ) from error
        if captured_warnings:
            raise LocalSearchError(
                str(error),
                code=type(error).__name__,
                warnings=captured_warnings,
            ) from error
        raise error
    assert result is not None
    if captured_warnings:
        result["warnings"] = list(result.get("warnings") or []) + captured_warnings
    return result


def _init_payload(
    *,
    discovery: Discovery,
    index_path: Path,
    manifest: dict[str, Any],
    dry_run: bool,
    ingestion: dict[str, Any] | None,
) -> dict[str, Any]:
    next_command = f'retriever local search "<query>" --index {index_path}'
    return {
        "command": "init",
        "ok": True,
        "dry_run": dry_run,
        "index_path": str(_resolve_index(index_path)),
        "lancedb_uri": str(_lancedb_uri(index_path)),
        "lancedb_table": DEFAULT_LANCEDB_TABLE,
        "corpus_root": str(discovery.corpus_root),
        "documents_discovered": manifest["documents_discovered"],
        "documents_processed": manifest["documents_processed"],
        "documents": [doc.to_manifest() for doc in discovery.documents],
        "documents_skipped": discovery.skipped,
        "chunk_count": manifest["chunk_count"],
        "embedding_model": manifest["embedding_model"],
        "inference": manifest["inference"],
        "groups": (ingestion or {}).get("groups", []),
        "warnings": manifest["warnings"],
        "next_recommended_command": next_command,
    }


def _format_init_text(payload: dict[str, Any]) -> str:
    lines = [
        f"Index: {payload['index_path']}",
        f"Documents selected: {len(payload['documents'])}",
        f"Documents processed: {payload['documents_processed']}",
        f"Chunks indexed: {payload['chunk_count']}",
    ]
    if payload["warnings"]:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in payload["warnings"])
    lines.append(f"Next: {payload['next_recommended_command']}")
    return "\n".join(lines)


@app.command("init")
def init(
    input_path: Path = typer.Argument(..., help="File or directory of local documents to index.", path_type=Path),
    index: Path = typer.Option(DEFAULT_INDEX, "--index", help="Local index directory.", path_type=Path),
    include: list[str] = typer.Option([], "--include", help="Glob of files to include; repeatable."),
    exclude: list[str] = typer.Option([], "--exclude", help="Glob of files to exclude; repeatable."),
    max_docs: int = typer.Option(DEFAULT_MAX_DOCS, "--max-docs", min=1, help="Safety cap for documents indexed."),
    max_pages: Optional[int] = typer.Option(None, "--max-pages", min=1, help="Optional safety cap for PDF pages."),
    inference: str = typer.Option("local", "--inference", help="Embedding inference mode: local, remote, or auto."),
    embed_invoke_url: Optional[str] = typer.Option(None, "--embed-invoke-url", help="Remote embedding endpoint URL."),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Remote NIM API key; defaults to env vars."),
    embedding_model: str = typer.Option(DEFAULT_EMBEDDING_MODEL, "--embedding-model", help="Embedding model name."),
    text_chunk_max_tokens: int = typer.Option(1024, "--text-chunk-max-tokens", min=1),
    text_chunk_overlap_tokens: int = typer.Option(150, "--text-chunk-overlap-tokens", min=0),
    dry_run: bool = typer.Option(False, "--dry-run", help="Discover files and emit planned work without indexing."),
    output: str = typer.Option("text", "--output", help="Output format: text or json."),
) -> None:
    """Create or update a local document index."""

    try:
        inference_config = _resolve_inference_config(
            inference=inference,
            embed_invoke_url=embed_invoke_url,
            api_key=api_key,
        )
        discovery = _discover_documents(
            input_path,
            include=include,
            exclude=exclude,
            max_docs=max_docs,
            max_pages=max_pages,
        )
        manifest = _manifest_base(
            discovery=discovery,
            index_path=index,
            inference_config=inference_config,
            embedding_model=embedding_model,
            warnings=list(discovery.warnings),
        )
        ingestion = None
        if dry_run:
            manifest["warnings"].append("Dry run only; no index was written.")
        elif discovery.documents:
            ingestion = _run_ingestion_for_output(
                output=output,
                discovery=discovery,
                index_path=index,
                inference_config=inference_config,
                embedding_model=embedding_model,
                api_key=api_key,
                text_chunk_max_tokens=text_chunk_max_tokens,
                text_chunk_overlap_tokens=text_chunk_overlap_tokens,
            )
            manifest["documents_processed"] = ingestion["documents_processed"]
            manifest["chunk_count"] = ingestion["chunk_count"]
            _write_manifest(index, manifest)
        else:
            _reset_lancedb_dir(index)
            _write_manifest(index, manifest)

        payload = _init_payload(
            discovery=discovery,
            index_path=index,
            manifest=manifest,
            dry_run=dry_run,
            ingestion=ingestion,
        )
        _echo_payload(payload, output=output, text_formatter=_format_init_text)
    except Exception as exc:
        _fail("init", exc, output=output)


def _parse_json_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _format_hit(hit: dict[str, Any], *, rank: int, show_context: bool) -> dict[str, Any]:
    metadata = _parse_json_dict(hit.get("metadata"))
    source_file = str(hit.get("source_id") or hit.get("source") or hit.get("path") or "")
    chunk_text = str(hit.get("text") or "")
    distance = hit.get("_distance")
    score = hit.get("_rerank_score", hit.get("_score"))
    return {
        "rank": rank,
        "source_file": source_file,
        "page": hit.get("page_number"),
        "section": metadata.get("section") or metadata.get("title"),
        "chunk_text": chunk_text if show_context else "",
        "score": score,
        "distance": distance,
        "metadata": metadata,
    }


def _search_index(
    *,
    query: str,
    index: Path,
    top_k: int,
    show_context: bool,
    inference: str | None,
    embed_invoke_url: str | None,
    api_key: str | None,
    embedding_model: str | None,
) -> dict[str, Any]:
    manifest = _load_manifest(index)
    table_name = str(manifest.get("lancedb_table") or DEFAULT_LANCEDB_TABLE)
    table_info = _table_info(index, table_name)
    stale = _detect_staleness(manifest)
    warnings = _warning_messages(manifest, stale, table_info)
    if not table_info["table_exists"]:
        raise LocalSearchError(
            f"Local index table {table_name!r} does not exist at {_lancedb_uri(index)}.",
            code="index_table_missing",
            warnings=warnings,
        )
    if int(table_info.get("row_count") or 0) == 0:
        raise LocalSearchError("Local index is empty.", code="index_empty", warnings=warnings)

    manifest_inference = manifest.get("inference") if isinstance(manifest.get("inference"), dict) else {}
    inference_config = _resolve_inference_config(
        inference=inference or str(manifest_inference.get("requested") or "auto"),
        embed_invoke_url=embed_invoke_url or manifest_inference.get("embed_invoke_url"),
        api_key=api_key,
    )
    resolved_model = resolve_embed_model(embedding_model or manifest.get("embedding_model") or DEFAULT_EMBEDDING_MODEL)
    if inference_config.endpoint_mode == "local":
        _validate_local_inference_available()

    from nemo_retriever.retriever import Retriever

    retriever = Retriever(
        vdb="lancedb",
        vdb_kwargs={"uri": str(_lancedb_uri(index)), "table_name": table_name},
        embedder=resolved_model,
        top_k=int(top_k),
        embedding_endpoint=inference_config.embed_invoke_url,
        embedding_use_grpc=False if inference_config.embed_invoke_url else None,
        embedding_api_key=resolve_remote_api_key(api_key) or "",
        reranker=False,
    )
    hits = retriever.query(query, top_k=int(top_k))
    results = [_format_hit(hit, rank=i + 1, show_context=show_context) for i, hit in enumerate(hits)]

    manifest["last_query_at"] = _utc_now()
    _write_manifest(index, manifest)
    return {
        "command": "search",
        "ok": True,
        "query": query,
        "index_metadata": {
            "index_path": str(_resolve_index(index)),
            "lancedb_uri": str(_lancedb_uri(index)),
            "lancedb_table": table_name,
            "chunk_count": int(table_info.get("row_count") or 0),
            "embedding_model": resolved_model,
            "inference": {
                "requested": inference_config.requested,
                "endpoint_mode": inference_config.endpoint_mode,
                "embed_invoke_url": inference_config.embed_invoke_url,
            },
        },
        "results": results,
        "warnings": warnings,
    }


def _can_reuse_index(
    *,
    manifest: dict[str, Any],
    discovery: Discovery,
    index: Path,
    embedding_model: str,
) -> bool:
    return bool(
        _index_reuse_decision(
            manifest=manifest,
            discovery=discovery,
            index=index,
            embedding_model=embedding_model,
        )["can_reuse"]
    )


def _index_reuse_decision(
    *,
    manifest: dict[str, Any],
    discovery: Discovery,
    index: Path,
    embedding_model: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    table_name = str(manifest.get("lancedb_table") or DEFAULT_LANCEDB_TABLE)
    table_info = _table_info(index, table_name)
    if not table_info["table_exists"]:
        reasons.append("index_table_missing")
    elif int(table_info.get("row_count") or 0) <= 0:
        reasons.append("index_empty")

    staleness = _detect_staleness(manifest)
    if staleness["changed_documents"]:
        reasons.append("documents_changed")
    if staleness["missing_documents"]:
        reasons.append("documents_missing")
    if staleness["new_supported_documents"]:
        reasons.append("new_supported_documents")

    existing_paths = [str(doc.get("path")) for doc in manifest.get("documents", [])]
    discovered_paths = [str(doc.path) for doc in discovery.documents]
    if existing_paths != discovered_paths:
        reasons.append("document_selection_changed")
    if str(manifest.get("input_path")) != str(discovery.input_path):
        reasons.append("input_path_changed")
    if list(manifest.get("include") or []) != discovery.include:
        reasons.append("include_filter_changed")
    if list(manifest.get("exclude") or []) != discovery.exclude:
        reasons.append("exclude_filter_changed")
    if resolve_embed_model(str(manifest.get("embedding_model") or "")) != resolve_embed_model(embedding_model):
        reasons.append("embedding_model_changed")

    return {
        "can_reuse": not reasons,
        "reasons": reasons,
        "staleness": staleness,
        "table": {
            "table_exists": bool(table_info["table_exists"]),
            "row_count": int(table_info.get("row_count") or 0),
        },
    }


def _format_search_text(payload: dict[str, Any]) -> str:
    lines = [
        f"Query: {payload['query']}",
        f"Index: {payload['index_metadata']['index_path']}",
        f"Results: {len(payload['results'])}",
    ]
    for result in payload["results"]:
        source = result.get("source_file") or "<unknown source>"
        page = result.get("page")
        where = f"{source}" + (f" page {page}" if page not in (None, -1) else "")
        score = result.get("score")
        distance = result.get("distance")
        metric = f" score={score}" if score is not None else (f" distance={distance}" if distance is not None else "")
        lines.append(f"{result['rank']}. {where}{metric}")
        if result.get("chunk_text"):
            lines.append(str(result["chunk_text"])[:1000])
    if payload["warnings"]:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in payload["warnings"])
    return "\n".join(lines)


@app.command("search")
def search(
    query: str = typer.Argument(..., help="Natural language search query."),
    index: Path = typer.Option(DEFAULT_INDEX, "--index", help="Local index directory.", path_type=Path),
    top_k: int = typer.Option(DEFAULT_TOP_K, "--top-k", min=1, help="Number of passages to retrieve."),
    output: str = typer.Option("text", "--output", help="Output format: text or json."),
    show_context: bool = typer.Option(False, "--show-context", help="Show retrieved chunk text in text output."),
    inference: Optional[str] = typer.Option(None, "--inference", help="Override query embedding mode."),
    embed_invoke_url: Optional[str] = typer.Option(None, "--embed-invoke-url", help="Remote embedding endpoint URL."),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Remote NIM API key; defaults to env vars."),
    embedding_model: Optional[str] = typer.Option(None, "--embedding-model", help="Override embedding model."),
) -> None:
    """Search an existing local document index."""

    try:
        payload = _search_index_for_output(
            output=output,
            query=query,
            index=index,
            top_k=top_k,
            show_context=show_context or _normalize_output(output) == "json",
            inference=inference,
            embed_invoke_url=embed_invoke_url,
            api_key=api_key,
            embedding_model=embedding_model,
        )
        if _normalize_output(output) == "text" and not show_context:
            for result in payload["results"]:
                result["chunk_text"] = ""
        _echo_payload(payload, output=output, text_formatter=_format_search_text)
    except Exception as exc:
        _fail("search", exc, output=output)


def ask_documents(
    *,
    input_path: Path,
    query: str,
    index: Path = DEFAULT_INDEX,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    max_docs: int = DEFAULT_MAX_DOCS,
    max_pages: int | None = None,
    inference: str = "local",
    embed_invoke_url: str | None = None,
    api_key: str | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    top_k: int = DEFAULT_TOP_K,
    output: str = "json",
    show_context: bool = True,
) -> dict[str, Any]:
    """Index if needed, then return retrieved evidence and index metadata."""

    inference_config = _resolve_inference_config(
        inference=inference,
        embed_invoke_url=embed_invoke_url,
        api_key=api_key,
    )
    discovery = _discover_documents(
        input_path,
        include=include or [],
        exclude=exclude or [],
        max_docs=max_docs,
        max_pages=max_pages,
    )
    manifest = _manifest_base(
        discovery=discovery,
        index_path=index,
        inference_config=inference_config,
        embedding_model=embedding_model,
        warnings=list(discovery.warnings),
    )
    index_action = "created"
    indexed_now = False
    reused_index = False
    reindexed = False
    reindex_reasons: list[str] = []
    try:
        existing_manifest = _load_manifest(index)
    except LocalSearchError as exc:
        existing_manifest = None
        reindex_reasons = [exc.code]
    if existing_manifest is not None:
        reuse_decision = _index_reuse_decision(
            manifest=existing_manifest,
            discovery=discovery,
            index=index,
            embedding_model=embedding_model,
        )
        if reuse_decision["can_reuse"]:
            manifest = existing_manifest
            manifest.setdefault("warnings", [])
            index_action = "reused"
            reused_index = True
            reindex_reasons = []
        else:
            index_action = "reindexed"
            reindexed = True
            reindex_reasons = list(reuse_decision["reasons"])

    if reused_index:
        pass
    elif discovery.documents:
        ingestion = _run_ingestion_for_output(
            output=output,
            discovery=discovery,
            index_path=index,
            inference_config=inference_config,
            embedding_model=embedding_model,
            api_key=api_key,
            text_chunk_max_tokens=1024,
            text_chunk_overlap_tokens=150,
        )
        manifest["documents_processed"] = ingestion["documents_processed"]
        manifest["chunk_count"] = ingestion["chunk_count"]
        _write_manifest(index, manifest)
        indexed_now = True
    else:
        index_action = "empty"
        _reset_lancedb_dir(index)
        _write_manifest(index, manifest)

    search_payload = _search_index_for_output(
        output=output,
        query=query,
        index=index,
        top_k=top_k,
        show_context=show_context or _normalize_output(output) == "json",
        inference=inference,
        embed_invoke_url=embed_invoke_url,
        api_key=api_key,
        embedding_model=embedding_model,
    )
    return {
        "command": "ask",
        "ok": True,
        "answer": None,
        "answer_generation": "not_configured",
        "query": query,
        "input_path": str(input_path),
        "resolved_input_path": str(discovery.input_path),
        "corpus_root": str(discovery.corpus_root),
        "index_path": str(_resolve_index(index)),
        "index_action": index_action,
        "indexed_now": indexed_now,
        "reused_index": reused_index,
        "reindexed": reindexed,
        "reindex_reasons": reindex_reasons,
        "evidence": search_payload["results"],
        "index_metadata": search_payload["index_metadata"],
        "warnings": search_payload["warnings"],
    }


@app.command("ask")
def ask(
    input_path: Path = typer.Argument(..., help="File or directory of local documents to index.", path_type=Path),
    query: str = typer.Argument(..., help="Natural language question."),
    index: Path = typer.Option(DEFAULT_INDEX, "--index", help="Local index directory.", path_type=Path),
    include: list[str] = typer.Option([], "--include", help="Glob of files to include; repeatable."),
    exclude: list[str] = typer.Option([], "--exclude", help="Glob of files to exclude; repeatable."),
    max_docs: int = typer.Option(DEFAULT_MAX_DOCS, "--max-docs", min=1),
    max_pages: Optional[int] = typer.Option(None, "--max-pages", min=1),
    inference: str = typer.Option("local", "--inference", help="Embedding inference mode: local, remote, or auto."),
    embed_invoke_url: Optional[str] = typer.Option(None, "--embed-invoke-url", help="Remote embedding endpoint URL."),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Remote NIM API key; defaults to env vars."),
    embedding_model: str = typer.Option(DEFAULT_EMBEDDING_MODEL, "--embedding-model", help="Embedding model name."),
    top_k: int = typer.Option(DEFAULT_TOP_K, "--top-k", min=1, help="Number of passages to retrieve."),
    output: str = typer.Option("text", "--output", help="Output format: text or json."),
    show_context: bool = typer.Option(False, "--show-context", help="Show retrieved chunk text in text output."),
) -> None:
    """Index if needed, then search. First pass returns cited evidence, not generated prose."""

    try:
        payload = ask_documents(
            input_path=input_path,
            query=query,
            index=index,
            include=include,
            exclude=exclude,
            max_docs=max_docs,
            max_pages=max_pages,
            inference=inference,
            embed_invoke_url=embed_invoke_url,
            api_key=api_key,
            embedding_model=embedding_model,
            top_k=top_k,
            output=output,
            show_context=show_context,
        )
        if _normalize_output(output) == "text" and not show_context:
            for result in payload["evidence"]:
                result["chunk_text"] = ""
        _echo_payload(payload, output=output, text_formatter=_format_ask_text)
    except Exception as exc:
        _fail("ask", exc, output=output)


def _format_ask_text(payload: dict[str, Any]) -> str:
    lines = [
        f"Question: {payload['query']}",
        f"Index action: {payload.get('index_action', 'unknown')}",
        "Answer generation is not configured; showing retrieved evidence.",
    ]
    if payload.get("reindex_reasons"):
        lines.append("Reindex reasons: " + ", ".join(str(reason) for reason in payload["reindex_reasons"]))
    for result in payload["evidence"]:
        source = result.get("source_file") or "<unknown source>"
        page = result.get("page")
        where = f"{source}" + (f" page {page}" if page not in (None, -1) else "")
        lines.append(f"{result['rank']}. {where}")
        if result.get("chunk_text"):
            lines.append(str(result["chunk_text"])[:1000])
    if payload["warnings"]:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in payload["warnings"])
    return "\n".join(lines)


def _status_payload(index: Path) -> dict[str, Any]:
    manifest = _load_manifest(index)
    table_name = str(manifest.get("lancedb_table") or DEFAULT_LANCEDB_TABLE)
    table_info = _table_info(index, table_name)
    stale = _detect_staleness(manifest)
    warnings = _warning_messages(manifest, stale, table_info)
    docs = manifest.get("documents") if isinstance(manifest.get("documents"), list) else []
    skipped = manifest.get("documents_skipped") if isinstance(manifest.get("documents_skipped"), list) else []
    if table_info["readable"] and table_info["table_exists"]:
        chunk_count = int(table_info.get("row_count") or 0)
    else:
        chunk_count = int(manifest.get("chunk_count") or 0)
    return {
        "command": "status",
        "ok": True,
        "index_path": str(_resolve_index(index)),
        "corpus_root": manifest.get("corpus_root"),
        "created_at": manifest.get("created_at"),
        "updated_at": manifest.get("updated_at"),
        "last_query_at": manifest.get("last_query_at"),
        "documents": {
            "processed": int(manifest.get("documents_processed") or 0),
            "configured": len(docs),
            "skipped": len(skipped),
            "supported_file_count": len(docs),
            "unsupported_file_count": sum(1 for item in skipped if item.get("reason") == "unsupported_format"),
        },
        "chunks": chunk_count,
        "embedding_model": manifest.get("embedding_model"),
        "vector_dimensions": table_info.get("vector_dimensions"),
        "inference": manifest.get("inference"),
        "lancedb": table_info,
        "staleness": stale,
        "health_checks": _health_checks(manifest, table_info, stale),
        "warnings": warnings,
    }


def _format_status_text(payload: dict[str, Any]) -> str:
    lines = [
        f"Index: {payload['index_path']}",
        f"Corpus: {payload.get('corpus_root')}",
        f"Documents processed: {payload['documents']['processed']}",
        f"Chunks: {payload['chunks']}",
        f"Updated: {payload.get('updated_at')}",
    ]
    for check in payload["health_checks"]:
        lines.append(f"{check['status'].upper()}: {check['name']} - {check['message']}")
    if payload["warnings"]:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in payload["warnings"])
    return "\n".join(lines)


@app.command("status")
def status(
    index: Path = typer.Option(DEFAULT_INDEX, "--index", help="Local index directory.", path_type=Path),
    output: str = typer.Option("text", "--output", help="Output format: text or json."),
) -> None:
    """Inspect a local index and its health checks."""

    try:
        payload = _status_payload(index)
        _echo_payload(payload, output=output, text_formatter=_format_status_text)
    except Exception as exc:
        _fail("status", exc, output=output)


def _doctor_check(name: str, status: str, message: str, **extra: Any) -> dict[str, Any]:
    payload = {"name": name, "status": status, "message": message}
    payload.update(extra)
    return payload


def _doctor_payload(
    *,
    index: Path,
    inference: str,
    input_path: Path | None,
    embed_invoke_url: str | None,
    api_key: str | None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    checks.append(
        _doctor_check(
            "python",
            "ok",
            f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            executable=sys.executable,
        )
    )
    try:
        from nemo_retriever.version import get_version_info

        info = get_version_info()
        checks.append(_doctor_check("nemo_retriever_version", "ok", str(info.get("full_version") or info)))
    except Exception as exc:
        checks.append(_doctor_check("nemo_retriever_version", "warning", f"{type(exc).__name__}: {exc}"))

    try:
        import lancedb  # type: ignore

        version = getattr(lancedb, "__version__", "unknown")
        checks.append(_doctor_check("lancedb_import", "ok", f"lancedb import succeeded (version={version})."))
    except Exception as exc:
        checks.append(_doctor_check("lancedb_import", "error", f"{type(exc).__name__}: {exc}"))

    normalized_inference = _normalize_inference(inference)
    endpoint = (embed_invoke_url or "").strip() or None
    if normalized_inference == "remote" and endpoint is None:
        endpoint = DEFAULT_REMOTE_EMBED_ENDPOINT
    if endpoint:
        key = resolve_remote_api_key(api_key)
        status_value = "ok" if key else "error"
        checks.append(
            _doctor_check(
                "remote_api_key",
                status_value,
                (
                    "Remote API key is configured."
                    if key
                    else "Remote inference needs NVIDIA_API_KEY, NGC_API_KEY, or --api-key."
                ),
            )
        )
        checks.append(_doctor_check("remote_embedding_endpoint", "ok", endpoint))
    elif normalized_inference == "local":
        try:
            import torch  # type: ignore

            cuda_available = bool(torch.cuda.is_available())
            checks.append(
                _doctor_check(
                    "cuda",
                    "ok" if cuda_available else "warning",
                    "CUDA is available." if cuda_available else "CUDA is not available; local GPU inference may fail.",
                )
            )
        except Exception as exc:
            checks.append(_doctor_check("cuda", "warning", f"Could not import torch: {type(exc).__name__}: {exc}"))

    manifest_exists = _manifest_path(index).exists()
    if manifest_exists:
        try:
            manifest = _load_manifest(index)
            table_name = str(manifest.get("lancedb_table") or DEFAULT_LANCEDB_TABLE)
            table_info = _table_info(index, table_name)
            checks.append(
                _doctor_check(
                    "index_readable",
                    "ok" if table_info["readable"] and table_info["table_exists"] else "error",
                    (
                        f"LanceDB rows: {table_info.get('row_count', 0)}"
                        if table_info["readable"]
                        else str(table_info.get("error"))
                    ),
                )
            )
        except Exception as exc:
            checks.append(_doctor_check("index_readable", "error", f"{type(exc).__name__}: {exc}"))
    else:
        checks.append(_doctor_check("index_readable", "warning", "No local-search manifest exists yet."))

    if input_path is not None:
        try:
            discovery = _discover_documents(
                input_path,
                include=[],
                exclude=[],
                max_docs=DEFAULT_MAX_DOCS,
                max_pages=None,
            )
            checks.append(
                _doctor_check(
                    "sample_ingestion_dry_run",
                    "ok" if discovery.documents else "warning",
                    f"Selected {len(discovery.documents)} supported document(s); skipped {len(discovery.skipped)}.",
                )
            )
        except Exception as exc:
            checks.append(_doctor_check("sample_ingestion_dry_run", "error", f"{type(exc).__name__}: {exc}"))
    else:
        checks.append(
            _doctor_check(
                "sample_ingestion_dry_run",
                "skipped",
                "Pass --input-path to run document discovery without indexing.",
            )
        )

    has_error = any(check["status"] == "error" for check in checks)
    return {
        "command": "doctor",
        "ok": not has_error,
        "index_path": str(_resolve_index(index)),
        "checks": checks,
        "warnings": [check["message"] for check in checks if check["status"] == "warning"],
    }


def _format_doctor_text(payload: dict[str, Any]) -> str:
    lines = [f"Index: {payload['index_path']}"]
    lines.extend(f"{check['status'].upper()}: {check['name']} - {check['message']}" for check in payload["checks"])
    return "\n".join(lines)


@app.command("doctor")
def doctor(
    index: Path = typer.Option(DEFAULT_INDEX, "--index", help="Local index directory.", path_type=Path),
    inference: str = typer.Option("local", "--inference", help="Inference mode to validate: local, remote, or auto."),
    input_path: Optional[Path] = typer.Option(None, "--input-path", help="Optional sample path for dry-run discovery."),
    embed_invoke_url: Optional[str] = typer.Option(None, "--embed-invoke-url", help="Remote embedding endpoint URL."),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Remote NIM API key; defaults to env vars."),
    output: str = typer.Option("text", "--output", help="Output format: text or json."),
) -> None:
    """Diagnose local-search configuration and index readability."""

    payload = _doctor_payload(
        index=index,
        inference=inference,
        input_path=input_path,
        embed_invoke_url=embed_invoke_url,
        api_key=api_key,
    )
    _echo_payload(payload, output=output, text_formatter=_format_doctor_text)
    if not payload["ok"]:
        raise typer.Exit(code=1)


@app.command("clean")
def clean(
    index: Path = typer.Option(DEFAULT_INDEX, "--index", help="Local index directory.", path_type=Path),
    yes: bool = typer.Option(False, "--yes", help="Delete the local index without prompting."),
) -> None:
    """Delete a local index directory."""

    index_path = _resolve_index(index)
    if not index_path.exists():
        typer.echo(f"Index does not exist: {index_path}")
        return
    if not yes:
        raise typer.BadParameter("Pass --yes to delete the local index.")
    shutil.rmtree(index_path)
    typer.echo(f"Deleted local index: {index_path}")
