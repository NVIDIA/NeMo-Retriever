# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable, resumable Parquet staging for Ray batch ingest.

Input files are sorted and grouped into shards. Each shard runs as one Ray
execution whose terminal VDB upload is staged as Parquet by write tasks (refer
to :mod:`nemo_retriever.common.vdb.staged_parquet`). A shard commits only
after every part is verified, its staging directory is renamed into place, and
an fsynced record is appended to ``manifest.jsonl``. Rerunning with the same
staging directory and inputs skips committed shards and rewrites the rest.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import shutil
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pyarrow.parquet as pq

from nemo_retriever.common.vdb.arrow import (
    EMBEDDING_MODEL_METADATA_KEY,
    EMBEDDING_MODEL_REVISION_METADATA_KEY,
    cached_vector_dimension,
)
from nemo_retriever.common.vdb.records import VdbUploadError, _raise_for_empty_vdb_conversion
from nemo_retriever.common.vdb.staged_parquet import (
    PART_SUFFIX,
    SHARDS_DIRNAME,
    StageTarget,
    file_sha256,
    find_duplicate_ids,
    fsync_dir,
    shard_dir,
)

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.jsonl"
MANIFEST_VERSION = 1
_LOCK_NAME = ".lock"
_MAX_REPORTED_ITEMS = 20
# Fields that bind a staging directory to one input set and stage configuration.
_HEADER_IDENTITY_FIELDS = ("version", "shard_files", "inputs_sha256", "settings_sha256")


class StagingError(RuntimeError):
    """A staging directory or shard failed a durability or reconciliation check."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StagedInput:
    """One input file and the attributes that identify its content for resume."""

    document: str
    path: str
    size: int
    mtime_ns: int

    def as_json(self) -> dict[str, Any]:
        return {"document": self.document, "path": self.path, "size": self.size, "mtime_ns": self.mtime_ns}


@dataclass(frozen=True)
class Shard:
    """A deterministic group of input files that stages and commits as one unit."""

    index: int
    shard_id: str
    inputs: tuple[StagedInput, ...]

    @property
    def documents(self) -> list[str]:
        return [item.document for item in self.inputs]


def plan_shards(documents: Sequence[str], shard_files: int) -> list[Shard]:
    """Sort input files and group them into shards of at most ``shard_files`` files.

    Shard IDs derive from the shard's position and its files' resolved paths,
    sizes, and modification times, so the same inputs always give the same
    shards. Missing and repeated files fail before any work starts.
    """
    if isinstance(shard_files, bool) or not isinstance(shard_files, int) or shard_files <= 0:
        raise ValueError("shard_files must be a positive integer")
    inputs: dict[str, StagedInput] = {}
    for document in documents:
        path = os.path.realpath(os.fspath(document))
        if not os.path.isfile(path):
            raise StagingError(f"Parquet staging requires local input files; {document!r} is not a file")
        if path in inputs:
            raise StagingError(f"Input file {path!r} appears more than once")
        stat = os.stat(path)
        inputs[path] = StagedInput(document=str(document), path=path, size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    ordered = [inputs[path] for path in sorted(inputs)]
    shards = []
    for index, start in enumerate(range(0, len(ordered), shard_files)):
        members = tuple(ordered[start : start + shard_files])
        digest = _sha256_json([item.as_json() for item in members])
        shards.append(Shard(index=index, shard_id=f"{index:06d}-{digest[:16]}", inputs=members))
    return shards


class StagingManifest:
    """Append-only JSONL ledger of one staging directory.

    The first line is the header that binds the directory to one input set
    and configuration. Each later line is fsynced before it counts. A final
    line without a newline is a torn append from a crash; it is truncated on
    open, which leaves its shard uncommitted.
    """

    def __init__(self, path: str, records: list[dict[str, Any]]) -> None:
        self.path = path
        self.records = records

    @property
    def header(self) -> dict[str, Any]:
        return self.records[0]

    @property
    def committed_shards(self) -> dict[str, dict[str, Any]]:
        return {record["shard_id"]: record for record in self.records if record.get("type") == "shard"}

    @classmethod
    def open(cls, stage_dir: str, header: Mapping[str, Any]) -> StagingManifest:
        path = os.path.join(stage_dir, MANIFEST_NAME)
        if not os.path.exists(path):
            unexpected = sorted(name for name in os.listdir(stage_dir) if name != _LOCK_NAME)
            if unexpected:
                raise StagingError(
                    f"Staging directory {stage_dir!r} has no {MANIFEST_NAME} but is not empty: {unexpected[:5]}"
                )
            record = {"type": "header", "created_at": _now(), **header}
            temp_path = f"{path}.{uuid.uuid4().hex}.tmp"
            with open(temp_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
            fsync_dir(stage_dir)
            return cls(path, [record])

        manifest = cls(path, cls._read(path))
        existing = manifest.header
        if existing.get("type") != "header":
            raise StagingError(f"{path!r} does not start with a staging header")
        changed = [name for name in _HEADER_IDENTITY_FIELDS if existing.get(name) != header.get(name)]
        if changed:
            raise StagingError(
                f"Staging directory {stage_dir!r} was created for different inputs or settings "
                f"(changed: {', '.join(changed)}). Rerun with the original inputs and options, or use a new "
                "staging directory."
            )
        return manifest

    @staticmethod
    def _read(path: str) -> list[dict[str, Any]]:
        with open(path, "rb") as handle:
            data = handle.read()
        complete_length = data.rfind(b"\n") + 1
        if complete_length != len(data):
            logger.warning("Truncating a torn final record in %s", path)
            with open(path, "r+b") as handle:
                handle.truncate(complete_length)
                handle.flush()
                os.fsync(handle.fileno())
        records = []
        for number, line in enumerate(data[:complete_length].splitlines(), start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise StagingError(f"{path!r} line {number} is not valid JSON") from exc
        if not records:
            raise StagingError(f"{path!r} has no staging header")
        shard_ids = [record["shard_id"] for record in records if record.get("type") == "shard"]
        repeated = sorted(shard_id for shard_id, count in Counter(shard_ids).items() if count > 1)
        if repeated:
            raise StagingError(f"{path!r} commits shards more than once: {repeated[:5]}")
        return records

    def append(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(dict(record), sort_keys=True, default=str) + "\n"
        with open(self.path, "ab") as handle:
            handle.write(line.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        self.records.append(dict(record))


@contextlib.contextmanager
def _exclusive_lock(stage_dir: str) -> Iterator[None]:
    with open(os.path.join(stage_dir, _LOCK_NAME), "a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StagingError(f"Another process is using staging directory {stage_dir!r}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def verify_shard_parts(
    stage_dir: str, directory: str, parts: Sequence[Mapping[str, Any]], *, checksums: bool = False
) -> list[str]:
    """Return problems that make a shard's parts differ from their recorded metadata."""
    problems = []
    expected = {os.path.basename(part["path"]) for part in parts}
    present = (
        {name for name in os.listdir(directory) if name.endswith(PART_SUFFIX)} if os.path.isdir(directory) else set()
    )
    for name in sorted(present - expected):
        problems.append(f"unrecorded part {name}")
    for part in parts:
        path = os.path.join(stage_dir, part["path"])
        if not os.path.isfile(path):
            problems.append(f"missing part {part['path']}")
            continue
        if os.path.getsize(path) != part["bytes"]:
            problems.append(f"part {part['path']} has {os.path.getsize(path)} bytes, expected {part['bytes']}")
            continue
        rows = pq.read_metadata(path).num_rows
        if rows != part["rows"]:
            problems.append(f"part {part['path']} has {rows} rows, expected {part['rows']}")
        if checksums and file_sha256(path) != part["sha256"]:
            problems.append(f"part {part['path']} checksum differs from the manifest")
    return problems


def reconcile_stage_dir(stage_dir: str, manifest: StagingManifest, planned_shard_ids: Iterable[str]) -> list[str]:
    """Delete everything uncommitted under ``shards/`` and verify committed shards.

    Returns the names of the removed directories. A committed shard that no
    longer matches its record fails closed and names the shard.
    """
    committed = manifest.committed_shards
    unknown = sorted(set(committed) - set(planned_shard_ids))
    if unknown:
        raise StagingError(f"Manifest commits shards that the current inputs do not plan: {unknown[:5]}")
    shards_root = os.path.join(stage_dir, SHARDS_DIRNAME)
    removed = []
    if os.path.isdir(shards_root):
        for name in sorted(os.listdir(shards_root)):
            if name not in committed:
                shutil.rmtree(os.path.join(shards_root, name))
                removed.append(name)
        if removed:
            fsync_dir(shards_root)
            logger.info("Removed %d uncommitted staging directories: %s", len(removed), removed[:5])
    problems = {
        shard_id: issues
        for shard_id, record in committed.items()
        if (issues := verify_shard_parts(stage_dir, shard_dir(stage_dir, shard_id), record["parts"]))
    }
    if problems:
        detail = "; ".join(f"{shard_id}: {', '.join(issues[:3])}" for shard_id, issues in sorted(problems.items()))
        raise StagingError(f"Committed shards no longer match the manifest: {detail}")
    return removed


def _task_results(result: Any) -> list[dict[str, Any]]:
    if result is None:
        return []
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        return list(to_dict(orient="records"))
    return [dict(item) for item in result]


def _find_duplicate_ids(stage_dir: str, part_paths: list[str]) -> list[str]:
    """Check ID uniqueness in a Ray task so ID columns never load in the driver."""
    import ray

    if not part_paths:
        return []
    if not ray.is_initialized():
        return find_duplicate_ids(stage_dir, part_paths)
    return ray.get(ray.remote(num_cpus=0)(find_duplicate_ids).remote(stage_dir, part_paths))


def _part_identity(stage_dir: str, part_path: str) -> dict[str, Any]:
    schema = pq.read_schema(os.path.join(stage_dir, part_path))
    metadata = schema.metadata or {}
    return {
        "vector_dim": cached_vector_dimension(schema),
        "embedding_model_name": metadata.get(EMBEDDING_MODEL_METADATA_KEY, b"").decode("utf-8") or None,
        "embedding_model_revision": metadata.get(EMBEDDING_MODEL_REVISION_METADATA_KEY, b"").decode("utf-8") or None,
    }


def commit_shard(
    manifest: StagingManifest, shard: Shard, target: Any, results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Verify one attempt's write-task results, publish its directory, and record the shard.

    Order matters for durability: verify every part, rename the attempt
    directory to the shard directory, fsync, then append the manifest record.
    A crash before the append leaves the shard uncommitted.
    """
    stage_dir = target.stage_dir
    totals: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    upstream_fields: Counter[str] = Counter()
    duplicate_ids: list[str] = []
    stage_errors: list[dict[str, Any]] = []
    vector_dims = set()
    parts = []
    for result in results:
        for key in ("graph_rows", "records", "rows", "missing_embeddings", "upstream_errors", "stage_error_count"):
            totals[key] += int(result.get(key) or 0)
        dropped.update(result.get("dropped") or {})
        rejected.update(result.get("rejected") or {})
        upstream_fields.update(result.get("upstream_error_fields") or {})
        duplicate_ids.extend(result.get("duplicate_ids") or [])
        stage_errors.extend(result.get("stage_errors") or [])
        if result.get("rows"):
            vector_dims.add(result.get("vector_dim"))
        parts.extend(dict(part) for part in result.get("parts") or [])

    if totals["stage_error_count"]:
        sources = sorted({str(error.get("source_identifier")) for error in stage_errors})
        raise StagingError(
            f"Shard {shard.shard_id} has {totals['stage_error_count']} stage error(s); sources: {sources[:5]}"
        )
    if totals["missing_embeddings"]:
        raise VdbUploadError(
            "vdb_upload is refusing a partial write because searchable rows are missing embeddings: "
            f"shard={shard.shard_id}, input rows={totals['graph_rows']}, uploadable rows={totals['records']}, "
            f"missing embedding={totals['missing_embeddings']}."
        )
    if len(vector_dims) > 1:
        raise StagingError(f"Shard {shard.shard_id} produced vectors of several dimensions: {sorted(vector_dims)}")
    paths = [part["path"] for part in parts]
    repeated_paths = sorted(path for path, count in Counter(paths).items() if count > 1)
    if repeated_paths or duplicate_ids:
        raise StagingError(
            f"Shard {shard.shard_id} produced duplicate row IDs: {(duplicate_ids or repeated_paths)[:5]}. "
            "Rows with the same ID are the same row; refer to graph_row_identity()."
        )
    if sum(part["rows"] for part in parts) != totals["rows"]:
        raise StagingError(f"Shard {shard.shard_id} part rows do not add up to its staged rows")
    problems = verify_shard_parts(stage_dir, target.attempt_dir, parts)
    if problems:
        raise StagingError(f"Shard {shard.shard_id} parts failed verification: {', '.join(problems[:5])}")
    duplicates = _find_duplicate_ids(stage_dir, paths)
    if duplicates:
        raise StagingError(
            f"Shard {shard.shard_id} produced duplicate row IDs across parts: {duplicates[:5]}. "
            "Rows with the same ID are the same row; refer to graph_row_identity()."
        )
    identities = [_part_identity(stage_dir, path) for path in paths]
    if any(identity != identities[0] for identity in identities):
        raise StagingError(f"Shard {shard.shard_id} parts disagree on vector dimension or embedding model")

    final_dir = shard_dir(stage_dir, shard.shard_id)
    if os.path.isdir(target.attempt_dir):
        os.rename(target.attempt_dir, final_dir)
    else:
        os.makedirs(final_dir)
    fsync_dir(os.path.join(stage_dir, SHARDS_DIRNAME))
    for part in parts:
        part["path"] = os.path.relpath(os.path.join(final_dir, os.path.basename(part["path"])), stage_dir)

    record = {
        "type": "shard",
        "shard_id": shard.shard_id,
        "index": shard.index,
        "inputs": shard.documents,
        "parts": parts,
        "rows": totals["rows"],
        "graph_rows": totals["graph_rows"],
        "records": totals["records"],
        "upstream_errors": totals["upstream_errors"],
        "upstream_error_fields": dict(upstream_fields),
        "rejected": dict(rejected),
        "dropped": dict(dropped),
        **(identities[0] if identities else {"vector_dim": None}),
        "committed_at": _now(),
    }
    manifest.append(record)
    return record


@dataclass(frozen=True)
class StagingSummary:
    """Counts for one staging run over all planned shards."""

    stage_dir: str
    shards: int
    staged_shards: int
    skipped_shards: int
    removed_uncommitted: tuple[str, ...]
    graph_rows: int
    rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_dir": self.stage_dir,
            "shards": self.shards,
            "staged_shards": self.staged_shards,
            "skipped_shards": self.skipped_shards,
            "removed_uncommitted": list(self.removed_uncommitted),
            "graph_rows": self.graph_rows,
            "staged_rows": self.rows,
        }


def stage_documents(
    documents: Sequence[str],
    *,
    stage_dir: str | os.PathLike[str],
    shard_files: int,
    settings: Mapping[str, Any],
    run_shard: Callable[[Shard, Any], Any],
    rows_per_part: int = 8192,
    write_concurrency: int = 8,
    stage_error_columns: Sequence[str] = (),
) -> StagingSummary:
    """Stage every uncommitted shard, then check the run as the in-driver sink would.

    ``settings`` must contain everything that changes staged rows (pipeline
    and embedding configuration). ``run_shard(shard, target)`` executes the
    ingest graph for ``shard.documents`` with ``target`` as the Parquet
    staging target and returns the executor's per-task results.
    """
    stage_dir = os.path.abspath(os.fspath(stage_dir))
    shards = plan_shards(documents, shard_files)
    header = {
        "version": MANIFEST_VERSION,
        "shard_files": shard_files,
        "inputs_sha256": _sha256_json([[item.as_json() for item in shard.inputs] for shard in shards]),
        "settings_sha256": _sha256_json(settings),
        "settings": dict(settings),
        "n_inputs": sum(len(shard.inputs) for shard in shards),
        "n_shards": len(shards),
    }
    os.makedirs(stage_dir, exist_ok=True)
    with _exclusive_lock(stage_dir):
        manifest = StagingManifest.open(stage_dir, header)
        removed = reconcile_stage_dir(stage_dir, manifest, [shard.shard_id for shard in shards])
        staged = 0
        for shard in shards:
            if shard.shard_id in manifest.committed_shards:
                continue
            target = StageTarget(
                stage_dir=stage_dir,
                shard_id=shard.shard_id,
                attempt=uuid.uuid4().hex[:12],
                rows_per_part=rows_per_part,
                write_concurrency=write_concurrency,
                stage_error_columns=tuple(stage_error_columns),
            )
            logger.info(
                "Staging shard %d/%d (%s, %d files)", shard.index + 1, len(shards), shard.shard_id, len(shard.inputs)
            )
            record = commit_shard(manifest, shard, target, _task_results(run_shard(shard, target)))
            logger.info(
                "Committed shard %s with %d rows in %d parts", shard.shard_id, record["rows"], len(record["parts"])
            )
            staged += 1

        committed = manifest.committed_shards.values()
        graph_rows = sum(record["graph_rows"] for record in committed)
        rows = sum(record["rows"] for record in committed)
        if graph_rows and not sum(record["records"] for record in committed):
            _raise_for_empty_vdb_conversion(
                row_count=graph_rows,
                upstream_error_count=sum(record["upstream_errors"] for record in committed),
                upstream_error_fields=sum(
                    (Counter(record["upstream_error_fields"]) for record in committed), Counter()
                ),
                rejection_reasons=sum((Counter(record["rejected"]) for record in committed), Counter()),
            )
        return StagingSummary(
            stage_dir=stage_dir,
            shards=len(shards),
            staged_shards=staged,
            skipped_shards=len(shards) - staged,
            removed_uncommitted=tuple(removed),
            graph_rows=graph_rows,
            rows=rows,
        )
